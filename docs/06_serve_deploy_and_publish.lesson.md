# 06. From a Model File to a Service and Deployment

In Lesson 05, you saved the weights, configuration, and tokenizer in one file. Now you will build a program that reads that file once and answers text-generation requests. This step is called **model serving**. Making the program available for others to use is called **deployment**.

This lesson starts with a small request. You first test the program and its errors in the notebook, then build a runnable Docker image and a standalone project bundle. Server and publishing commands are run in a separate terminal. Running every cell does not create a permanent server or an internet deployment.

**Prerequisite:** [Lesson 05](05_train_evaluate_and_checkpoint.ipynb). For an independent run, you can use the bundled demo model; you do not need to train again. If the selected model from Lesson 05 exists in the current run directory, this lesson uses it and prints the exact path.

A service that runs successfully does not prove the model has high language quality. This small GPT generates text continuations; it is not a general assistant. Measure quality, latency, and serving capacity separately from the correctness of the program contract.

<!-- cell -->
## 1. Find the Environment and Select the Right Model

As in earlier lessons, you walk from the current directory through its parents to find the project. The `TINYLLM_RUN_DIR` environment variable points to the training results. If it is not set, the notebook uses `artifacts/runs/notebook`.

The presence of `selected.pt` means that a selected model bundle exists at that path; it does not mean that the model is high quality. If the file is absent, the bundled model at `artifacts/demo/best.pt` is used. The program must not start training when it receives its first request.

<!-- cell -->

```python
from pathlib import Path
import sys
import os
import json
import hashlib
import torch

ROOT = None
for candidate in [Path.cwd()] + list(Path.cwd().parents):
    if (candidate / "src" / "tinyllm").is_dir():
        ROOT = candidate
        break
if ROOT is None:
    raise RuntimeError("Open this notebook from inside tiny-llm-fa.")

source_path = str(ROOT / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)
torch.set_num_threads(2)

default_run_root = ROOT / "artifacts" / "runs" / "notebook"
run_root = Path(os.environ.get("TINYLLM_RUN_DIR", str(default_run_root)))
trained_path = run_root / "selected.pt"
if trained_path.is_file():
    CHECKPOINT = trained_path
    model_source = "selected model from notebook 05"
else:
    CHECKPOINT = ROOT / "artifacts" / "demo" / "best.pt"
    model_source = "bundled educational demo"

assert CHECKPOINT.is_file(), "Checkpoint file is missing."
print("Source:", model_source)
print("Checkpoint:", CHECKPOINT)
```

<!-- cell -->
**Goal of the next example:** Before building the service, load the selected file directly and generate a short continuation. This checks the model separately from HTTP concerns. If this step fails, first inspect the file path and model compatibility.

**Inference** means running the model to obtain a response. No gradients or weight updates are used here. The project loader puts the model into `eval` mode.

<!-- cell -->

```python
from tinyllm.artifacts import load_checkpoint
from tinyllm.model import generate

loaded_model, loaded_tokenizer, loaded_metadata = load_checkpoint(CHECKPOINT)
prompt = "The model "
prompt_ids = torch.tensor([loaded_tokenizer.encode(prompt)], dtype=torch.long)
output_ids = generate(loaded_model, prompt_ids, count=12, seed=7)
print("Context limit:", loaded_model.config.context)
print("Vocabulary:", loaded_tokenizer.vocab_size)
print("Training step:", loaded_metadata.get("step"))
print("Direct output:", loaded_tokenizer.decode(output_ids[0].tolist()))
```

<!-- cell -->
The output contains the initial text and the generated continuation. The context limit is the number of tokens the model can see in one computation. Because this tokenizer works at the character level, one known character is one token. Do not assume this equivalence for another model.

## 2. What Does an HTTP Request Contain?

An **API (Application Programming Interface)** is a contract through which one program asks another program to perform an operation. In this lesson, the request is sent over **HTTP**, the same protocol a browser uses to retrieve a page.

An **endpoint** is the part of a URL that identifies a capability. `/health` reports service health, while `/generate` generates text. `GET` reads health information and `POST` sends a generation request.

The generation request body is a **JSON** object. JSON is a text format for storing values such as strings, numbers, lists, and dictionaries. In Python, you create a dictionary and the request tool converts it to JSON.

**Goal of the example:** Inspect the request data before involving the network or FastAPI. Keep the output count small so the experiment stays quick.

<!-- cell -->

```python
request_body = {
    "prompt": "The model ",
    "max_new_tokens": 12,
    "temperature": 0.8,
    "top_k": 20,
    "seed": 7,
}
json_text = json.dumps(request_body)  # Convert the dictionary to JSON text.
restored_body = json.loads(json_text)  # Convert the JSON text back to Python data.
print("JSON:", json_text)
assert restored_body == request_body
```

<!-- cell -->
`prompt` is the starting text, `max_new_tokens` is the maximum number of new tokens, `temperature` controls sampling randomness, `top_k` limits the sampling candidates, and `seed` sets the sampling seed. The generation mechanism was covered in Lesson 04. The same seed helps compare samples in the same environment; exact reproduction across every CPU and GPU is not guaranteed.

## 3. Validate the Request Contract Before Calling the Model

**FastAPI** builds HTTP routes and responses. **Pydantic** validates request data against field types and constraints. Defining this contract lets the application reject invalid input before running the model.

The `GenerateRequest` class defines the request data. `Field` specifies minimums, maximums, and defaults. A default is used when the request omits a field. `extra="forbid"` rejects unknown fields, so a typo in an option name does not go unnoticed.

<!-- cell -->

```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    # This is the general string limit; the true context limit is checked after tokenization.
    prompt: str = Field(min_length=1, max_length=256)
    max_new_tokens: int = Field(default=80, ge=1, le=128)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_k: int = Field(default=20, ge=0, le=256)
    seed: int = Field(default=7, ge=0, le=2147483647)

validated_request = GenerateRequest(**request_body)  # Convert dictionary keys into named arguments.
print("Validated:", validated_request.model_dump())
defaults = GenerateRequest(prompt="The model ")
print("Defaults:", defaults.model_dump())

try:
    GenerateRequest(prompt="The model ", max_new_tokens=1000)
except ValidationError as error:
    print("Expected validation error:", error.errors()[0]["type"])
```

<!-- cell -->
The valid request contains all values. The second request takes omitted options from the defaults. A request for one thousand output tokens exceeds the limit of 128 and raises an error. This limit keeps the educational example affordable to run.

### A Declared Type Does Not Always Reject Every Conversion

This contract does not enable strict type handling. Pydantic can therefore coerce some inputs. In the installed project version, the numeric string `"12"` can become an integer, and the Boolean value `True` can become 1 for an integer field.

The example makes this behavior explicit so you do not claim that every nonnumeric value is rejected. If a future application needs a strict contract, define strict types and test their behavior; this project contract allows these conversions.

<!-- cell -->

```python
coerced = GenerateRequest(prompt="The model ", max_new_tokens="12", seed=True)
print("Converted token count:", coerced.max_new_tokens, type(coerced.max_new_tokens).__name__)
print("Converted boolean seed:", coerced.seed, type(coerced.seed).__name__)
assert coerced.max_new_tokens == 12 and coerced.seed == 1

for invalid_options in [{"temperature": float("inf")}, {"max_new_tokens": "many"}, {"typo": 1}]:
    try:
        GenerateRequest(prompt="The model ", **invalid_options)
    except ValidationError as error:
        print("Rejected:", invalid_options, "reason:", error.errors()[0]["type"])
```

<!-- cell -->
An infinite value, text that cannot be converted to a number, and an extra field are rejected. Here you saw the class error directly. In an HTTP request, FastAPI returns a contract error with status 422.

## 4. Load the Model Once When the Service Starts

A service has two distinct phases: application startup and request processing. The **lifespan** is where shared resources are prepared when the service starts. Load the weights there and keep the model in `app.state`; requests reuse the prepared model.

If you read the file for every request, loading time and extra memory use are added to each response. Loading at startup also exposes a missing file immediately. The service is not for training and does not change weights during a request.

**Goal of the next cell:** Build an application with only a health endpoint. Add the generation endpoint in the following cell so each part has a clear responsibility. The educational application will eventually use the same route contract as the main model.

The `@app.get(...)` decorator connects a function to a route. `@asynccontextmanager` defines the startup and shutdown lifecycle, and `yield` is the point where control is handed over for serving after preparation.

<!-- cell -->

```python
from contextlib import asynccontextmanager
import threading
import time
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

def create_lesson_app(checkpoint):
    path = Path(checkpoint)

    @asynccontextmanager
    async def lifespan(app):
        if not path.is_file():
            raise RuntimeError(f"Checkpoint missing: {path}")
        # This section runs at service startup, not on every request.
        app.state.model, app.state.tokenizer, app.state.metadata = load_checkpoint(path)
        yield  # Resources are ready; the service can receive requests.

    app = FastAPI(title="Tiny LLM educational API", lifespan=lifespan)
    # The lock lets only one text generation enter this model at a time.
    app.state.generation_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_type": "character_base_model",
            "context": app.state.model.config.context,
            "trained_step": app.state.metadata.get("step"),
        }

    return app

lesson_app = create_lesson_app(CHECKPOINT)
print("Application created; startup will run when the client context begins.")
```

<!-- cell -->
Creating `lesson_app` has not run the startup lifecycle yet. In the next experiment, the `with TestClient(...)` block activates it. The `async` keyword in the lifecycle definition follows FastAPI’s execution contract; generation in this example is still an ordinary synchronous function.

## 5. What Steps Does the Generation Endpoint Take?

After validating the general request contract, the model’s limits are checked as well. The tokenizer is strict, so an unknown character is rejected. If the initial token count exceeds the context, the service rejects it explicitly. The contract does not silently truncate the input.

A **lock** limits concurrent access. In this example, the lock is acquired without waiting. If another generation is running, a new request receives status 429. This is not a full queue or per-user rate limiter; it is only a lock inside the same process.

After generation, `text` contains the prompt and continuation together. `continuation` contains only the newly generated part. Timing is measured inside the endpoint with `perf_counter`; it does not include network time, time waiting outside the service, or work before the timer starts.

<!-- cell -->

```python
def add_generation_endpoint(app):
    @app.post("/generate")
    def completion(request: GenerateRequest):
        tokenizer = app.state.tokenizer
        try:
            encoded = tokenizer.encode(request.prompt)  # Strict mode raises on an unknown character.
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        if len(encoded) > app.state.model.config.context:
            raise HTTPException(status_code=422, detail="Prompt exceeds model context; shorten it explicitly")

        lock = app.state.generation_lock
        if not lock.acquire(blocking=False):  # If it is busy, do not wait in a queue.
            raise HTTPException(status_code=429, detail="Model busy; retry shortly")
        try:
            started = time.perf_counter()
            ids = torch.tensor([encoded], dtype=torch.long)
            generated = generate(
                app.state.model,
                ids,
                count=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                seed=request.seed,
            )
            result_ids = generated[0].tolist()
            continuation_ids = result_ids[len(encoded):]
            return {
                "text": tokenizer.decode(result_ids),
                "continuation": tokenizer.decode(continuation_ids),
                "prompt_tokens": len(encoded),
                "generated_tokens": len(continuation_ids),
                "elapsed_ms": round(1000 * (time.perf_counter() - started), 2),
            }
        finally:
            lock.release()  # Release the lock even if generation raises an error.

add_generation_endpoint(lesson_app)
print("Generation endpoint registered.")
```

<!-- cell -->
Now you have every part of the endpoint. FastAPI converts the returned dictionary into a JSON response. `HTTPException` creates an error response, and its `detail` field explains the reason. The `finally` block prevents the lock from remaining held after an error.

The lock exists inside one process. If you run multiple workers, each worker has its own model and lock. This example uses one worker and one model. At larger scale, queues, request batching, monitoring, and resource management require separate design decisions.

## 6. Test the Real Request Path Without Opening a Port

**TestClient** sends requests through FastAPI and runs the startup lifecycle, but it does not open a network server or port. This test is more complete than calling `generate` directly because it also checks input validation and response conversion.

A **status code** describes the result of processing a request. 200 means success, 422 means the request is invalid for this service, and 429 in this implementation means that the model is busy. Do not assume that every API uses 429 for exactly this lock condition.

**Cell goal:** Check a successful request, text reproducibility, several invalid inputs, and the busy state. For the 429 test, acquire the lock deliberately from outside; this simulates a state and does not measure real concurrent capacity.

<!-- cell -->

```python
with TestClient(lesson_app) as client:
    health_response = client.get("/health")
    assert health_response.status_code == 200
    print("Health:", health_response.json())

    response = client.post("/generate", json=request_body)
    assert response.status_code == 200
    lesson_result = response.json()
    print("Successful result:", lesson_result)
    assert lesson_result["generated_tokens"] == request_body["max_new_tokens"]
    assert lesson_result["text"] == request_body["prompt"] + lesson_result["continuation"]

    repeated = client.post("/generate", json=request_body).json()
    assert repeated["text"] == lesson_result["text"]  # Elapsed time does not have to match.

    too_long = "a" * (health_response.json()["context"] + 1)
    bad_requests = [
        {"prompt": ""},
        {"prompt": "😀"},
        {"prompt": too_long},
        {"prompt": "The model ", "max_new_tokens": 1000},
        {"prompt": "The model ", "unexpected_field": 1},
    ]
    for bad_body in bad_requests:
        rejected = client.post("/generate", json=bad_body)
        print("Rejected:", repr(bad_body), "status:", rejected.status_code)
        assert rejected.status_code == 422

    lesson_app.state.generation_lock.acquire()  # Simulate a generation already in progress.
    try:
        busy_response = client.post("/generate", json=request_body)
        print("Busy response:", busy_response.status_code, busy_response.json())
        assert busy_response.status_code == 429
    finally:
        lesson_app.state.generation_lock.release()
```

<!-- cell -->
An invalid request may be rejected in two different places: Pydantic checks field shapes and limits, while the generation endpoint checks vocabulary and context. Both return 422 under this contract. Repeated text should match; do not expect repeated timing to match.

The tokenizer used by this course does not contain this emoji in its vocabulary. Another model might accept it; this error is a property of this model. The limit on new tokens also differs from the limit on the initial text. After each step, the generation function sees only the last context window and can produce a total output longer than that context.

## 7. Check the Packaged Application Too

The publishable application is in `src/tinyllm/api.py`. Its model logic and contract match the example above, but `/` also provides an HTML page for trying text. `/docs` is FastAPI’s interactive documentation and testing interface. `/openapi.json` exposes the machine-readable contract definition.

The alias `packaged_create_app` keeps the real project function distinct from the educational example. You test both with the same model file and request body.

<!-- cell -->

```python
from tinyllm.api import create_app as packaged_create_app

with TestClient(packaged_create_app(CHECKPOINT)) as client:
    home = client.get("/")
    docs = client.get("/docs")
    schema = client.get("/openapi.json")
    assert home.status_code == docs.status_code == schema.status_code == 200
    assert "text/html" in home.headers["content-type"]
    assert "/generate" in schema.json()["paths"]

    packaged_response = client.post("/generate", json=request_body)
    assert packaged_response.status_code == 200
    packaged_result = packaged_response.json()
    for key in ["text", "continuation", "prompt_tokens", "generated_tokens"]:
        assert packaged_result[key] == lesson_result[key]
    print("Packaged API:", packaged_result)
    print("Home page, docs, schema and generation contract passed.")
```

<!-- cell -->
Keeping these fields aligned prevents the lesson example from drifting away from the real application. We intentionally did not compare two independent timing runs. Additional model and API tests are in `tests/test_model_and_api.py`; run the following command from the project root:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

## 8. Start the service in a separate terminal

**Uvicorn** receives network requests and passes them to FastAPI. `tinyllm.api:app` identifies the module and application object. The address `127.0.0.1` is reachable only from this computer. Port 8000 is used in this experiment; ports keep network applications separate on one host.

Open a separate PowerShell terminal in the project root. If you ran Lesson 05 with its default path, these commands use the selected model:

```powershell
$env:TINYLLM_CHECKPOINT = 'artifacts/runs/notebook/selected.pt'
.venv/Scripts/python.exe -m uvicorn tinyllm.api:app --host 127.0.0.1 --port 8000 --workers 1
```

If you did not run training, explicitly select the demo checkpoint:

```powershell
$env:TINYLLM_CHECKPOINT = 'artifacts/demo/best.pt'
.venv/Scripts/python.exe -m uvicorn tinyllm.api:app --host 127.0.0.1 --port 8000 --workers 1
```

If `TINYLLM_RUN_DIR` pointed elsewhere, put the actual `CHECKPOINT` path printed at the start of the lesson into the environment variable. Restart the server after changing it so the new model is loaded.

Open the [local page](http://127.0.0.1:8000) and [service documentation](http://127.0.0.1:8000/docs) in a browser. The terminal remains occupied by the service until you stop it with `Ctrl+C`. That is why the startup command is not placed in a normal code cell: running it there can keep the kernel waiting.

**Manual test goal:** send a request from another terminal. `ConvertTo-Json` converts a PowerShell dictionary into a JSON request body.

```powershell
$body = @{prompt='The model '; max_new_tokens=12; temperature=0.8; seed=7} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/generate -ContentType 'application/json' -Body $body
```

**Optional cell goal:** send a network request from the notebook only after you have started the server yourself. The default is off so that the notebook can run independently without an external server.

<!-- cell -->

```python
RUN_LOCAL_HTTP_REQUEST = False  # Set to True after starting Uvicorn manually.
if RUN_LOCAL_HTTP_REQUEST:
    import httpx
    response = httpx.post("http://127.0.0.1:8000/generate", json=request_body, timeout=30)
    response.raise_for_status()  # Turn an HTTP error response into a Python exception.
    print(response.json())
else:
    print("Optional network request skipped; TestClient checks already ran.")
```

<!-- cell -->
## 9. Prepare a self-contained service bundle

**Model bundle (artifact bundle)** is the set of files you carry to another environment to run the model. It differs from the full training directory: the service does not need raw training data, the local virtual environment, or every experiment output.

The bundle contains code, installation configuration, Dockerfile, the selected model file, the license, and data provenance. `best.pt` also contains the configuration and tokenizer; its name is fixed inside the bundle so the service default points to it. The model file is copied; no retraining occurs.

Create the bundle in a new directory so files from an older bundle cannot be included accidentally. The original model file remains in its current location. The following code copies files locally and uploads nothing.

<!-- cell -->

```python
import shutil
import tempfile

run_root.mkdir(parents=True, exist_ok=True)
bundle = Path(tempfile.mkdtemp(prefix="space_bundle_", dir=run_root))
for filename in ["Dockerfile", "pyproject.toml", "LICENSE", ".dockerignore"]:
    shutil.copy2(ROOT / filename, bundle / filename)

shutil.copytree(
    ROOT / "src" / "tinyllm",
    bundle / "src" / "tinyllm",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
demo_folder = bundle / "artifacts" / "demo"
demo_folder.mkdir(parents=True)
bundled_checkpoint = demo_folder / "best.pt"
shutil.copy2(CHECKPOINT, bundled_checkpoint)
shutil.copy2(ROOT / "docs" / "SPACE_README.md", bundle / "README.md")

# Keep data provenance and licensing with the bundle; the training data itself is not needed for inference.
provenance_folder = bundle / "provenance"
provenance_folder.mkdir()
shutil.copy2(ROOT / "data" / "manifest.json", provenance_folder / "data_manifest.json")

bundle_model, bundle_tokenizer, bundle_metadata = load_checkpoint(bundled_checkpoint)
assert bundle_tokenizer.to_dict() == loaded_tokenizer.to_dict()
assert bundle_metadata == loaded_metadata
print("Bundle path:", bundle)
print("Bundled model step:", bundle_metadata.get("step"))
```

<!-- cell -->
The printed path has a unique name; use that exact path in the Docker instructions. This project includes the compatible architecture in the bundle code. Weights alone are not the complete model runtime, and code alone is not trained weights.

### Record the manifest and bundle hash

**Manifest** is a list of bundle contents and provenance. Here you record filenames and hashes so you can later identify which model was published with which code. A hash does not prove that a source is trustworthy or a model is high quality; it helps detect content and change.

`data_sha256` comes from the training metadata. The current data provenance file is copied as well. Comparing its hash with the metadata helps detect a mismatch between the model and its data description.

<!-- cell -->

```python
file_hashes = {}
for path in sorted(bundle.rglob("*")):
    if path.is_file():
        relative_name = path.relative_to(bundle).as_posix()
        file_hashes[relative_name] = hashlib.sha256(path.read_bytes()).hexdigest()

corpus_manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
artifact_manifest = {
    "checkpoint": "artifacts/demo/best.pt",
    "checkpoint_sha256": file_hashes["artifacts/demo/best.pt"],
    "model_metadata": bundle_metadata,
    "data_manifest_sha256": corpus_manifest["sha256"],
    "files_sha256": file_hashes,
}
(bundle / "artifact_manifest.json").write_text(
    json.dumps(artifact_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Files recorded:", len(file_hashes))
print("Model data hash:", bundle_metadata.get("data_sha256"))
print("Corpus manifest data hash:", corpus_manifest["sha256"])
if bundle_metadata.get("data_sha256"):
    assert bundle_metadata["data_sha256"] == corpus_manifest["sha256"]
```

<!-- cell -->
This list does not hash the Manifest file itself because the file is created at the end. If you change files later, rebuild the hash list. Required service files include weights, tokenizer, configuration, compatible code, and dependencies; metadata and provenance are also needed to trace the run and report results.

## 10. What does Docker package?

In Docker, an **image** is a ready-made application environment containing Python, libraries, code, and the model. A **container** is a live execution of that image. Build an image with `build` and run a container with `run`.

`Dockerfile` contains the instructions for building this environment. Read it from the real bundle so the guide stays aligned with executable code. This cell only displays the file; it does not run Docker.

<!-- cell -->

```python
dockerfile_text = (bundle / "Dockerfile").read_text(encoding="utf-8")
for line_number, line in enumerate(dockerfile_text.splitlines(), start=1):
    print(f"{line_number:02d}: {line}")
```

<!-- cell -->
Each instruction has a specific role:

- **`FROM`:** selects the base Python image. The `slim` image is smaller and includes the basic tools.
- **`WORKDIR`:** sets the working directory inside the container to `/app`, which differs from your Windows folder.
- **`ENV`:** sets Python runtime options and the number of compute threads.
- **`RUN useradd`:** creates the `learner` user with UID 1000.
- **`COPY`:** places installation configuration, code, and the model inside the image.
- **Installation `RUN`:** installs the CPU build of PyTorch and then the project package. This step usually needs internet access.
- **`USER`:** runs the final service as the ordinary `learner` user. This service does not need to write into the weights directory.
- **`EXPOSE`:** documents internal port 7860; by itself it does not connect the port to the host.
- **`CMD`:** starts Uvicorn with one worker when the container starts. `0.0.0.0` accepts requests from interfaces inside the container; Docker controls the host mapping.

Installing libraries, loading weights, and generating responses are separate from training. This Dockerfile does not retrain the model during build or startup. Package versions are recorded in `pyproject.toml`, and the tested environment is recorded in `environment-lock.txt`.

## 11. Build the image and run it on Windows

Docker Desktop must be running in Linux container mode. Earlier checks ran without Docker; if they passed but this step fails, first inspect Docker status and the build log.

In a separate terminal, change to the **actual printed bundle path**. Replace the example path below with your own; the final directory name changes on each run:

```powershell
Set-Location 'C:/PATH/TO/space_bundle_CREATED_BY_THIS_NOTEBOOK'
docker build -t tiny-llm-fa:lesson06 .
docker run --rm --name tiny-llm-fa-lesson06 -p 127.0.0.1:7861:7860 tiny-llm-fa:lesson06
```

The final `build` argument uses the current folder as the build context. `-t` supplies the image name. `--rm` removes the temporary build container after it stops, but not the image. The `--name` value identifies the container for log inspection.

`127.0.0.1:7861:7860` maps port 7861 on this computer to port 7860 inside the container and keeps it local. Port 7861 avoids colliding with a demo service that may already use 7860. If 7861 is occupied, change only the host port.

Open the [model page](http://127.0.0.1:7861), [health endpoint](http://127.0.0.1:7861/health), and [API documentation](http://127.0.0.1:7861/docs). Use another terminal to inspect logs and the user running the same container:

```powershell
docker logs tiny-llm-fa-lesson06
docker exec tiny-llm-fa-lesson06 id
```

The `id` output should show UID 1000 for `learner`. If the container is not running, `exec` fails; read the startup log first. Stop the container with `Ctrl+C` in the terminal where it is running.

## 12. Package the course as an independent GitHub project

The Space bundle runs the service. For your personal GitHub repository, you want notebooks, code, training data, guides, tests, and the model example in one independent project.

This course lives inside an Obsidian notes folder. Running `git add .` from its parent could publish unrelated notes. `scripts/package_course.py` selects only course files and writes them to a ZIP; it excludes the virtual environment, local run outputs, Git files, and external data.

**Cell goal:** build the project bundle and inspect it before publishing. This is local and does not require a GitHub account. A fresh Python process uses the same notebook environment.

<!-- cell -->

```python
import subprocess
import zipfile

completed = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "package_course.py")],
    cwd=ROOT,
    check=True,  # Stop the cell if packaging fails.
    capture_output=True,
    text=True,
    encoding="utf-8",
)
print(completed.stdout)
course_zip = ROOT / "dist" / "tiny-llm-fa.zip"
with zipfile.ZipFile(course_zip) as archive:
    archived_names = archive.namelist()
    assert all(name.startswith("tiny-llm-fa/") for name in archived_names)
    assert not any("/.venv/" in name or "/.git/" in name for name in archived_names)
    assert not any("/artifacts/runs/" in name for name in archived_names)
    print("Archive files:", len(archived_names))
    print("First files:\n", "\n".join(archived_names[:12]))
```

<!-- cell -->
The archive is written to `dist/tiny-llm-fa.zip`. It includes the demo model from `artifacts/demo`; your selected training model in `artifacts/runs` does not automatically replace the demo. The Space bundle includes the selected `CHECKPOINT`, whose path is recorded in the Manifest.

If you want your own training result in the full repository, add the selected weights and report to a clear directory such as `artifacts/my_model` after inspection, and update the service path. Record the license, provenance, step count, and quality limitations beside it. The current packager does not automatically include that new directory; update its allowlist intentionally.

### Extract the files outside the Vault

Extract the ZIP outside Obsidian and outside other repositories, for example into a new project folder. The ZIP contains a `tiny-llm-fa` directory. Enter the directory containing `README.md`, `notebooks`, and `src`.

Create an empty GitHub repository. If you want to run these commands without an initial merge, do not add a README or license during repository creation; those files already exist locally.

**Repository** stores the project file history. `git init` creates it locally, `git add` selects files for a commit, and `git commit` records a version. `remote` stores the GitHub repository address and `push` sends recorded commits there.

Run the following commands **inside the extracted project directory**. Replace the sample account and repository names with your real values:

```powershell
git init -b main
git add .
git status --short
git commit -m "Add Persian tiny GPT course and deployment demo"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Before `commit` and `push`, `git status` should list only files from this project. Review any outputs you intend to publish separately; a notebook output may contain personal paths or input text. This check concerns the actual file contents.

GitHub displays `.ipynb` files but does not execute them for visitors. To publish complete outputs, review the executed copy from `artifacts/notebook_execution`, place it in a directory such as `examples`, and commit it. Do not present a 16-step smoke run as a full training run.

This course model is small enough for ordinary Git. If weights become large later, store them in the Hugging Face Model Hub and record the exact identifier or revision in the code repository. **Git LFS** stores large files in Git; it is not necessary to start with it for this small model.

The workflow in `.github/workflows` checks the model and API tests and runs a short notebook execution on GitHub. It does not replace a full quality report.

## 13. Publish the service to a Hugging Face Space

**Spaces** on Hugging Face hosts demo applications. GitHub stores the code, while the Space runs the application. A local training run did not require an HF account; this publishing step does.

In a Docker Space, `sdk: docker` and `app_port: 7860` at the top of `README.md` define the environment and application port. These settings and UID 1000 are described in the [official Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker). The internal port is 7860; host port 7861 was only the local Docker mapping on Windows.

**Cell goal:** inspect the actual README from the bundle and verify the two required values. You are checking the packaged file itself rather than rewriting the settings elsewhere.

<!-- cell -->

```python
space_readme = (bundle / "README.md").read_text(encoding="utf-8")
print(space_readme.split("---", 2)[1])  # Display the README configuration block.
assert "sdk: docker" in space_readme
assert "app_port: 7860" in space_readme
print("Ready-to-upload bundle:", bundle)
```

<!-- cell -->
Follow this sequence to publish:

1. Sign in to Hugging Face and create a new Space from the [Space creation page](https://huggingface.co/new-space), selecting the `Docker` SDK.

2. Upload the files and folders **inside the printed bundle** to the Space repository root. Do not add the `space_bundle_...` directory as an extra path level. The root should contain `Dockerfile`, `pyproject.toml`, and `README.md`, with weights at `artifacts/demo/best.pt`.

3. Replace the default Space README with the bundled README so `sdk` and `app_port` are correct. Do not upload training code or the Windows virtual environment.

4. Read the build and startup logs after uploading. If installation fails, inspect the message from that stage. If weights are missing, verify the path in the Space repository; the file must not be one directory deeper than expected.

5. After startup, test the text `The model ` in the page. Copy the public application URL from the Space and open `/health` and `/docs`. The model step shown by health must match the bundle metadata.

6. Record the Space URL, Manifest, training step, and model limitations in your GitHub README. Present the runnable result with weight provenance.

For manual web uploads, never put an access token in code or a notebook. If you later automate uploads, keep credentials in the environment-specific secret mechanism. A Space’s resources and availability depend on its host settings; this example does not guarantee commercial capacity or uptime.

The cells in this file only build a local bundle. They do not publish a public URL or personal repository; complete publishing commands with your real account and names.

## 14. Exercises and interview questions

**Exercise 1:** Change the request body to ask for five new tokens. Inspect `text`, `continuation`, and `generated_tokens`, then restore the original value.

**Exercise 2:** Explain the difference between 422 and 429 for this service with a sample request. Does 429 mean the user quota is exhausted in this program?

**Exercise 3:** Why is the model loaded at startup? If you create four workers instead of one, why should you expect four models and four independent locks?

**Exercise 4:** Which files are needed to publish the service that a weights file alone cannot provide? Where is the meaning of the identifiers stored?

**Exercise 5:** In `127.0.0.1:7861:7860`, which port does the Windows browser use and which port is inside the container? Does `EXPOSE` create this mapping by itself?

<!-- fa-lint-ignore-next-line -->
<details><summary>Open the answers after solving the exercises</summary>

**Answer 1:** The new-token count should be 5. The full text contains the prompt and continuation; continuation reconstructs only identifiers after the prompt. The request output limit is separate from the number of prompt tokens.

**Answer 2:** Empty input or an unknown character returns 422. A held lock returns 429. This example has no per-user quota or time-window rate limiter; 429 means the model is busy in the same process.

**Answer 3:** Loading at startup removes repeated per-request cost and exposes missing-weight errors early. Workers are separate processes, so each loads its own model. An in-process lock does not coordinate all workers.

**Answer 4:** Publishing requires compatible code and architecture, libraries, and runtime configuration. The packaged weight file contains the Config and tokenizer; the tokenizer character list defines identifier meaning. Keep the license, data provenance, and metadata for traceability.

**Answer 5:** The browser opens host port 7861 and the application listens on container port 7860. `-p` creates the mapping; `EXPOSE` is documentation.

</details>

**Question: Does a successful deployment mean the model is a good assistant?** No. Successful execution means the bundle loads and the service contract works. Text quality, factuality, capacity, access control, and monitoring require separate evaluation and design. This is an educational continuation model.

**Question: Does this model have a KV cache?** No. This project recomputes the final window on every generation step. If you add Key/Value caching later, measure time, memory, and output correctness separately; the name GPT does not imply a cache.

**Next step:** [Lesson 07: Real Hugging Face tools and the next path](07_huggingface_and_next_steps.ipynb). It examines a real tokenizer, multilingual embeddings, and an external dataset, and separates pretraining, fine-tuning, and RAG.
