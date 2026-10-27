# 05. How Does the Model Learn from Examples and Save Its Results?

In Lesson 04, you built a small model that turns input IDs into scores for the next tokens. Its initial weights are random; building the architecture alone does not mean that it has learned language. This lesson changes the weights by showing the model real examples, then measures and saves the result for later use.

**Training** here means repeating a process: take a text sample, compare the model outputs with the correct continuation, compute the error, and update the weights to reduce that error. The text supplies the labels; after each position, the next character is the target.

**Prerequisite:** [Lesson 04](04_build_tiny_gpt.ipynb). You do not need variables from that file; this notebook creates its own environment. Reviewing the one-number learning example in Lesson 00 makes the gradient section easier.

First, inspect one small batch and one training step. Once the details are clear, run two complete experiments. A normal experiment has 400 steps, and its duration depends on your computer. The experiment section gives you a chance to pause after reading the explanation.

<!-- cell -->

## 1. Prepare the Environment and Results Directory

As in the previous lessons, locate the project directory by checking the current directory and its parents. The data is local; this lesson does not require a Hugging Face account or a dataset download.

This notebook uses the `TINYLLM_SMOKE` environment variable for a quick code check. When it is `1`, each experiment has 16 steps. That mode is for finding program failures and cannot establish model quality. A normal run has 400 steps when the variable is unset.

`RUN_DIR` is the directory where this run stores its results. Automated runs can change it with the `TINYLLM_RUN_DIR` environment variable. The prepared examples in `artifacts/demo` are not replaced by this lesson.

<!-- cell -->
```python
from pathlib import Path
import sys
import os
import math
import json
import hashlib
import torch

ROOT = None
for candidate in [Path.cwd()] + list(Path.cwd().parents):
    if (candidate / "src" / "tinyllm").is_dir():
        ROOT = candidate
        break
if ROOT is None:
    raise RuntimeError("Run this notebook from inside tiny-llm-fa.")

source_path = str(ROOT / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)  # This lets import find the code in the current project.

torch.set_num_threads(2)
torch.manual_seed(7)
SMOKE = os.environ.get("TINYLLM_SMOKE") == "1"
STEPS = 16 if SMOKE else 400
EVAL_EVERY = 8 if SMOKE else 100  # Number of steps between evaluation records.
default_run_dir = ROOT / "artifacts" / "runs" / "notebook"
RUN_DIR = Path(os.environ.get("TINYLLM_RUN_DIR", str(default_run_dir)))
RUN_DIR.mkdir(parents=True, exist_ok=True)  # Create the required directories when they do not exist.
print("Project:", ROOT.name, "steps per experiment:", STEPS)
print("Run directory:", RUN_DIR)
```

<!-- cell -->

Running the lesson again overwrites files with the same names in this run directory. To keep an experiment, change `RUN_DIR` to a new directory before starting another run. Building and checking the small example takes much less time than the complete experiment.

## 2. Why Do We Have Three Data Splits?

The **training** split is the only split used to compute gradients and update weights. The **validation** split supports decisions such as choosing the learning rate and selecting the best step. The **test** split is held out for the final evaluation of the selected model.

If you use an exam to choose how to study, that exam is no longer an independent measure of learning. Likewise, this lesson does not compute test loss until model selection is finished. Counting documents or checking that they are not duplicates is different from using test results to tune the model.

This course uses 6 instructional texts and 144 short Persian and English scenarios. The split is performed by document, and each split is then converted to token IDs. The vocabulary is built from the training split only. `load_data` returns the documents, the text for each split, the tokenizer, and the ID tensor for each split.

<!-- cell -->
```python
from tinyllm.data import load_data, get_batch
from tinyllm.model import Config, TinyGPT, generate
from tinyllm.training import evaluate, run_training
from tinyllm.artifacts import save_checkpoint, load_checkpoint

documents, texts, tokenizer, data = load_data(ROOT / "data")
for split_name in ["train", "validation", "test"]:
    document_count = 0
    for document in documents:
        if document["split"] == split_name:
            document_count += 1
    unknown_ratio = (data[split_name] == 0).float().mean().item()
    print(split_name, "documents:", document_count,
          "tokens:", data[split_name].numel(), "unknown ratio:", unknown_ratio)

print("First training document:\n", documents[0]["text"])
```

<!-- cell -->

In the output, the unknown ratio is the share of zero IDs among all tokens in that split. Report it alongside the model result; loss measured on unknown inputs is not equivalent to loss on fully known text.

### Is the Absence of Exact Duplicates Enough?

A document ID must not be shared by two splits. You also check for exact duplicate text. `SHA256` produces a stable string from the bytes of a text; this **hash** is useful for comparing file contents. In this check, equal hashes indicate exact duplicate text.

The `&` operator finds the members shared by two sets. `assert not shared` says that execution should stop with an error if the shared set is not empty.

<!-- cell -->
```python
split_ids = {}
split_hashes = {}
for split_name in data:
    split_ids[split_name] = set()
    split_hashes[split_name] = set()
    for document in documents:
        if document["split"] == split_name:
            split_ids[split_name].add(document["id"])
            text_bytes = document["text"].encode("utf-8")
            split_hashes[split_name].add(hashlib.sha256(text_bytes).hexdigest())

for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not (split_ids[first] & split_ids[second])
    assert not (split_hashes[first] & split_hashes[second])
print("No repeated document IDs or exact text hashes across splits.")
```

<!-- cell -->

This check does not rule out shared sentences or similar templates. The scenarios use common templates, and the validation and test splits come from the same family of synthetic text. The result is therefore useful for observing local pattern learning, but it is not a measure of general language or reasoning ability. The data-generation script is in `scripts/build_data.py`, and the data provenance is described in `data/README.md`.

## 3. Shift the Input and Target by One Position

Suppose the text is “model.” For input “m,” the target is “o”; for input “mo,” the next target is “d.” During training, all of these positions are computed together. The causal mask prevents the current position from seeing a future target in its input.

A **window** is the part of the text that the model sees in one computation. If the input length is `T`, you need `T+1` tokens to construct all targets. The first `T` tokens are input `x`, and the last `T` tokens are target `y`. We call this offset a **shift**.

**Example goal:** See this relationship in a short text before randomizing samples. The slice `[:4]` takes the first four elements, while `[1:5]` takes four elements from position one up to (but not including) position five.

<!-- cell -->
```python
short_text = "The model "
short_ids = torch.tensor(tokenizer.encode(short_text), dtype=torch.long)
x_one = short_ids[:4]
y_one = short_ids[1:5]
print("Input:", repr(tokenizer.decode(x_one.tolist())))
print("Target:", repr(tokenizer.decode(y_one.tolist())))
for position in range(4):
    print("Position:", position,
          "input:", repr(tokenizer.decode([x_one[position]])),
          "target:", repr(tokenizer.decode([y_one[position]])))
assert torch.equal(x_one[1:], y_one[:-1])
```

<!-- cell -->

The input should be `The ` and the target should be `he m`. At the position of `e`, the model sees the preceding `The` and is expected to predict the space. Setting `y=x` instead of applying this shift creates a different training problem and is incorrect for this language model.

### Group Several Windows into a Batch

A **batch** is a group of windows processed together in one step. The project’s `get_batch` function randomly chooses a start position for each window, builds the input and target, and places the windows along a new dimension. In PyTorch, this operation is called `stack`.

The annotated version below performs the same work with an explicit loop. You provide an independent random generator for choosing the start positions. This separation prevents evaluation sampling from changing the training sampling stream.

<!-- cell -->
```python
def explained_get_batch(source, context, batch_size, generator, device="cpu"):
    if len(source) <= context or context <= 0 or batch_size <= 0:
        raise ValueError("Need positive dimensions and more than context tokens.")
    # The upper bound of randint is exclusive; the last valid start is one less than this bound.
    starts = torch.randint(len(source) - context, (batch_size,), generator=generator)
    input_windows = []
    target_windows = []
    for start in starts:
        input_windows.append(source[start:start + context])
        target_windows.append(source[start + 1:start + context + 1])
    x = torch.stack(input_windows)
    y = torch.stack(target_windows)
    return x.to(device), y.to(device)

example_rng = torch.Generator().manual_seed(8)
x, y = explained_get_batch(data["train"], context=8, batch_size=2, generator=example_rng)
for row in range(2):
    print("x:", repr(tokenizer.decode(x[row].tolist())))
    print("y:", repr(tokenizer.decode(y[row].tolist())))
print("Shapes:", tuple(x.shape), tuple(y.shape))
assert torch.equal(x[:, 1:], y[:, :-1])

# With the same seed, the reference implementation should return exactly the same inputs and targets.
reference_rng = torch.Generator().manual_seed(8)
reference_x, reference_y = get_batch(data["train"], 8, 2, reference_rng)
assert torch.equal(x, reference_x) and torch.equal(y, reference_y)
```

<!-- cell -->

Both shapes are `(2, 8)`: two samples with eight positions each. The final target must also be inside the data; therefore, for data of length `N`, the upper bound for start positions is `N-context`, and that bound itself is not selected.

## 4. Understand Prediction Error with One Simple Number

For each position, the model produces several **raw scores, or logits**. Softmax converts these scores into probabilities. During training, the target ID tells us which probability should increase.

**Cross-entropy loss** for a correct target is the negative log probability assigned to that target. A lower target probability produces a larger penalty. The function does not merely count “right or wrong”; it also captures the model’s confidence in the correct answer.

In the next example, there are three possible answers and the correct answer has ID zero. Its probability is 0.7. For comparison with the library implementation, you use log probabilities as raw scores; Softmax maps those scores back to the original probabilities.

<!-- cell -->
```python
import torch.nn.functional as F

probabilities = torch.tensor([[0.7, 0.2, 0.1]])
target = torch.tensor([0], dtype=torch.long)
manual_loss = -probabilities[0, 0].log()
library_loss = F.cross_entropy(probabilities.log(), target)
print("Manual loss:", manual_loss.item())
print("Cross-entropy:", library_loss.item())
torch.testing.assert_close(manual_loss, library_loss)

for target_id in [0, 1, 2]:
    print("Target:", target_id, "probability:", probabilities[0, target_id].item(),
          "loss:", -probabilities[0, target_id].log().item())
```

<!-- cell -->

For probability 0.7, the loss is about 0.357. If you change the target to the option with probability 0.1, the loss becomes about 2.303. Both values come from the same formula; changing the target selects a different probability.

This loss uses the natural logarithm, and its unit is a **nat**. In our model, loss is averaged over the positions in all samples, so we report it as average loss per token. Because this model uses a character tokenizer, later plots will show “loss per character.”

### Establish a Simple Reference Point

If you assign each of the `V` vocabulary options the same probability, `1/V`, the loss is `log(V)`. This is the **uniform baseline**. A random model does not necessarily reach exactly this value, but it gives you a simple reference for comparison.

**Perplexity** is computed as `exp(loss)`. A simple interpretation is the effective number of options among which the model is uncertain. Compare metrics only when the data and tokenization are comparable; do not directly compare character loss with loss from a different tokenizer.

<!-- cell -->
```python
uniform_loss = math.log(tokenizer.vocab_size)
uniform_perplexity = math.exp(uniform_loss)
print("Vocabulary:", tokenizer.vocab_size)
print("Uniform loss:", uniform_loss)
print("Uniform perplexity:", uniform_perplexity)
assert math.isclose(uniform_perplexity, tokenizer.vocab_size)
```

<!-- cell -->

## 5. Open Up a Single Training Step

To observe a real weight update, build a separate demonstration model. This model is not used in the two main experiments, so the one-step exercise does not change their starting point.

An **optimizer** uses gradients to update weights. In Lesson 00, you performed this update by hand. Here you use **AdamW**, which maintains averages of gradients and squared gradients to adjust the size of each weight update.

The **learning rate**, written as `lr`, controls the size of updates. A larger rate does not necessarily mean better or faster training; it can make the computation unstable. **Weight decay** in AdamW moves weights slightly toward zero at every step and is separate from the data-loss gradient. Its value in this lesson is 0.01.

<!-- cell -->
```python
torch.manual_seed(7)
demo_config = Config(vocab=tokenizer.vocab_size)
demo_model = TinyGPT(demo_config)
demo_optimizer = torch.optim.AdamW(demo_model.parameters(), lr=0.002, weight_decay=0.01)
demo_rng = torch.Generator().manual_seed(8)
batch_x, batch_y = get_batch(data["train"], demo_config.context, 16, demo_rng)
parameter_count = sum(parameter.numel() for parameter in demo_model.parameters())
print("Parameters:", parameter_count)
print("Batch shape:", tuple(batch_x.shape))
print("Learning rate:", demo_optimizer.param_groups[0]["lr"])
```

<!-- cell -->

`model.parameters()` returns the trainable weights. The `numel` function counts the numbers in each tensor, and their sum is the model’s parameter count. The parameter count is not the number of input tokens or training steps.

`model.train()` enables the training behavior of layers such as Dropout. With Dropout, some values are randomly removed during training. This command does not learn any weights; it only selects the execution mode for the layers.

**Next-cell goal:** Observe every stage of a weight update in execution order. Before `optimizer.step`, inspect one sample weight so the difference between “computing a gradient” and “changing a weight” remains clear.

<!-- cell -->
```python
demo_model.train()  # Enable training mode, including Dropout behavior.
tracked_weight = next(demo_model.parameters())  # Select one weight tensor for comparison.
before_step = tracked_weight.detach().clone()   # Make an independent copy of the value before the update.

logits, loss = demo_model(batch_x, batch_y)  # Forward pass: predictions and average loss.
if not torch.isfinite(loss):  # Detect NaN or infinity before continuing.
    raise RuntimeError("Non-finite training loss.")

demo_optimizer.zero_grad(set_to_none=True)  # Clear gradients left from the previous step.
loss.backward()  # Compute the loss sensitivity to each weight and store it in its grad field.
assert torch.equal(tracked_weight, before_step)  # backward has not changed the weights yet.

# Clipping limits the overall gradient norm; the returned value is the norm before clipping.
norm_before_clip = torch.nn.utils.clip_grad_norm_(demo_model.parameters(), max_norm=1.0)
demo_optimizer.step()  # AdamW now updates the weights using the gradients and its internal state.
print("Loss for this batch:", loss.item())
print("Logits shape:", tuple(logits.shape))
print("Gradient norm before clipping:", float(norm_before_clip))
print("Tracked weight changed:", not torch.equal(tracked_weight, before_step))
assert not torch.equal(tracked_weight, before_step)
```

<!-- cell -->

The logits should have shape `(16, 64, V)`: sixteen windows, sixty-four positions, and one score for each vocabulary item. The weight-change output should be `True`. Do not infer overall model quality from the loss decrease on this single batch.

### Why Clear the Previous Gradient?

In PyTorch, calling `backward` accumulates gradients. Sometimes this accumulation is intentional, for example when simulating a larger batch with several smaller batches. This method is called **gradient accumulation**. In this lesson’s loop, each step is independent, so `zero_grad` is required. `set_to_none=True` removes the previous gradient value.

### What Does Gradient Clipping Do?

The gradient **norm** here is the square root of the sum of squared components across all weights. **Gradient clipping** scales all gradients by a shared factor when this norm exceeds one. It can prevent extreme updates; it does not automatically fix a bad learning rate, corrupted data, or an incorrect model.

The main experiments repeat the same order: run the model, check the numbers, clear the previous gradients, compute new gradients, clip them, and update the weights. AdamW also has internal state, so two steps with similar gradients do not necessarily produce identical updates.

## 6. How Is Evaluation Different from Training?

During **evaluation**, you compute predictions and loss but do not update the weights. When comparing two models, keep the inputs and the behavior of random layers consistent.

`model.eval()` disables Dropout. `torch.no_grad()` disables recording the computation graph for gradients. They are independent: `eval` does not disable gradient computation, and `no_grad` does not change Dropout behavior.

The function below is an annotated version of the project `evaluate` function. Each call creates a sampler with seed 123 so the same windows are evaluated. The model’s previous mode is saved, and the `finally` block restores it even when an error occurs.

<!-- cell -->
```python
@torch.no_grad()  # Apply this rule to the entire function.
def explained_evaluate(model, source, rounds=12, batch_size=16, seed=123):
    previous_mode = model.training
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    losses = []
    device = next(model.parameters()).device
    try:
        for _ in range(rounds):  # English explanatory comment.
            x, y = get_batch(source, model.config.context, batch_size, rng, device=device)
            _, batch_loss = model(x, y)
            losses.append(batch_loss.item())
        return sum(losses) / len(losses)
    finally:
        model.train(previous_mode)

first_estimate = explained_evaluate(demo_model, data["validation"])
second_estimate = evaluate(demo_model, data["validation"])
print("Explained evaluation:", first_estimate)
print("Project evaluation:", second_estimate)
assert first_estimate == second_estimate
assert demo_model.training  # English explanatory comment.
```

<!-- cell -->

The two estimates should match. From this point on, use the project's main function, which has the same behavior. The mean across 12 batches is an **error estimate**; it is not the loss over the full dataset or a statistical confidence interval. Windows may overlap. A serious evaluation requires a fixed policy and independent data.

Training and validation loss are both measured with `eval` so Dropout behavior does not affect the comparison. The instantaneous loss of one training batch is not exactly the same quantity as these reported values.

## 7. What Does a Checkpoint Store?

A **checkpoint** is a collection of information that lets you rebuild a model. Saving the weights alone is not enough. **Config** defines the architecture size, while the tokenizer defines the meaning of IDs in the weight tables.

The `state_dict` method returns a dictionary of the model tensors. Besides parameters, it stores registered buffers such as the attention mask. This project's checkpoint keeps the configuration, tokenizer, dictionary, and training-run information together.

**Metadata** describes a run, such as its step number, seed, learning rate, and data hash. Metadata does not replace the original data; the hash helps identify which data version was used. Full data provenance remains in the dataset files.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
from dataclasses import asdict  # English explanatory comment.

data_sha256 = hashlib.sha256((ROOT / "data" / "documents.json").read_bytes()).hexdigest()
cpu_state = {}
for name, tensor in demo_model.state_dict().items():
    # English explanatory comment.
    cpu_state[name] = tensor.detach().cpu()

example_payload = {
    "format_version": 1,
    "config": asdict(demo_model.config),
    "tokenizer": tokenizer.to_dict(),
    "model": cpu_state,
    "metadata": {"step": 1, "seed": 7, "lr": 0.002, "data_sha256": data_sha256},
}
print("Package keys:", list(example_payload))
print("Config:", example_payload["config"])
print("Tokenizer vocabulary:", len(example_payload["tokenizer"]["chars"]) + 1)
print("State tensors:", len(example_payload["model"]))
```

<!-- cell -->

`save_checkpoint` writes this structure with `torch.save`. Alongside the `.pt` weight file, it creates a `.json` file for readable configuration and metadata and a `.sha256` file for the content hash. A hash detects changes; by itself, it does not prove that the source is trusted.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->

## 8. Two Controlled Experiments with One Difference

So far, you have run one step and inspected evaluation and saving. Now you repeat the same process. The `run_training` function in `src/tinyllm/training.py` is the course's real loop; it uses the same `TinyGPT`, `get_batch`, `evaluate`, and `save_checkpoint`.

Only the learning rate changes between the two experiments: 0.002 in `baseline` and 0.02 in `higher_lr`. Here, `baseline` names the reference GPT experiment; do not confuse it with the uniform baseline or the bigram model in the next section.

The architecture, data, step count, batch size, Dropout, and seeds are identical. At the start of each experiment, the function sets seed 7 for the weights and seed 8 for training-batch selection. Evaluation and sampling use local generators, so they do not consume the later training-sampling stream.

This loop has no **learning-rate scheduler**; the rate stays constant throughout each experiment. Other models may warm up the rate during the first steps or reduce it later, but this lesson omits that extra variable so the effect of one setting remains clear.

### Order of Operations in the Main Loop

At step zero, the model is still random; training loss, validation loss, and a generated sample are recorded. One update is then performed. Evaluation runs again after each configured interval and at the final step.

If validation loss is lower than the previous best, `best.pt` is saved. The best step is selected from the evaluated steps; not every step is evaluated separately. At the end, `last.pt` stores the final weights together with optimizer state. The best and last models are not necessarily the same.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

**Before running:** This cell performs both complete training runs. If you reduce `STEPS` for a quick study, use the same value for both runs and report the result together with the actual step count. More steps do not necessarily improve quality.

<!-- cell -->
```python
def train_experiment(name, learning_rate):
    folder = RUN_DIR / name
    print("Starting:", name, "learning rate:", learning_rate)
    # English explanatory comment.
    last_model, trained_tokenizer, rows, summary = run_training(
        data_dir=ROOT / "data",
        out_dir=folder,
        steps=STEPS,
        eval_every=EVAL_EVERY,
        seed=7,
        width=64,
        heads=4,
        layers=2,
        context=64,
        batch_size=16,
        lr=learning_rate,
        dropout=0.1,
        threads=2,
    )
    assert trained_tokenizer.to_dict() == tokenizer.to_dict()
    # English explanatory comment.
    return rows, summary

baseline_rows, baseline_summary = train_experiment("baseline", learning_rate=0.002)
comparison_rows, comparison_summary = train_experiment("higher_lr", learning_rate=0.02)
```

<!-- cell -->

Each log line shows the step and the loss for both splits. Do not assume that the larger rate will be better or worse; read the numbers from your run. If it performs better on this synthetic data, that result is not a general recommendation. The completion time is also recorded in the report.

Each experiment directory contains these files:

- **`training_log.csv`:** Step numbers and training and validation losses.
- **`samples.md`:** Generated text at the recorded steps, including before training.
- **`best.pt`:** The model with the lowest recorded validation loss and its selection metadata.
- **`last.pt`:** The final model and optimizer state.
- **`report.json`:** Configuration, parameter count, timing, and the best experiment result.

`.json` and `.sha256` files are also created beside the weights. Two experiments provide evidence from one seed. Stronger conclusions require multiple seeds and independent data.

<!-- cell -->

## 9. Read the Loss Curve

The plot shows how loss changes over the steps. **Overfitting** means the model improves on training data but that improvement does not transfer to other data. Falling training loss together with rising validation loss can indicate it; distribution differences and program failures must also be checked.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
for label, rows in [("baseline", baseline_rows), ("higher_lr", comparison_rows)]:
    steps = [row["step"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    validation_losses = [row["validation_loss"] for row in rows]
    plt.plot(steps, train_losses, "--", label=label + " train")
    plt.plot(steps, validation_losses, label=label + " validation")
plt.axhline(uniform_loss, color="gray", label="uniform")
plt.xlabel("Weight updates")
plt.ylabel("Cross entropy / character")
plt.legend()
plt.tight_layout()
plt.savefig(RUN_DIR / "training_curves.png", dpi=140)
plt.show()

for name, summary in [("baseline", baseline_summary), ("higher_lr", comparison_summary)]:
    print(name, "best step:", summary["best_step"],
          "best validation:", summary["best_validation_loss"],
          "elapsed seconds:", round(summary["elapsed_seconds"], 2))
```

<!-- cell -->

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

### Inspect the Beginning, Middle, and End Too

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
def read_training_samples(folder):
    content = (folder / "samples.md").read_text(encoding="utf-8")
    samples = []
    for section in content.split("## Step ")[1:]:
        step_text = section.split("\n", 1)[0]
        generated_text = section.split("```text\n", 1)[1].rsplit("\n```", 1)[0]
        samples.append((int(step_text), generated_text))
    return samples

baseline_samples = read_training_samples(RUN_DIR / "baseline")
comparison_samples = read_training_samples(RUN_DIR / "higher_lr")
for name, samples in [("baseline", baseline_samples), ("higher_lr", comparison_samples)]:
    print("EXPERIMENT:", name)
    selected_positions = [0, len(samples) // 2, len(samples) - 1]
    for position in selected_positions:
        step, generated_text = samples[position]
        print("STEP:", step, "\n", generated_text, "\n")
```

<!-- cell -->

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

## 10. Select the Best Model with Validation

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
candidates = {
    "baseline": baseline_summary["best_validation_loss"],
    "higher_lr": comparison_summary["best_validation_loss"],
}
winner = min(candidates, key=candidates.get)  # English explanatory comment.
best_model, best_tokenizer, metadata = load_checkpoint(RUN_DIR / winner / "best.pt")
print("Candidates:", candidates)
print("Selected experiment:", winner)
print("Selected step:", metadata["step"])
print("Data hash matches:", metadata["data_sha256"] == data_sha256)
assert metadata["data_sha256"] == data_sha256
```

<!-- cell -->

### Do We Have the Same Model After Saving and Loading?

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
probe = torch.tensor([best_tokenizer.encode("The model ")], dtype=torch.long)
with torch.no_grad():
    logits_before = best_model(probe)[0]

selected_path = RUN_DIR / "selected.pt"
selected_sha = save_checkpoint(best_model, best_tokenizer, selected_path, metadata)
reloaded_model, reloaded_tokenizer, reloaded_metadata = load_checkpoint(selected_path)
with torch.no_grad():
    logits_after = reloaded_model(probe)[0]

torch.testing.assert_close(logits_before, logits_after, rtol=0, atol=0)
assert reloaded_tokenizer.encode("The model ") == probe[0].tolist()
assert reloaded_metadata == metadata
print("Exact save/load logits match.")
print("Selected checkpoint:", selected_path)
print("SHA256:", selected_sha)
sample_ids = generate(reloaded_model, probe, count=120, seed=7)
print(best_tokenizer.decode(sample_ids[0].tolist()))
```

<!-- cell -->

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
payload = torch.load(selected_path, map_location="cpu", weights_only=True)
print("Checkpoint keys:", list(payload))
print("Config:", payload["config"])
sidecar = json.loads(selected_path.with_suffix(".json").read_text(encoding="utf-8"))
print("Sidecar metadata:", sidecar["metadata"])
assert sidecar["sha256"] == hashlib.sha256(selected_path.read_bytes()).hexdigest()

# English explanatory comment.
last_payload = torch.load(RUN_DIR / winner / "last.pt", map_location="cpu", weights_only=True)
print("Optimizer in selected checkpoint:", "optimizer" in payload)
print("Optimizer in last checkpoint:", "optimizer" in last_payload)
```

<!-- cell -->

## 11. Is GPT Better Than the Simple Bigram?

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
counts = torch.ones(tokenizer.vocab_size, tokenizer.vocab_size)
train_source = data["train"]
for position in range(len(train_source) - 1):
    first_id = train_source[position]
    second_id = train_source[position + 1]
    counts[first_id, second_id] += 1
bigram_probabilities = counts / counts.sum(dim=-1, keepdim=True)

comparison_rng = torch.Generator().manual_seed(123)
bigram_losses = []
for _ in range(12):
    x, y = get_batch(data["validation"], best_model.config.context, 16, comparison_rng)
    # English explanatory comment.
    correct_probabilities = bigram_probabilities[x, y]
    bigram_losses.append(-correct_probabilities.log().mean().item())
bigram_validation_loss = sum(bigram_losses) / len(bigram_losses)
gpt_validation_loss = evaluate(best_model, data["validation"])
print("Same-window validation | bigram:", bigram_validation_loss)
print("Same-window validation | GPT:", gpt_validation_loss)
print("GPT loss below bigram:", gpt_validation_loss < bigram_validation_loss)
```

<!-- cell -->

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

## 12. Final Test Only After Selection

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

This section explains the corresponding training, evaluation, checkpoint, and model-selection detail.

<!-- cell -->
```python
report = {
    "steps_per_run": STEPS,
    "candidates": candidates,
    "winner": winner,
    "gpt_validation_loss": gpt_validation_loss,
    "bigram_validation_loss": bigram_validation_loss,
    "uniform_loss": uniform_loss,
    "parameters": sum(parameter.numel() for parameter in best_model.parameters()),
    "vocab_size": best_tokenizer.vocab_size,
    "seed": 7,
    "smoke": SMOKE,
    "selected_checkpoint": str(selected_path),
    "selected_sha256": selected_sha,
    "data_sha256": data_sha256,
    "data_limitation": "Small synthetic bilingual corpus with shared templates; not general-language evaluation.",
}
if not SMOKE:
    test_loss = evaluate(best_model, data["test"], rounds=24, seed=321)
    report["test_loss"] = test_loss
    report["test_perplexity"] = math.exp(test_loss)
else:
    report["test_note"] = "Smoke mode: no final quality claim or test evaluation."

report_path = RUN_DIR / "report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("Final report:", report_path)
```

<!-- cell -->

## 13. Exercises and Interview Questions

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

<!-- fa-lint-ignore-next-line -->
<details><summary>Open the answers after solving the exercises</summary>

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

</details>

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.

**Interview exercise:** Explain this point using the results from your own run.
