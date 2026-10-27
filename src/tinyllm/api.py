"""Local demo API, usable unchanged in a Docker Space."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
import torch
from .artifacts import load_checkpoint
from .model import generate


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=256)
    max_new_tokens: int = Field(default=80, ge=1, le=128)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_k: int = Field(default=20, ge=0, le=256)
    seed: int = Field(default=7, ge=0, le=2147483647)


def create_app(checkpoint=None):
    path = Path(checkpoint or os.environ.get("TINYLLM_CHECKPOINT", "artifacts/demo/best.pt"))
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if not path.is_file():
            raise RuntimeError(f"Checkpoint missing: {path}; run notebook 05 or the trainer first")
        torch.set_num_threads(int(os.environ.get("TINYLLM_THREADS", "2")))
        app.state.model, app.state.tokenizer, app.state.metadata = load_checkpoint(path)
        yield

    app = FastAPI(title="Tiny LLM — educational demo", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_type": "character_base_model",
                "context": app.state.model.config.context,
                "trained_step": app.state.metadata.get("step")}

    @app.post("/generate")
    def completion(request: GenerateRequest):
        tokenizer = app.state.tokenizer
        try:
            encoded = tokenizer.encode(request.prompt)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if len(encoded) > app.state.model.config.context:
            raise HTTPException(422, "Prompt exceeds model context; shorten it explicitly")
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "Model busy; retry shortly")
        try:
            start = time.perf_counter()
            ids = torch.tensor([encoded], dtype=torch.long)
            result = generate(app.state.model, ids, request.max_new_tokens,
                              request.temperature, request.top_k, request.seed)[0].tolist()
            return {"text": tokenizer.decode(result),
                    "continuation": tokenizer.decode(result[len(encoded):]),
                    "prompt_tokens": len(encoded), "generated_tokens": len(result) - len(encoded),
                    "elapsed_ms": round(1000 * (time.perf_counter() - start), 2)}
        finally:
            lock.release()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return '''<!doctype html><html lang="en" dir="ltr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tiny language model laboratory</title>
<style>body{font:18px/1.8 sans-serif;max-width:760px;margin:48px auto;padding:24px;background:#f5f7fb;color:#17243c}
textarea,pre{box-sizing:border-box;width:100%;padding:14px;border-radius:8px;border:1px solid #b8c6dc}
pre{white-space:pre-wrap;overflow-wrap:anywhere;direction:ltr;background:white}
button{padding:10px 22px;background:#245ac6;color:white;border:0;border-radius:8px;cursor:pointer}</style>
<h1>Tiny language model: from code to service</h1>
<p>This educational model generates text continuations. Compare quality with the validation curve.</p>
<label for="prompt">Prompt; at most 64 characters from the training vocabulary</label>
<textarea id="prompt" dir="auto" rows="3" maxlength="64">The model </textarea>
<button id="run">Generate continuation</button><span id="status" role="status"></span>
<pre id="output" aria-live="polite"></pre><a href="/docs">API documentation and playground</a>
<script>document.getElementById('run').onclick=async()=>{
const button=document.getElementById('run');button.disabled=true;
document.getElementById('status').textContent=' Generating…';
try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({prompt:document.getElementById('prompt').value,max_new_tokens:80})});
const d=await r.json();document.getElementById('output').textContent=r.ok?d.text:JSON.stringify(d.detail);
document.getElementById('status').textContent=r.ok?' '+d.elapsed_ms+' ms':' Error';
}catch(e){document.getElementById('status').textContent=' Connection failed';}
finally{button.disabled=false;}};</script></html>'''

    return app


app = create_app()
