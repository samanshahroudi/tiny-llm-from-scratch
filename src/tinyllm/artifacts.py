"""Weights + tokenizer + configuration form one deployable model artifact."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from .data import CharTokenizer
from .model import Config, TinyGPT


def save_checkpoint(model, tokenizer, path, metadata=None, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "config": asdict(model.config),
        "tokenizer": tokenizer.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps({
        "config": payload["config"], "metadata": payload["metadata"], "sha256": digest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return digest


def load_checkpoint(path, device="cpu"):
    # Restricts unpickling to tensor/basic data types. Still use trusted artifacts.
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = Config(**payload["config"])
    tokenizer = CharTokenizer(payload["tokenizer"]["chars"])
    if tokenizer.vocab_size != config.vocab:
        raise ValueError("Tokenizer vocabulary does not match model")
    model = TinyGPT(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, tokenizer, payload["metadata"]
