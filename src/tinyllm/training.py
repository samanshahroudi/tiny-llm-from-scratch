"""Reproducible CPU teaching runs, with best-validation model selection."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from .artifacts import save_checkpoint
from .data import get_batch, load_data
from .model import Config, TinyGPT, generate


@torch.no_grad()
def evaluate(model, source, rounds=12, batch_size=16, seed=123):
    was_training = model.training
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    losses = []
    try:
        for _ in range(rounds):
            x, y = get_batch(source, model.config.context, batch_size, rng,
                             device=next(model.parameters()).device)
            _, loss = model(x, y)
            losses.append(loss.item())
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def run_training(data_dir, out_dir, steps=400, eval_every=100, seed=7,
                 width=64, heads=4, layers=2, context=64,
                 batch_size=16, lr=0.002, dropout=0.1, threads=2):
    if steps < 0 or eval_every <= 0 or lr <= 0 or not math.isfinite(lr):
        raise ValueError("Invalid training controls")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    documents, texts, tokenizer, data = load_data(data_dir)
    if min(map(len, data.values())) <= context:
        raise ValueError("Every split must be longer than context")
    model = TinyGPT(Config(tokenizer.vocab_size, context, width, heads, layers, dropout))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_rng = torch.Generator().manual_seed(seed + 1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_sha = hashlib.sha256((Path(data_dir) / "documents.json").read_bytes()).hexdigest()
    rows, samples, best = [], [], float("inf")
    start = time.perf_counter()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            train_loss = evaluate(model, data["train"])
            val_loss = evaluate(model, data["validation"])
            row = {"step": step, "train_loss": train_loss, "validation_loss": val_loss}
            rows.append(row)
            prompt = "The model "
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            text = tokenizer.decode(generate(model, ids, count=120, seed=seed)[0].tolist())
            samples.append(f"## Step {step}\n\n```text\n{text}\n```\n")
            if val_loss < best:
                best = val_loss
                save_checkpoint(model, tokenizer, out_dir / "best.pt", {
                    "step": step, "seed": seed, "validation_loss": val_loss,
                    "train_loss": train_loss, "data_sha256": corpus_sha,
                    "torch": str(torch.__version__), "lr": lr,
                    "kind": "educational_character_base_model",
                })
            print(f"step={step:4d} train={train_loss:.4f} validation={val_loss:.4f}", flush=True)
        if step == steps:
            break
        model.train()
        x, y = get_batch(data["train"], context, batch_size, train_rng)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    save_checkpoint(model, tokenizer, out_dir / "last.pt", {
        "step": steps, "seed": seed, "data_sha256": corpus_sha,
        "kind": "educational_character_base_model",
    }, optimizer=optimizer)
    with (out_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "samples.md").write_text("\n".join(samples), encoding="utf-8")
    report = {
        "seed": seed, "steps": steps, "lr": lr, "device": "cpu", "threads": threads,
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": tokenizer.vocab_size, "split_tokens": {k: len(v) for k, v in data.items()},
        "best_step": min(rows, key=lambda r: r["validation_loss"])["step"],
        "best_validation_loss": best, "elapsed_seconds": time.perf_counter() - start,
        "data_sha256": corpus_sha, "python_note": "See environment-lock.txt for tested versions",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model, tokenizer, rows, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/runs/cli"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()
    _, _, _, report = run_training(**vars(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
