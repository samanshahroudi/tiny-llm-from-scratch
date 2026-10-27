"""Generate the original English teaching corpus, split by document before tokenizing."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]

LESSONS = [
    ("The model reads a token and predicts the next token. A token is a unit of text. "
     "The scores are called logits. Softmax turns scores into probabilities. "
     "A likely token is not a verified fact. We check the source before using an answer.\n"),
    ("Attention mixes value vectors from earlier positions. A query is compared with keys. "
     "The causal mask hides future positions. The head returns a weighted sum of values. "
     "Several heads learn several ways to combine information.\n"),
    ("Training changes the weights. Inference uses the weights without updating them. "
     "A prompt gives temporary context. A database stores application memory. "
     "A cache saves intermediate keys and values while generating a continuation.\n"),
    ("We split documents before building the vocabulary. The training set teaches patterns. "
     "The validation set helps us choose settings. The test set is reserved for final evaluation. "
     "Repeated sentences can make validation misleading.\n"),
    ("The input and target differ by one position. The model sees hello and learns its continuation. "
     "Cross entropy rewards probability assigned to the observed next token. "
     "The optimizer uses gradients to change trainable parameters.\n"),
    ("We save the weights, the vocabulary, and the configuration together. "
     "The service loads the model once at startup. The client sends a prompt. "
     "The server returns a continuation and measured latency.\n"),
]


def main():
    rng = random.Random(19)
    docs = [{"id": f"lesson-{i}", "split": "train", "text": text}
            for i, text in enumerate(LESSONS)]
    services = ["search", "billing", "shipping", "support", "catalog", "analytics",
                "inventory", "checkout", "profile", "booking", "reports", "gateway"]
    checks = ["token budget", "input schema", "response time", "data source",
              "model version", "validation loss", "memory usage", "request size",
              "training log", "error rate", "cache size", "output length"]
    combinations = [(s, c) for s in services for c in checks]
    rng.shuffle(combinations)
    for i, (service, check) in enumerate(combinations):
        split = "train" if i < 108 else "validation" if i < 126 else "test"
        # Same grammar across splits, unique service/check combinations.
        text = (f"The engineer works on the {service} service. Before deployment, the engineer checks the {check}. "
                f"The model reads the supplied context and generates a continuation. "
                f"The {service} team records the result and compares it with a baseline. "
                f"If the result is wrong, the team inspects the input and the data. "
                f"A small test helps the engineer find a problem before release. Report {i + 1}.")
        docs.append({"id": f"scenario-{i:03d}", "split": split, "text": text})
    path = ROOT / "data" / "documents.json"
    path.write_text(json.dumps(docs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "name": "Original bilingual engineering teaching corpus",
        "source": "scripts/build_data.py; authored/generated for this course; no external download",
        "license": "MIT", "seed": 19,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "documents": {s: sum(d["split"] == s for d in docs) for s in ("train", "validation", "test")},
        "limitations": "Template overlap across splits; held-out combinations, not independent natural language evaluation.",
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
