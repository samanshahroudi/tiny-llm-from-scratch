"""Train-only character vocabulary and inspectable shifted batches."""
import json
from pathlib import Path
import torch


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        if len(set(self.chars)) != len(self.chars) or any(len(ch) != 1 for ch in self.chars):
            raise ValueError("chars must be unique single Unicode code points")
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1  # 0 is <unk>

    @classmethod
    def fit(cls, text):
        return cls(sorted(set(text)))

    def encode(self, text, strict=True):
        unknown = sorted(set(text) - set(self.chars))
        if unknown and strict:
            raise ValueError(f"Characters outside the trained vocabulary: {unknown!r}")
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join("�" if int(i) == 0 else self.chars[int(i) - 1] for i in ids)

    def to_dict(self):
        return {"type": "character", "unk_id": 0, "chars": self.chars}


def load_data(data_dir):
    documents = json.loads((Path(data_dir) / "documents.json").read_text(encoding="utf-8"))
    split_texts = {}
    for split in ("train", "validation", "test"):
        selected = [d["text"] for d in documents if d["split"] == split]
        if not selected:
            raise ValueError(f"Missing {split} documents")
        split_texts[split] = "\n\n".join(selected) + "\n"
    tokenizer = CharTokenizer.fit(split_texts["train"])
    tensors = {name: torch.tensor(tokenizer.encode(text, strict=False), dtype=torch.long)
               for name, text in split_texts.items()}
    return documents, split_texts, tokenizer, tensors


def get_batch(source, context, batch_size, generator, device="cpu"):
    if len(source) <= context or context <= 0 or batch_size <= 0:
        raise ValueError("Need positive dimensions and more than context tokens")
    starts = torch.randint(len(source) - context, (batch_size,), generator=generator)
    x = torch.stack([source[i:i + context] for i in starts])
    y = torch.stack([source[i + 1:i + context + 1] for i in starts])
    return x.to(device), y.to(device)
