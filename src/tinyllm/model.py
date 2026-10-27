"""GPT-style causal decoder; intentionally separate attention heads for teaching."""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab: int
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab, self.context, self.width, self.heads, self.layers) <= 0:
            raise ValueError("All dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class Head(nn.Module):
    def __init__(self, c):
        super().__init__()
        head_size = c.width // c.heads
        self.query = nn.Linear(c.width, head_size, bias=False)
        self.key = nn.Linear(c.width, head_size, bias=False)
        self.value = nn.Linear(c.width, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(c.context, c.context)).bool())
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        time = x.shape[1]
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~self.mask[:time, :time], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.dropout(weights) @ v


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = nn.ModuleList([Head(c) for _ in range(c.heads)])
        self.project = nn.Linear(c.width, c.width)
        self.attention_dropout = nn.Dropout(c.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(c.width, 4 * c.width), nn.GELU(),
            nn.Linear(4 * c.width, c.width), nn.Dropout(c.dropout),
        )
        self.norm1 = nn.LayerNorm(c.width)
        self.norm2 = nn.LayerNorm(c.width)

    def forward(self, x):
        normalized = self.norm1(x)
        joined = torch.cat([head(normalized) for head in self.heads], dim=-1)
        x = x + self.attention_dropout(self.project(joined))
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.token = nn.Embedding(c.vocab, c.width)
        self.position = nn.Embedding(c.context, c.width)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.output = nn.Linear(c.width, c.vocab)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context:
            raise ValueError("ids must have shape (B, T), with 1 <= T <= context")
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.token(ids) + self.position(positions)
        hidden = self.blocks(hidden)
        logits = self.output(self.norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError("targets must have the same shape as ids")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab), targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def generate(model, ids, count=80, temperature=0.8, top_k=20, seed=7):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Generation expects one non-empty prompt: (1, T)")
    if count < 0 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("count, temperature and top_k must be non-negative and finite")
    # A local RNG avoids changing the training RNG when we show samples.
    rng = torch.Generator(device=ids.device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(count):
            window = ids[:, -model.config.context:]
            logits, _ = model(window)
            scores = logits[:, -1, :]
            if temperature == 0:
                next_id = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k:
                    cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                probs = F.softmax(scores, dim=-1)
                next_id = torch.multinomial(probs, 1, generator=rng)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
    finally:
        model.train(was_training)
