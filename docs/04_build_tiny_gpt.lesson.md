# 04. Build a Tiny Language Model, from One Layer to a Complete Model

In the previous lessons, you saw text become IDs, each ID become a vector, and attention connect information from earlier positions. Now you will combine these parts in a real program. By the end, the model will produce a score for the next token at every position and generate a continuation by selecting tokens one after another.

You will write this model from scratch; it does not download trained weights from the internet. The first generated text will therefore probably be meaningless. The weights are trained in Lesson 05. Separating “building the network” from “training the network” keeps the purpose of each part clear.

**Prerequisites:** Lessons 00–03. You saw tensors, axes, embeddings, attention, and gradients there; brief reminders appear here as you use them. Run the cells from top to bottom. Each notebook has separate memory, so this lesson does not depend on variables left over from the previous lesson.

The code in this lesson behaves like the implementation in `src/tinyllm`. The notebook version has more explicit loops and comments to make it easier to read. At the end, you will copy the weights and verify that both versions produce the same output.

<!-- cell -->

## 1. Prepare the Environment and Data

First, locate the project path. The loop starts in the current folder and checks its parents. The folder containing `src/tinyllm` is the project root. Adding `src` to `sys.path` lets Python find this project’s modules.

You set the number of CPU threads to two, which is suitable for an educational run of this small model. The fixed starting value for the random number generator, or random seed, is called `seed`. This makes repeated runs in the same environment easier to compare; hardware and library-version differences can still change the result.

<!-- cell -->

```python
# Import path, math, and class-definition utilities.
from pathlib import Path
from dataclasses import dataclass
import sys
import math

# PyTorch provides tensors and gradient computation.
import torch
from torch import nn
import torch.nn.functional as F

ROOT = None
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "src" / "tinyllm").is_dir():
        ROOT = candidate
        break
if ROOT is None:
    raise RuntimeError("Open this notebook inside the tiny-llm project")

source_path = str(ROOT / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)

torch.set_num_threads(2)
torch.manual_seed(7)
print("Project:", ROOT.name)
print("PyTorch:", torch.__version__)
```

<!-- cell -->

The `load_data` function reads the texts, tokenizer, and ID tensors. A tokenizer is a tool that converts between text and numeric IDs. In this project, each known character is one token. The small source dataset is already inside the project folder; this cell does not download anything.

You assign the function’s four outputs to four variables. This way of opening a multi-part result is called `unpacking`. In this lesson, you use the training data only to inspect the computation path; you do not change the weights yet.

<!-- cell -->

```python
from tinyllm.data import load_data

documents, texts, tokenizer, data = load_data(ROOT / "data")
print("Vocabulary IDs:", tokenizer.vocab_size)
print("Training IDs:", data["train"].shape)
print("First characters:", repr(texts["train"][:40]))
```

<!-- cell -->

The `Vocabulary IDs` output shows the number of possible IDs, including the unknown-character ID. In this version of the data, it is 83. The network must produce this many output scores for every input position. You use `V` as the short name for this number.

<!-- cell -->

## 2. Inspect a Trainable Layer Before Building the Model

A linear layer, or `Linear`, turns several input numbers into several output numbers. Each output is a weighted sum of the inputs; when `bias` is enabled, a trainable constant is added as well. “Linear” refers to this transformation.

Two inputs and three outputs require three rows of weights. In the example below, you set the weights by hand so the calculation can be checked: the input `[2, 3]` should produce `[2, 3, 5]`. The third row adds the two inputs.

<!-- cell -->

```python
# This layer takes 2 features and produces 3; it has no bias.
small_linear = nn.Linear(2, 3, bias=False)

# Manually setting weights is not training, so no gradient graph is needed.
with torch.no_grad():
    small_linear.weight.copy_(torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]))

small_input = torch.tensor([[2.0, 3.0]])
small_output = small_linear(small_input)
print("Weights:", small_linear.weight)
print("Output:", small_output)
print("Manual:", small_input @ small_linear.weight.T)
```

<!-- cell -->

The two outputs are identical. The `@` operator performs matrix multiplication, and `.T` swaps the rows and columns of a two-dimensional matrix. The library stores layer weights with shape `(out_features, in_features)`; for the manual calculation, you transpose them.

Model layers start with random weights. The training algorithm changes these weights later. When you call a layer like a function, for example `small_linear(small_input)`, PyTorch manages its execution path.

<!-- cell -->

## 3. The Model Class and Its Configuration

A Python class defines a type of object. An instance contains data and behavior. Here, you keep configuration in `Config` and the network in `TinyGPT`. The configuration determines which network is built; the configuration itself is not a trainable weight.

Keep these names in mind when reading shapes:

| Name | Meaning | Example in the main model |
|---|---|---|
| Batch size, `B` | How many texts are processed at once | One text in this lesson’s example |
| Input length, `T` | How many tokens from each text are provided | Eight tokens in the example |
| Vector width, `C` | How many features each token has | 64 |
| Number of IDs, `V` | How many choices exist for the next token | 83 |
| Number of heads, `H` | How many independent attention heads are combined | Four |

The `context` setting is the maximum input length; `T` itself can be smaller. The `layers` setting is the number of consecutive blocks. `dropout` refers to randomly setting some values to zero during training so the network depends less on one particular path.

<!-- cell -->

```python
# dataclass generates the class constructor from these fields.
# frozen=True prevents accidental changes to this instance’s fields later.
@dataclass(frozen=True)
class Config:
    vocab: int                # Number of token IDs, including the unknown ID
    context: int = 64         # Maximum input length
    width: int = 64           # Number of features per token
    heads: int = 4            # Number of attention heads
    layers: int = 2           # Number of Transformer blocks
    dropout: float = 0.1      # Probability of dropping a value during training

    def __post_init__(self):
        # This function runs after an instance is created.
        sizes = [self.vocab, self.context, self.width, self.heads, self.layers]
        if min(sizes) <= 0:
            raise ValueError("All dimensions must be positive")
        # The remainder must be zero so every head receives an equal share.
        if self.width % self.heads != 0:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

config = Config(vocab=tokenizer.vocab_size)
print(config)
print("Features per head:", config.width // config.heads)
```

<!-- cell -->

Writing `vocab: int` tells the reader the expected type; Python does not enforce every type at runtime from this annotation alone. `%` gives the remainder, and `//` performs integer division. With width 64 and four heads, each head has 16 output features.

Catch the invalid-configuration error deliberately so the notebook keeps running. Building a network with width 65 and four heads violates this model’s contract.

<!-- cell -->

```python
try:
    invalid_config = Config(vocab=83, width=65, heads=4)
except ValueError as error:
    # Catching the error here is part of observing the expected behavior.
    print("Expected error:", error)
```

<!-- cell -->

## 4. Turn an Attention Head into a Trainable Object

In Lesson 03, you computed attention with small tensors. Now three linear layers create query, key, and value vectors. A query says what information each position wants; a key is used for comparison; a value is the information combined by the attention weights. These describe the vectors’ roles; training learns their actual values.

The `nn.Module` class is the base for network components in PyTorch. Inheriting from it and calling `super().__init__()` prepares the machinery that registers layers and weights. When you place a layer in `self.query`, the library registers its weights in the model.

A causal mask is a table that blocks looking into the future. You register it with `register_buffer`: it is saved and moved with the model, but it is not a trainable parameter. This distinction matters for saving the model correctly.

<!-- cell -->

```python
class Head(nn.Module):
    def __init__(self, c):
        super().__init__()                     # Prepare the mechanism that registers network components
        head_size = c.width // c.heads         # Feature share for one head
        self.query = nn.Linear(c.width, head_size, bias=False)
        self.key = nn.Linear(c.width, head_size, bias=False)
        self.value = nn.Linear(c.width, head_size, bias=False)

        # The lower triangle and diagonal are allowed; the upper triangle is the future.
        allowed = torch.tril(torch.ones(c.context, c.context)).bool()
        self.register_buffer("mask", allowed)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        # The input has shape (B, T, C); axis 1 is the sequence length.
        time = x.shape[1]
        q = self.query(x)                      # Queries for all positions
        k = self.key(x)                        # Keys for all positions
        v = self.value(x)                      # Values that can be combined

        # Move the last two key axes so every query is compared with every key.
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.shape[-1])
        # ~ inverts True and False in a Boolean tensor.
        future = ~self.mask[:time, :time]
        scores = scores.masked_fill(future, float("-inf"))
        # Convert scores to attention weights along the key axis.
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v                    # Weighted sum of the values
```

<!-- cell -->

The `__init__` method creates the components once. The `forward` method defines what happens each time data enters the module. When you write `head(x)`, PyTorch calls `forward` through the module execution path.

Queries, keys, and values have shape `(B, T, C/H)`. The score table has shape `(B, T, T)`: one score for every input position against every position it may read. Negative infinity becomes zero after `softmax`, so future information cannot enter the sum.

To inspect the dimensions, create a small head with four input features and two output features. Disable dropout in this example so the result is easy to interpret.

<!-- cell -->

```python
tiny_config = Config(vocab=10, context=4, width=4, heads=2, layers=1, dropout=0.0)
example_head = Head(tiny_config)
example_hidden = torch.randn(1, 3, 4)       # One sequence, three positions, four features
head_output = example_head(example_hidden)
print("Input:", tuple(example_hidden.shape))
print("Output:", tuple(head_output.shape))
print("Allowed positions:", example_head.mask[:3, :3].int())
print("Trainable names:", [name for name, _ in example_head.named_parameters()])
print("Saved names:", list(example_head.state_dict()))
```

<!-- cell -->

The output shape is `(1, 3, 2)`. In the allowed table, row one sees only column one; row three sees all three columns. `mask` appears in the saved `state_dict`, but it is not an optimizable parameter.

The bracket expression, such as `[name for name, _ in ...]`, is a compact loop for building a list. Here `_` means that you do not need the second value; it is a conventional placeholder name.

<!-- cell -->

## 5. How Do Dropout Training and Evaluation Differ?

`train()` activates training mode and `eval()` activates evaluation mode. These names do not themselves train weights or measure quality; they change the behavior of layers such as `Dropout`. Weight updates still require gradients and an optimizer step.

In training mode, `Dropout` sets some values to zero and scales the remaining values to preserve the expected mean. With a dropout probability of 0.5, for example, the remaining values are doubled. In evaluation mode, the layer returns its input unchanged.

<!-- cell -->

```python
dropout_example = nn.Dropout(0.5)
ones = torch.ones(12)

dropout_example.train()
print("Training:", dropout_example(ones))

dropout_example.eval()
print("Evaluation:", dropout_example(ones))
```

<!-- cell -->

Training output contains zeros and twos; the number of zeros can vary between runs. In evaluation, every value is one. When this operation is applied to attention weights, a row does not necessarily sum to one during training. In evaluation, dropout is disabled and each `softmax` row still sums to one.

`torch.no_grad()` serves a different purpose: it disables recording operations for gradients. Text generation usually needs both `eval()` for stable behavior and `no_grad()` to avoid the cost of gradient computation.

<!-- cell -->

## 6. Build a Transformer Block

A block first applies several attention heads. Their outputs are concatenated to rebuild width `C`. A linear layer combines these features, and the result is added to the previous input. This sum is a `residual connection`; it preserves a direct path for information and gradients.

The next part is the small feature-processing network called `mlp`. It first expands the width fourfold, applies the nonlinear function `GELU`, and projects back to the original width. “Nonlinear” means its behavior is more than a simple weighted sum. This part transforms each position independently; attention handles communication between positions.

`LayerNorm` normalizes the feature values at each position using that vector’s own mean and spread, then applies a learned scale and shift. For example, it operates on a four-value vector rather than on all sequences in the batch together. In this model it comes before each main sublayer, which is called `pre-norm`.

<!-- cell -->

```python
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()

        # ModuleList registers the heads; a regular list would not do this.
        attention_heads = []
        for _ in range(c.heads):
            attention_heads.append(Head(c))
        self.heads = nn.ModuleList(attention_heads)

        self.project = nn.Linear(c.width, c.width)
        self.attention_dropout = nn.Dropout(c.dropout)
        # Sequential applies components in order to the previous component’s output.
        self.mlp = nn.Sequential(
            nn.Linear(c.width, 4 * c.width),
            nn.GELU(),
            nn.Linear(4 * c.width, c.width),
            nn.Dropout(c.dropout),
        )
        self.norm1 = nn.LayerNorm(c.width)
        self.norm2 = nn.LayerNorm(c.width)

    def forward(self, x):
        normalized = self.norm1(x)              # Normalize the features at each position
        head_results = []
        for head in self.heads:
            head_results.append(head(normalized))
        joined = torch.cat(head_results, dim=-1) # Concatenate along the feature axis
        attention_result = self.project(joined)
        attention_result = self.attention_dropout(attention_result)
        x = x + attention_result                # Attention residual connection
        feature_result = self.mlp(self.norm2(x))
        return x + feature_result               # Feature-processing residual connection
```

<!-- cell -->

`dim=-1` selects the last axis. Here two outputs of shape `(1, 3, 2)` become an output of shape `(1, 3, 4)`. Every block input and output has shape `(B, T, C)`. Keeping the shape unchanged lets you stack blocks without changing the input contract.

The loops spell out the same operations that the module version expresses more compactly. `ModuleList` parameters appear in `model.parameters()`, so the optimizer can update them.

<!-- cell -->

```python
example_block = Block(tiny_config)
example_block.eval()                          # Disable dropout
block_output = example_block(example_hidden)
print("Block input:", tuple(example_hidden.shape))
print("Block output:", tuple(block_output.shape))
assert block_output.shape == example_hidden.shape
```

<!-- cell -->

This experiment does not measure model quality; it checks the shape contract. Output values changed, but the batch size, number of positions, and vector width stayed fixed.

<!-- cell -->

## 7. Connect Embeddings, Blocks, and the Output

The model input is integer token IDs with shape `(B, T)`. The token embedding table maps each ID to a vector of length `C`. A separate position table provides a vector for each position; adding the two tells the network both “which token?” and “at which position?”.

The position table has shape `(T, C)`. When it is added to `(B, T, C)`, PyTorch applies it to every batch item. This is called `broadcasting`, or automatic expansion for compatible operations. The position table is learned in this model; its values are not fixed sinusoidal values.

The final layer produces `V` scores for each position. These raw scores are called `logits`. To compute training loss, compare them with the correct token IDs. The model does not return text; converting scores into token choices and text happens during generation.

<!-- cell -->

```python
class TinyGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.token = nn.Embedding(c.vocab, c.width)       # Token table
        self.position = nn.Embedding(c.context, c.width) # Position table

        block_list = []
        for _ in range(c.layers):
            block_list.append(Block(c))
        # * expands a list into separate constructor arguments.
        self.blocks = nn.Sequential(*block_list)
        self.norm = nn.LayerNorm(c.width)
        self.output = nn.Linear(c.width, c.vocab)

    def forward(self, ids, targets=None):
        # The input must have two axes and its length must fit the configuration.
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context:
            raise ValueError("ids must have shape (B, T), with 1 <= T <= context")

        # Positions always start at zero within this window.
        positions = torch.arange(ids.shape[1], device=ids.device)
        token_vectors = self.token(ids)
        position_vectors = self.position(positions)
        hidden = token_vectors + position_vectors
        hidden = self.blocks(hidden)
        logits = self.output(self.norm(hidden))

        loss = None                            # No loss without a target
        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError("targets must have the same shape as ids")
            # Flatten each position into one row of choice scores.
            flat_logits = logits.reshape(-1, self.config.vocab)
            flat_targets = targets.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_targets)
        return logits, loss
```

<!-- cell -->

In `reshape`, `-1` means that dimension is inferred from the total number of elements. For an input of `(2, 3)` and a vocabulary of ten, `(2, 3, 10)` becomes `(6, 10)`. The six targets also become a vector of `(6,)`. `cross_entropy` measures loss for each row and averages by default.

Do not apply a separate `softmax` before this function; it accepts raw scores and performs the required operation stably. `loss` is a scalar. A larger value means the model assigned less probability to the correct targets. Lesson 05 uses this value to compute gradients.

<!-- cell -->

## 8. Build the Input and Correct Target

To predict the next token, the target is shifted one position ahead of the input. If the input is “bab”, the targets begin at the second character. In the real data, the first window uses IDs zero through seven and the target window uses IDs one through eight.

`unsqueeze(0)` adds an axis at the beginning of a tensor. An eight-ID vector has shape `(8,)`; to represent one eight-token example, convert it to `(1, 8)`.

<!-- cell -->

```python
model = TinyGPT(config)                        # Weights are still random
x = data["train"][:8].unsqueeze(0)             # Input: eight tokens
y = data["train"][1:9].unsqueeze(0)            # Target: one position ahead
print("Input IDs:", x.tolist())
print("Target IDs:", y.tolist())
print("Input text:", repr(tokenizer.decode(x[0].tolist())))
print("Target text:", repr(tokenizer.decode(y[0].tolist())))

logits, loss = model(x, y)
print("Logits:", tuple(logits.shape))
print("Loss:", loss.item())
assert logits.shape == (1, 8, tokenizer.vocab_size)
```

<!-- cell -->

For the current data, the output has shape `(1, 8, 83)`: one sequence, eight positions, and 83 scores per position. A random model usually has a loss near the logarithm of the number of choices, but random weights do not necessarily produce a uniform distribution, so do not expect exact equality.

For a numerical example, suppose the model assigns equal probability to all 83 choices. The loss for each correct target is `-log(1/83)`, which equals `log(83)`.

<!-- cell -->

```python
uniform_logits = torch.zeros(1, tokenizer.vocab_size)  # Equal score for every choice
one_target = torch.tensor([1], dtype=torch.long)       # One correct choice
uniform_loss = F.cross_entropy(uniform_logits, one_target)
print("Uniform loss:", uniform_loss.item())
print("log(V):", math.log(tokenizer.vocab_size))

# Without a target, the network returns only scores.
scores_only, no_loss = model(x)
print("No targets -> loss:", no_loss)
```

<!-- cell -->

## 9. Observe Data Moving Through the Model

`register_forward_hook` attaches an observer function to a layer; the observer runs after the layer executes. Here, a `hook` is simply a point where you can inspect the computation path. You use it only to print output shapes, not to modify the network output.

The small function below creates an observer with a chosen label. This preserves each layer name when the function is created. Every observer returns a handle for removal; remove the handles after the experiment so later runs do not print duplicates.

<!-- cell -->

```python
def make_shape_observer(label):
    def observe(module, inputs, result):
        print(label, tuple(result.shape))
    return observe

handles = []
watched_layers = [
    ("token", model.token),
    ("position", model.position),
    ("blocks", model.blocks),
    ("output", model.output),
]
try:
    for name, module in watched_layers:
        handle = module.register_forward_hook(make_shape_observer(name))
        handles.append(handle)
    observed_logits, observed_loss = model(x, y)
finally:
    # Remove installed observers even if computation fails.
    for handle in handles:
        handle.remove()
```

<!-- cell -->

The token table outputs `(1, 8, 64)` and the position table outputs `(8, 64)`. The blocks preserve `(1, 8, 64)`. The output layer converts it to `(1, 8, 83)`. These four prints expose the path “ID, vector, processing, score.”

<!-- cell -->

## 10. Count Parameters and Weight Memory

A parameter is a trainable value. The weights of a linear layer and the vectors in an embedding table are parameters; a causal mask is not. A parameter tensor may contain thousands of elements; use `numel()` to count them.

`named_parameters()` returns each parameter name and tensor. Inspect a few examples first, then count them all. This count does not depend on the number of input positions in this cell; the architecture was fixed when the model was created.

<!-- cell -->

```python
measured = 0
for parameter in model.parameters():
    measured += parameter.numel()
print("Trainable numbers:", measured)

shown = 0
for name, parameter in model.named_parameters():
    print(name, tuple(parameter.shape), parameter.numel())
    shown += 1
    if shown == 8:
        break

# float32 uses four bytes per element; one MiB is 2^20 bytes.
weights_mib = measured * 4 / (2 ** 20)
print("FP32 weights MiB:", weights_mib)
```

<!-- cell -->

With the default configuration, there are 114,515 trainable values. The estimate covers weights only. During training, gradients, optimizer state, and intermediate activations also require memory. For example, `AdamW` usually keeps two auxiliary value sets per parameter; you will use it in the next lesson.

To connect the count to the architecture, calculate each part separately. A linear layer with `C` inputs and `D` outputs has `C×D` weights and, when bias is enabled, another `D` values.

| Component | Number of trainable values |
|---|---|
| Token table | `V × C` |
| Position table | `context × C` |
| All query, key, and value projections in one block | `3 × C²` |
| Head-output projection | `C² + C` |
| Two feature-processing linear layers | `8 × C² + 5 × C` |
| Two normalizations in one block | `4 × C` |
| Final normalization | `2 × C` |
| Output layer | `C × V + V` |

Each normalization has one scale and one shift per feature. The token table and output weights are separate in this model; some models share them using `weight tying`, but this implementation does not.

<!-- cell -->

```python
V = config.vocab
C = config.width
L = config.layers
context = config.context

one_block = 12 * C * C + 10 * C
formula = (V * C) + (context * C) + (L * one_block) + (2 * C) + (C * V + V)
print("From architecture:", formula)
print("From tensors:", measured)
assert formula == measured
```

<!-- cell -->

## 11. Generate a Continuation from Scores

One `forward` pass only produces scores. To generate a continuation, take the last position, select an ID from its scores, append that ID to the input, and run the network again. This loop is `autoregressive generation`: the previous output becomes part of the next input.

Three settings affect selection. In this project, temperature zero always selects the highest-scoring option; this is called `greedy` decoding. A positive temperature divides scores before `softmax`; a lower temperature concentrates selection on high-scoring options. `top_k` first removes low-scoring options; zero disables this filter.

The function below keeps the entire generated text in `ids`, but feeds only the last `context` tokens to the network at each step. The final text can therefore be longer than the model window. Information outside the window is not directly visible.

<!-- cell -->

```python
# This decorator disables gradient-graph recording for the entire function.
@torch.no_grad()
def generate(model, ids, count=80, temperature=0.8, top_k=20, seed=7):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Generation expects one non-empty prompt: (1, T)")
    if count < 0 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("count, temperature and top_k must be non-negative and finite")

    # A local generator separates sampling from the training random stream.
    rng = torch.Generator(device=ids.device).manual_seed(seed)
    was_training = model.training             # Remember the previous mode
    model.eval()                              # Keep dropout disabled during generation
    try:
        for _ in range(count):
            window = ids[:, -model.config.context:]
            logits, _ = model(window)         # We do not use loss
            scores = logits[:, -1, :]         # Scores after the last token

            if temperature == 0:
                # keepdim preserves the new-token axis: shape (1, 1).
                next_id = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k > 0:
                    available = min(top_k, scores.shape[-1])
                    best_scores = scores.topk(available).values
                    cutoff = best_scores[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                probs = F.softmax(scores, dim=-1)
                next_id = torch.multinomial(probs, 1, generator=rng)

            # Axis 1 is time; append one token to its end.
            ids = torch.cat([ids, next_id], dim=1)
        return ids
    finally:
        # Restore the pre-generation mode even if an error occurs.
        model.train(was_training)
```

<!-- cell -->

`torch.multinomial` samples according to probabilities; it does not necessarily choose the highest probability. In the `top_k` filter, if several scores equal the cutoff, more than `k` options may remain. These details describe this implementation.

The `finally` block runs after normal completion or an error. Here it prevents the model mode from being changed for the rest of the notebook. This model has no special end-of-sequence token named `EOS`; the requested number of new tokens is the stopping condition.

<!-- cell -->

```python
prompt = "The model "
prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
generated = generate(model, prompt_ids, count=80, temperature=0.8, top_k=20, seed=7)
generated_text = tokenizer.decode(generated[0].tolist())
print(generated_text)
print("Initial length:", prompt_ids.shape[1])
print("Final length:", generated.shape[1])
assert generated.shape[1] == prompt_ids.shape[1] + 80
```

<!-- cell -->

Gibberish is expected at this stage; no training step has happened yet. This experiment checks only the connection between input, network, selection, and text conversion. In Lesson 05, you will compare this path before and after training.

In this function, the window is truncated when the input grows. The Lesson 06 API rejects an initial input that is too long so the user sees the limit. This model has no key-value storage, or `KV cache`; it recomputes the whole window at every step. Lesson 03 showed a small cache example. Using a cache correctly also requires position management and window-limit handling.

<!-- cell -->

## 12. Does the Educational Version Match the Module Version?

The reusable version for training and serving is in `src/tinyllm/model.py`. Parameter names match between the two versions, so you can obtain the notebook weights with `state_dict()` and load them into the package model with `load_state_dict()`.

Run this comparison in evaluation mode. In training mode, dropout can make separate runs differ. `assert_close` raises an error when the difference exceeds the allowed tolerance. With zero tolerances, you check exact equality.

<!-- cell -->

```python
from tinyllm.model import TinyGPT as PackageGPT
from tinyllm.model import generate as package_generate

deployed = PackageGPT(config)                  # The same architecture in the package
deployed.load_state_dict(model.state_dict())   # The same weights and buffers
model.eval()
deployed.eval()

with torch.no_grad():
    notebook_scores, _ = model(x)
    package_scores, _ = deployed(x)
torch.testing.assert_close(notebook_scores, package_scores, rtol=0, atol=0)

# Sampling with the same seed and settings should also produce the same IDs.
local_ids = generate(model, prompt_ids, count=12, seed=11)
package_ids = package_generate(deployed, prompt_ids, count=12, seed=11)
assert torch.equal(local_ids, package_ids)
print("Notebook/package forward and generation agree.")
```

<!-- cell -->

`as PackageGPT` only changes the imported name so it is not confused with the local class. The printed message means that the two versions produced identical scores and continuations from identical weights and input; this experiment makes no claim about language quality.

<!-- cell -->

## 13. The Complete Model Must Not See the Future Either

You tested the mask in one head; now check the result for the whole network. Create two inputs with the same first four tokens and different later tokens. The scores at the first four positions should remain equal. If later changes affect them, the computation path has allowed an invalid dependency.

<!-- cell -->

```python
first = x.clone()                             # Independent copy of the same input
second = x.clone()
second[:, 4:] = 0                             # Change only the future half

with torch.no_grad():
    first_scores, _ = model(first)
    second_scores, _ = model(second)
torch.testing.assert_close(first_scores[:, :4], second_scores[:, :4])
print("Changing future tokens did not change earlier predictions.")
```

<!-- cell -->

This property comes from the attention mask. The feature-processing layer and normalization also process each position independently, so they do not create a hidden path for future information. Seeing the correct future target during training can make loss look low while breaking the real generation task; that is why this check matters.

<!-- cell -->

## 14. Inspect a Small Change Yourself

Reduce the number of blocks from two to one. Before running the cell, predict the output shape and how many parameters will be removed. The input and output contract stays the same, but one block with `12 × C² + 10 × C` parameters is removed.

This change creates a new architecture; do not transfer weights trained for the two-block model to the one-block model without checking compatibility. In the next lesson, every training experiment should record its own architecture and settings.

<!-- cell -->

```python
one_layer_config = Config(vocab=tokenizer.vocab_size, layers=1, dropout=0.0)
one_layer_model = TinyGPT(one_layer_config)
one_layer_count = sum(parameter.numel() for parameter in one_layer_model.parameters())
one_layer_scores, _ = one_layer_model(x)
print("Output shape:", tuple(one_layer_scores.shape))
print("Removed parameters:", measured - one_layer_count)
print("One block:", one_block)
assert measured - one_layer_count == one_block
```

<!-- cell -->

## Lesson Review and Interview Practice

Answer in your own words first, then open the suggested answer:

1. What is the difference between building a network and training it?

2. What are the shapes of the `TinyGPT` input, `Head` output, `Block` output, and model scores?

3. Why do you register the mask as a buffer while the query weight is a parameter?

4. Why are both evaluation mode and disabled gradients useful for text generation?

5. Why do you not apply `softmax` yourself before `cross_entropy`?

6. Which axis does the feature-processing part combine?

7. Why can the number of output tokens exceed `context`?

<!-- fa-lint-ignore-next-line -->
<details><summary>Open the suggested answers</summary>

1. Building the network determines its components, weight shapes, and computation path. Training changes weight values using data, loss, gradients, and an optimizer.

2. The model input is `(B, T)`, a head outputs `(B, T, C/H)`, a block outputs `(B, T, C)`, and scores have shape `(B, T, V)`.

3. The mask is a fixed rule that permits looking at the past and is not trained, but it must remain with the model state. The query weight changes through gradients.

4. Evaluation mode disables dropout; disabling gradients stops recording operations for derivatives. Neither replaces the other.

5. The loss function expects raw scores and performs the conversion and loss calculation internally. Passing probabilities instead of scores changes its contract.

6. This part transforms the features at each position; attention creates connections between positions.

7. The loop keeps all generated IDs, but feeds only the latest window to the model for each prediction. The stored text length differs from the input length at each step.

</details>

**Next:** In [Lesson 05, Training and Evaluation](05_train_evaluate_and_checkpoint.ipynb), you will connect gradients to this network and measure loss reduction by comparing samples before and after training.
