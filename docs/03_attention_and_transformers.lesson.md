# 03. How Does Attention Combine Information from Tokens?

In the previous lesson, you converted each token into a vector. A vector is a list of numbers; for example, two numbers can represent one token. We now face a new question: a token representation should also depend on the tokens around it.

In the sentence “Maryam gave the book to Sara because she…”, the meaning of “she” cannot be determined from that word alone. Some of the necessary information is in earlier positions. **Attention** is an operation that creates a weighted combination of information from other positions for each position. During training, the model learns which weights this combination should use.

You will first work with small tables of numbers; you are not building a language model yet. Then you will connect the same calculation to one attention head, multiple heads, and a Transformer block. The final section explains why text generation is expensive and how caching previous computations can help.

**Prerequisite:** Lessons 00–02. You only need to remember tensors, tensor shapes, matrix multiplication, and embedding tables. Whenever we use one of these ideas, we will check the input and output shapes again. You do not need to memorize formulas before running the examples.

**Study method:** Run the cells from top to bottom. Pause briefly after the causal-mask section and read the outputs again. This lesson downloads no data or model and does not require a GPU.

<!-- cell -->
## 1. Prepare the environment

To compare our calculation with the project code, we first locate the project directory. `ROOT` stores the directory that contains `src/tinyllm`. Adding this directory to `sys.path` lets Python import files from this project.

`F` is a short name for PyTorch’s neural-network function module. We will later use its `softmax` and `scaled_dot_product_attention` functions. The `math` module also provides the square-root function, `sqrt`.

**Code goal:** Prepare an environment where numerical examples and the project’s real attention code can run in one kernel.

<!-- cell -->

```python
# Path makes directory operations easier.
from pathlib import Path
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F

# Start at the current directory and also check its parent directories.
ROOT = None
candidate_folders = [Path.cwd()] + list(Path.cwd().parents)
for candidate in candidate_folders:
    if (candidate / "src" / "tinyllm").is_dir():
        ROOT = candidate
        break
if ROOT is None:
    raise RuntimeError("Open this notebook from inside tiny-llm-fa.")

# Add the project source path to Python only when needed.
source_path = str(ROOT / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)

# Two CPU threads are enough for these small examples.
torch.set_num_threads(2)
torch.manual_seed(7)
torch.set_printoptions(precision=4, sci_mode=False)
print("Project:", ROOT.name)
print("PyTorch:", torch.__version__)
```

<!-- cell -->
The project name should be `tiny-llm-fa`. The version may include a `+cpu` suffix. `precision=4` changes only how many digits are displayed; it does not change the precision of numbers inside tensors. Setting the seed makes the random choices in this run more reproducible.

## 2. Understand a weighted average before attention

Suppose we have three information sources. The first has vector `[10, 0]`, the second has `[0, 20]`, and the third has `[10, 20]`. These numbers are artificial; we assign them no linguistic meaning in this example.

If every source has a weight of one third, you compute a simple average. If the weights differ, you compute a **weighted average**: multiply each vector by its weight and add the results. In attention, weights are non-negative and sum to one for each position.

**Code goal:** Compute the combination of three vectors with both an explicit sum and matrix multiplication.

<!-- cell -->

```python
# Each row is a source; each column is a numeric feature.
values = torch.tensor([[10.0, 0.0], [0.0, 20.0], [10.0, 20.0]])

# Assign 20%, 30%, and 50% weights, respectively.
mix_weights = torch.tensor([0.2, 0.3, 0.5])

# Explicit method: multiply each row by its weight and then add the rows.
mixed_by_sum = 0.2 * values[0] + 0.3 * values[1] + 0.5 * values[2]

# Matrix method: perform the same operation for both features at once.
mixed_by_matrix = mix_weights @ values
print("Weighted sum:", mixed_by_sum)
print("Matrix version:", mixed_by_matrix)
torch.testing.assert_close(mixed_by_sum, mixed_by_matrix)
```

<!-- cell -->
Both methods produce `[7, 16]`. The first feature comes from `0.2*10 + 0.3*0 + 0.5*10`. The second comes from `0.2*0 + 0.3*20 + 0.5*20`.

The `@` operator performs matrix multiplication. In the example above, a length-three vector is multiplied by a 3-by-2 table to produce a length-two vector. `assert_close` checks that the results are equal within the tiny error expected from floating-point arithmetic. If the difference is large, the cell stops with an error.

**Next question:** Where do the 20%, 30%, and 50% weights come from? Attention computes these weights from the input.

## 3. What are query, key, and value?

For each position, we create three vectors. The **query** vector is compared with the **key** vectors at other positions to produce scores. The **value** vector contains the information that is combined with the weights at the end. In code, their short names are `q`, `k`, and `v`.

A search analogy helps: you have a question, features for comparing sources, and content to retrieve from each source. These are not a text question, a dictionary key, and a dictionary value. All three are numeric vectors. Their names only describe their roles in the calculation.

For now, we write the vectors by hand so that the scores are easy to calculate. In a real model, all three are created from the input representation and trainable weights; we will code that part shortly.

**Code goal:** Create three positions, each with a two-dimensional query and key vector.

<!-- cell -->

```python
# We have three positions; each query vector has two numbers.
q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

# In this simple example, we choose keys equal to the queries.
k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

# The value vectors are the same three sources from the previous section.
v = values.clone()
print("Query shape:", tuple(q.shape))
print("Key shape:", tuple(k.shape))
print("Value shape:", tuple(v.shape))
```

<!-- cell -->
All three shapes are `(3, 2)`: three positions and two numbers per vector. Making `q` and `k` equal is only a teaching choice in this cell; a real network usually gives them separate weights. In this course the value dimension also equals the key dimension, but that equality is not required by every architecture.

### How does a dot product produce a score?

To compare two vectors of the same size, multiply corresponding components and add the results. This operation is the **dot product**. For `[1, 1]` and `[1, 0]`, the result is `1*1 + 1*0 = 1`.

This score depends on the vectors’ magnitudes and directions. A larger score receives more weight in this calculation; by itself, it is neither a percentage of semantic similarity nor the probability that a claim is true.

**Code goal:** Compare the query at the third position with all three keys.

<!-- cell -->

```python
# Index 2 is the third position; indexing starts at zero.
query_last = q[2]
scores_last = []
for key_index in range(k.shape[0]):
    # * multiplies component by component; sum adds the products.
    score = (query_last * k[key_index]).sum()
    scores_last.append(score.item())
print("Last query:", query_last)
print("Raw scores:", scores_last)
```

<!-- cell -->
The scores are `[1, 1, 2]`. The third query scores higher with the third key. These are not weights yet; the scores sum to four, and a negative input could produce a negative score.

### Compare every query with every key at once

In `q @ k.T`, each row of `q` is multiplied by each column of `k.T`. The `T` attribute swaps the rows and columns of a two-dimensional matrix. The result is a 3-by-3 table: **a row is a query position and a column is an information-source position.**

**Code goal:** View the same hand-computed scores as a matrix for every position.

<!-- cell -->

```python
# A (3, 2) shape multiplied by a (2, 3) shape produces a (3, 3) shape.
raw_scores = q @ k.T
print("All raw scores:\n", raw_scores)
print("Last row:", raw_scores[2])
```

<!-- cell -->
The last row is again `[1, 1, 2]`. The cell at row zero and column one is zero: the first query `[1, 0]` has a dot product of zero with the second key `[0, 1]`. Keep this row-and-column convention in mind; the mask and the diagram use it too.

## 4. From scores to weights with softmax

In Lesson 01, **softmax** converted output scores into probabilities for the next token. Here we apply the same function to attention scores, but the options are not vocabulary tokens; **they are positions in this input text.**

For each score `s`, softmax computes `exp(s)` and divides it by the sum of all exponentials in that row. The exponential is positive, and division by the sum makes the weights add up to one. The score ordering is preserved.

Before softmax, we divide the scores by the square root of the key dimension. Here the key dimension is two, whose square root is about 1.414. We will examine the reason after seeing the result.

**Code goal:** See two ways to convert the last row of scores into weights.

<!-- cell -->

```python
# The number of features in each key is d_k in the attention formula.
key_dimension = k.shape[-1]
scaled_scores = raw_scores / math.sqrt(key_dimension)

# Explicit method for one row: exponentiate, then divide by the sum.
exponentials = torch.exp(scaled_scores[2])
manual_weights = exponentials / exponentials.sum()

# Library method for every row; the last axis is the source axis.
attention_weights = F.softmax(scaled_scores, dim=-1)
print("Scaled last row:", scaled_scores[2])
print("Manual weights:", manual_weights)
print("Softmax weights:", attention_weights[2])
print("Row sums:", attention_weights.sum(dim=-1))
torch.testing.assert_close(manual_weights, attention_weights[2])
```

<!-- cell -->
The last-row weights are approximately `[0.2483, 0.2483, 0.5035]`. They sum to one; the third key receives the largest weight, but the other two are not removed. Standard attention does not make a hard choice of one source; it combines information from several sources.

`dim=-1` means “the last axis.” In a 3-by-3 table, the last axis is the columns. If you mistakenly apply softmax over the row axis, you will not create a distribution over sources for each query, and the meaning of the calculation changes.

### Weights combine the values

We now multiply the weight matrix by `v`. This creates one output vector for each query. Each component is the weighted sum of that component across the different sources.

**Code goal:** Compute the attention output for every position and manually for the last position.

<!-- cell -->

```python
# A (3, 3) shape multiplied by a (3, 2) shape produces a (3, 2) shape.
attention_output = attention_weights @ v

# Explicit method for the last query, to check what the matrix multiplication means.
last_output = torch.zeros(2)
for source_index in range(3):
    last_output = last_output + attention_weights[2, source_index] * v[source_index]
print("All outputs:\n", attention_output)
print("Last output by hand:", last_output)
torch.testing.assert_close(last_output, attention_output[2])
```

<!-- cell -->
The output at the last position is approximately `[7.5174, 15.0349]`. The third value vector `[10, 20]` contributed the most, but the output is not exactly equal to it. These numbers are not text yet; they are new numeric representations of each position for later layers to process.

You can now read the compact attention formula:

$$\operatorname{Attention}(Q،K،V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The uppercase `Q`, `K`, and `V` refer to the tables of all vectors. `QK^T` creates scores, scaling adjusts their magnitude, softmax creates weights, and the final multiplication combines the values.

## 5. A continuation model must not see the future

This course’s language model should predict the next token from the text already available. During training, we provide a complete text segment at once. If the first position can see the second position, it can read the answer to “What comes after the first token?” directly from the input. The loss decreases, but this information is unavailable during generation.

To prevent this, we create a **causal mask**. A mask is a table of `True` and `False` values. Allowed cells contain the current position and its past; future positions are forbidden. `tril` keeps the lower triangle of a matrix, including its diagonal.

**Code goal:** View the table of allowed connections.

<!-- cell -->

```python
# The initial 3-by-3 table has one in every cell.
allowed = torch.tril(torch.ones(3, 3)).bool()
print("Allowed positions:\n", allowed)

# The first row sees only itself; the second sees itself and one earlier position.
print("First query may read:", allowed[0].tolist())
print("Second query may read:", allowed[1].tolist())
```

<!-- cell -->
The rows are `[True, False, False]`, `[True, True, False]`, and `[True, True, True]`. This table does not infer linguistic meaning; it only specifies the access rule for positions.

To apply the mask, set the scores of forbidden cells to negative infinity. The exponential of negative infinity is zero, so softmax assigns zero weight to the future. `~` inverts Boolean values: an allowed cell becomes `False` and a forbidden cell becomes `True`, allowing `masked_fill` to fill only forbidden cells.

**Code goal:** See how masking before softmax makes future probabilities zero.

<!-- cell -->

```python
# Fill only cells that are forbidden in the allowed table.
causal_scores = scaled_scores.masked_fill(~allowed, float("-inf"))
causal_weights = F.softmax(causal_scores, dim=-1)
causal_output = causal_weights @ v
print("Masked scores:\n", causal_scores)
print("Causal weights:\n", causal_weights)
print("Causal output:\n", causal_output)

# The weights for each query sum to one, and future weights are exactly zero.
torch.testing.assert_close(causal_weights.sum(-1), torch.ones(3))
assert torch.equal(causal_weights.triu(1), torch.zeros(3, 3))
```

<!-- cell -->
The first row’s weights are `[1, 0, 0]` and its output is `[10, 0]`. Its only allowed source is itself. The second row splits its weight across the first and second sources; the third source has zero weight. The last row has the same weights as before because every source is allowed for the last position.

### View the weight table as an image

In the next image, each square is a weight. A darker cell has a larger weight. The vertical axis is the query position and the horizontal axis is the source position, matching the table’s row-and-column convention. Future cells are above the diagonal and have zero weight.

**Code goal:** View the triangular causal-attention structure alongside its numeric values.

<!-- cell -->

```python
import matplotlib.pyplot as plt

# numpy converts the numeric table into a format the plotting tool can read.
figure, axis = plt.subplots(figsize=(5, 4))
image = axis.imshow(causal_weights.numpy(), cmap="Greys", vmin=0, vmax=1)
axis.set_xlabel("Key / source position")
axis.set_ylabel("Query position")
axis.set_xticks([0, 1, 2])
axis.set_yticks([0, 1, 2])

# Write each cell’s value on the cell itself.
for row in range(3):
    for column in range(3):
        weight = causal_weights[row, column].item()
        text_color = "white" if weight > 0.5 else "black"
        axis.text(column, row, f"{weight:.2f}", ha="center", va="center", color=text_color)
figure.colorbar(image, ax=axis, label="Attention weight")
plt.show()
```

<!-- cell -->
The first row has only one non-zero cell, while the last row can access all three sources. Color is only a way to see the weights; this cell creates no new model or data.

### Why not zero future weights after softmax?

If you apply softmax to all sources first and then remove future weights, the remaining weights usually sum to less than one. You could renormalize after removal, but the clear and standard approach is to mask the scores and apply softmax once.

**Code goal:** See the effect of removing future weights without renormalizing.

<!-- cell -->

```python
# This method is intentionally incomplete: it removes weights after softmax.
wrong_weights = attention_weights.masked_fill(~allowed, 0.0)
print("Incomplete method, row sums:", wrong_weights.sum(dim=-1))
print("Correct method, row sums:", causal_weights.sum(dim=-1))
```

<!-- cell -->
The first two row sums in the incomplete method are less than one. This does not mean every post-softmax mask is wrong; the specific problem is removing weights without renormalizing. The project code applies the mask before softmax.

## 6. Build query, key, and value vectors from the input

So far we wrote `q`, `k`, and `v` by hand. In a real model, each token input is a vector. We call the table of these vectors `x`. To create queries, we multiply the input by the weight matrix `Wq`; we do the same with two independent matrices, `Wk` and `Wv`.

This transformation is called a **linear projection**. Here, “linear” means that the output is built by adding products of input numbers and weights. It does not refer to the algorithm’s runtime being linear.

The weight matrices change during training. Attention weights also change, but they are **recomputed for every input**; the attention matrix is not a fixed table stored in the model.

**Code goal:** Create two-feature queries, keys, and values from a four-feature input.

<!-- cell -->

```python
# We have three positions; each position has four features.
x = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0]])

# These simple weights make it possible to check the products by hand.
Wq = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
Wk = Wq.clone()
Wv = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.0, 20.0]])

# Three independent projections use the same input.
projected_q = x @ Wq
projected_k = x @ Wk
projected_v = x @ Wv
print("Projected queries:\n", projected_q)
print("Projected values:\n", projected_v)
torch.testing.assert_close(projected_q, q)
torch.testing.assert_close(projected_k, k)
torch.testing.assert_close(projected_v, v)
```

<!-- cell -->
The queries and keys are the same tables as before. The values are also `[10, 0]`, `[0, 20]`, and `[10, 20]`. A real model starts with random values and training corrects them.

### Add a batch axis

Lesson 00 introduced three-dimensional shapes as tables placed together. Attention inputs use shape `(B, T, C)`: `B` is batch size, `T` is positions per segment, and `C` is features per token.

For now, `B=1`, `T=3`, and `C=4`. `unsqueeze(0)` adds an axis at the beginning without changing the values. We use this three-dimensional shape to match the project code.

**Code goal:** Inspect the input shape and the three projections after adding the batch axis.

<!-- cell -->

```python
# Put the 3-by-4 table into a batch containing one item.
batch_x = x.unsqueeze(0)
batch_q = batch_x @ Wq
batch_k = batch_x @ Wk
batch_v = batch_x @ Wv
print("Input (B, T, C):", tuple(batch_x.shape))
print("Query (B, T, d):", tuple(batch_q.shape))

# Swap the final two axes; keep the batch axis first.
batch_scores = batch_q @ batch_k.transpose(-2, -1)
print("Scores (B, query T, key T):", tuple(batch_scores.shape))
```

<!-- cell -->
The input, query, and score shapes are `(1, 3, 4)`, `(1, 3, 2)`, and `(1, 3, 3)`. For a three-dimensional tensor, `transpose(-2, -1)` swaps only the last two axes; `.T` does not express that intent.

## 7. Write a reusable attention function

We now put the steps in a Python function. It takes the input and three weight matrices and returns the attention output and weight table. The weights are useful for teaching; the project code returns only the output.

The function builds a mask for the input length instead of assuming length three. PyTorch performs the batched matrix multiplication independently for each batch item.

**Code goal:** Keep every step clearly defined in one function.

<!-- cell -->

```python
def one_head_attention(hidden, query_weight, key_weight, value_weight, causal=True):
    # Each three-dimensional input represents a batch of text segments.
    queries = hidden @ query_weight
    keys = hidden @ key_weight
    values_for_mix = hidden @ value_weight

    # Compare each query with the keys at every position.
    dimension = keys.shape[-1]
    scores = queries @ keys.transpose(-2, -1)
    scores = scores / math.sqrt(dimension)

    if causal:
        # Allowed cells contain the current position and its past.
        time = hidden.shape[1]
        allowed_positions = torch.tril(torch.ones(time, time, device=hidden.device)).bool()
        scores = scores.masked_fill(~allowed_positions, float("-inf"))

    # Normalize over sources, then combine the value vectors.
    weights = F.softmax(scores, dim=-1)
    mixed_values = weights @ values_for_mix
    return mixed_values, weights
```

<!-- cell -->
This cell only defines the function. Attention is not computed until it is called. `causal=True` is the default; `False` removes the future restriction.

**Code goal:** Run the function and compare it with the step-by-step calculation.

<!-- cell -->

```python
# Store the function’s two outputs under separate names.
manual_output, manual_weights = one_head_attention(batch_x, Wq, Wk, Wv)
print("Function output:\n", manual_output[0])

# Index zero selects the only batch item.
torch.testing.assert_close(manual_output[0], causal_output)
torch.testing.assert_close(manual_weights[0], causal_weights)
print("Manual stages and function agree.")
```

<!-- cell -->
The final message means both methods agree. The function’s only visible difference is that it keeps the batch axis. This check connects each code line to its numeric result.

## 8. Did we really prevent future leakage?

We have a testable property: changing only the last token must not change earlier outputs under causal attention. Earlier positions cannot read the last token. In software engineering, this is a behavioral contract.

We copy the input and change only its last row. `clone` creates an independent tensor, so the original remains unchanged.

**Code goal:** Observe causal versus unmasked attention by changing the future.

<!-- cell -->

```python
# Change only the last position; the earlier positions remain untouched.
changed_x = batch_x.clone()
changed_x[:, 2, :] = changed_x[:, 2, :] + 20.0

old_causal, _ = one_head_attention(batch_x, Wq, Wk, Wv, causal=True)
new_causal, _ = one_head_attention(changed_x, Wq, Wk, Wv, causal=True)
old_open, _ = one_head_attention(batch_x, Wq, Wk, Wv, causal=False)
new_open, _ = one_head_attention(changed_x, Wq, Wk, Wv, causal=False)

# Compare only the first two outputs.
causal_difference = (old_causal[:, :2] - new_causal[:, :2]).abs().max().item()
open_difference = (old_open[:, :2] - new_open[:, :2]).abs().max().item()
print("Causal difference:", causal_difference)
print("Without mask difference:", open_difference)
assert causal_difference < 1e-6
assert open_difference > 1e-3
```

<!-- cell -->
The causal versions differ by zero or a value very close to zero. The unmasked versions differ substantially. `abs` takes absolute values and `max` finds the largest difference. Therefore, this check detects even a single changed feature in an earlier position.

During training, the desired next token computes the loss. It must not enter the prediction path as visible input at the same position. Using a target for loss differs from letting attention see it; Lesson 05 demonstrates the separation.

### Compare with PyTorch’s built-in function

The `scaled_dot_product_attention` function performs the same score calculation, scaling, masking, and value combination. Its inputs are `q`, `k`, and `v` themselves; it does not create the linear projections for you. Setting `is_causal=True` enables the causal mask.

In this comparison, `dropout_p=0.0` prevents random weight removal so the results can be compared directly. We will explain dropout while building a block in the next lesson.

**Code goal:** Check the teaching calculation against the library implementation.

<!-- cell -->

```python
reference_output = F.scaled_dot_product_attention(
    batch_x @ Wq,                 # query vectors
    batch_x @ Wk,                 # key vectors
    batch_x @ Wv,                 # value vectors
    is_causal=True,               # future access is forbidden.
    dropout_p=0.0,                # there is no random removal in this experiment.
)
torch.testing.assert_close(manual_output, reference_output, rtol=1e-4, atol=1e-5)
print("PyTorch reference agrees:", tuple(reference_output.shape))
```

<!-- cell -->
The output shape is `(1, 3, 2)`, and the comparison should pass. The `rtol` and `atol` options specify the relative and absolute error tolerances; different floating-point calculation paths can produce very small differences. The function contract is described in the [official PyTorch documentation](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html).

## 9. Why divide scores by the square root of the dimension?

Suppose query and key components are initially nearly independent, have mean zero, and have similar spread. As dimension grows, the dot product sums more terms, so score spread usually grows.

If scores are very large and far apart, softmax gives nearly all weight to one source. This extreme concentration is **saturation**. Dividing by `sqrt(d_k)` prevents the score scale from growing unnecessarily with key dimension at the start of training. This explanation is approximate; trained vectors are not necessarily independent.

To see the effect on weights, we first pass the same synthetic scores to softmax with different multipliers. This simple experiment is separate from the statistical discussion.

**Code goal:** See how increasing score gaps concentrates the weights.

<!-- cell -->

```python
# # The score gaps are small, so all three sources receive some weight.
base_scores = torch.tensor([0.0, 1.0, 2.0])
for multiplier in [1.0, 5.0, 20.0]:
    weights_probe = F.softmax(base_scores * multiplier, dim=-1)
    print("Multiplier:", multiplier, "weights:", weights_probe)
```

<!-- cell -->
With multiplier 20, almost all weight goes to the third source. Concentration is not always wrong; it may be useful after training. The problem is letting dimension alone cause it before learning.

### A small statistical experiment

**Standard deviation** summarizes numeric spread: it is usually larger when values are farther from their mean. `std` computes it. Instead of one example, we create hundreds of random vector pairs for each dimension.

**Code goal:** Compare raw and scaled score spread across several dimensions.

<!-- cell -->

```python
# A local generator keeps this example independent of other random state.
probe_rng = torch.Generator().manual_seed(11)
for dimension in [4, 16, 64, 256]:
    q_probe = torch.randn(512, dimension, generator=probe_rng)
    k_probe = torch.randn(512, dimension, generator=probe_rng)

    # Compute a dot product for each row with only the key from that row.
    raw_probe = (q_probe * k_probe).sum(dim=-1)
    scaled_probe = raw_probe / math.sqrt(dimension)
    print("Dimension:", dimension,
          "raw std:", round(raw_probe.std().item(), 2),
          "scaled std:", round(scaled_probe.std().item(), 2))
```

<!-- cell -->
The raw standard deviation grows with dimension, while the scaled value usually stays near one. Because the number of samples is limited, we should not expect exactly one. This division does not fix poor data or a poor learning rate; its specific job is to control the scale of attention scores.

### Extending the discussion: weight and gradient spread

We can use **entropy** to describe how spread out the weights are. For now, remember that weight concentrated on one option has low entropy, while weight distributed across more options has higher entropy. Its formula is `-sum(p*log(p))`.

In Lesson 00, the gradient measured the error sensitivity of one adjustable number. In the next experiment, we compute the error with respect to the scores themselves, so the gradient shows how sensitive this example is to score changes. This experiment does not measure the behavior of every parameter in a complete network.

**Code goal:** Observe the effect of scaling on entropy and gradients in a small problem.

<!-- cell -->

```python
# # Each row has eight options; the synthetic target is option zero in every row.
q_grad = torch.randn(32, 64, generator=probe_rng)
k_grad = torch.randn(8, 64, generator=probe_rng)
large_scores = q_grad @ k_grad.T
targets_probe = torch.zeros(32, dtype=torch.long)

for label, divisor in [("raw", 1.0), ("scaled", math.sqrt(64))]:
    # # These scores are independent, and we want gradients for the scores themselves.
    scores_probe = (large_scores / divisor).detach().requires_grad_(True)
    probabilities_probe = F.softmax(scores_probe, dim=-1)

    # clamp_min prevents log(0) in this report.
    safe_log = probabilities_probe.clamp_min(1e-12).log()
    entropy_probe = -(probabilities_probe * safe_log).sum(-1).mean()

    # # The Lesson 01 loss function compares the scores with the desired answer.
    loss_probe = F.cross_entropy(scores_probe, targets_probe)
    loss_probe.backward()
    print(label, "entropy:", round(entropy_probe.item(), 3),
          "gradient norm:", round(scores_probe.grad.norm().item(), 4))
```

<!-- cell -->
The scaled version usually has higher entropy. The `norm` value gives one measure of the overall gradient size. Do not conclude from this column alone that the entire network is stable or that every weight has a better gradient. The purpose of this section is to separate a clear measurement from a broad claim.

## 10. A real attention head from the project code

An **attention head** follows this same query, key, value, and combination path. In `src/tinyllm/model.py`, the `Head` class implements it. A class in Python defines a type of object; each object of this class stores its own weights and settings.

In PyTorch, model classes inherit from `nn.Module`. This base registers trainable weights and lets you run a layer by writing `layer(input)`. The `forward` method defines how inputs become outputs. You will follow the class construction line by line in Lesson 04; here the goal is to connect the numeric calculation to real executable code.

The `Config` class stores model settings. In this example, the representation width is four and the number of heads is two, so each head has two output features. We set the other values only to construct the object; we do not train a complete model.

**Code goal:** Build a head with the same hand-written weights so the two methods can be checked for equivalence.

<!-- cell -->

```python
# # These are the actual classes from this project, not an external model package.
from tinyllm.model import Config, Head, Block

# # Width four divided across two heads gives each head dimension two.
head_config = Config(vocab=5, context=8, width=4, heads=2, layers=1, dropout=0.0)
real_head = Head(head_config)

# # nn.Linear stores weights as (output, input), so a transpose is required.
# # We write the weights directly here and do not need to track gradients for this change.
with torch.no_grad():
    real_head.query.weight.copy_(Wq.T)
    real_head.key.weight.copy_(Wk.T)
    real_head.value.weight.copy_(Wv.T)

# # Evaluation mode disables training-specific behavior.
real_head.eval()
actual_output = real_head(batch_x)
print("Real Head output:\n", actual_output.detach()[0])
torch.testing.assert_close(actual_output, manual_output)
print("Project Head agrees with the step-by-step calculation.")
```

<!-- cell -->
The real head output matches the teaching function output. `detach` only removes the tensor from gradient tracking for display. We did not use it in the actual attention calculation; during training, the model must compute gradients for the query, key, and value weights.

The real class creates the mask once at the `context` size and takes the required slice on each run. PyTorch registers this fixed table as a **buffer**: it is saved and moved with the model, but it is not a learnable weight.

## 11. Why use multiple attention heads?

One head creates one set of attention weights for each position. Multiple heads create independent mappings and independent combinations. This lets the model extract different relationships from the same input. We do not decide in advance that “head one” will learn grammar or “head two” will learn meaning; such labels are not guaranteed.

In this course model, the total representation width `C` is divided among `H` heads, so each head has dimension `C/H`. We place the head outputs side by side; this is called **concatenation**. A linear layer then combines the joined outputs.

First create only two heads and inspect their shapes. The first head uses the same hand-written weights; the second starts with independent initial weights.

**Code goal:** See how two two-feature outputs form a four-feature representation.

<!-- cell -->

```python
# # The second head has independent weights; it has not been trained in this teaching cell.
second_head = Head(head_config)
second_head.eval()

first_result = real_head(batch_x)
second_result = second_head(batch_x)

# # The last axis contains the features; sequence length and batch size stay unchanged.
joined_result = torch.cat([first_result, second_result], dim=-1)
print("First head:", tuple(first_result.shape))
print("Second head:", tuple(second_result.shape))
print("Joined heads:", tuple(joined_result.shape))
```

<!-- cell -->
Each head has shape `(1, 3, 2)` and the concatenated output has shape `(1, 3, 4)`. The number of heads is not the number of tokens; here we have three tokens and two heads. In the main course configuration, width is 64 and there are four heads, so each head has dimension 16.

**Code goal:** Apply the linear projection after concatenating the heads.

<!-- cell -->

```python
# Input and output both have four features; the weights combine them.
output_projection = nn.Linear(4, 4)
projected_heads = output_projection(joined_result)
print("After output projection:", tuple(projected_heads.shape))
print("First token features:", projected_heads.detach()[0, 0])
```

<!-- cell -->
The output shape remains `(1, 3, 4)`, but the feature values change. In the project code, this layer is named `project`, and its weights are learned during training. A fixed shape does not mean fixed information.

## 12. How does the model know token order?

If a token receives only its content embedding, the same word at two positions has the same initial vector. Unmasked attention alone does not add a position number. If all input rows are permuted, the outputs follow the same permutation.

We test this property with an example. We deliberately disable the mask to separate content effects from the mask’s ordering rule. The permutation `[2, 0, 1]` puts row three first, then row one, then row two.

**Code goal:** Show that attention without positional information does not impose its own order on input content.

<!-- cell -->

```python
permutation = torch.tensor([2, 0, 1])

# Disable the mask in both runs so we test content dependence only.
plain_output, _ = one_head_attention(batch_x, Wq, Wk, Wv, causal=False)
permuted_input = batch_x[:, permutation, :]
permuted_output, _ = one_head_attention(permuted_input, Wq, Wk, Wv, causal=False)

# The new output should be the old output with the same permutation.
torch.testing.assert_close(permuted_output, plain_output[:, permutation, :])
print("Content-only attention follows the input permutation.")
```

<!-- cell -->
The comparison should pass. This property is called **permutation equivariance**. For now, the numeric explanation is enough. The claim applies to attention without a mask and without positions. A causal mask has an ordering constraint, and this course’s model also receives explicit positional information.

### Add a position vector to the token vector

In the small GPT, one embedding table stores tokens and a separate table stores position numbers. Position vector zero is added to the first token, position vector one to the second, and so on. This is component-wise addition; it does not double the feature count.

For clarity, we create the position table by hand. In the real model, the same table comes from an `nn.Embedding` layer with trainable weights.

**Code goal:** See the effect of adding positions to a three-token input.

<!-- cell -->

```python
# Each row represents a position number and has four features here.
position_vectors = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                 [0.1, 0.2, 0.3, 0.4],
                                 [0.2, 0.4, 0.6, 0.8]])
positioned_input = batch_x + position_vectors.unsqueeze(0)
print("Input plus positions:\n", positioned_input[0])

# Permute the content but assign new position numbers starting at zero.
reordered_with_positions = batch_x[:, permutation, :] + position_vectors.unsqueeze(0)
first_positioned, _ = one_head_attention(positioned_input, Wq, Wk, Wv, causal=False)
second_positioned, _ = one_head_attention(reordered_with_positions, Wq, Wk, Wv, causal=False)
difference = (second_positioned - first_positioned[:, permutation, :]).abs().max().item()
print("Difference after assigning new positions:", difference)
assert difference > 1e-3
```

<!-- cell -->
This time the output is not merely a permutation of the previous output because each content vector receives a new position number. The model can use this information during training; adding a random table alone does not mean the model already understands order or language.

This model uses **learned absolute positions**: it has a table for positions from zero to the context limit. Other methods exist. For example, `RoPE` rotates query and key vectors by position so their product depends on positional distance. RoPE is outside this introductory course and is not implemented in this project.

## 13. Attention is not the only part of a Transformer block

A **Transformer** stacks blocks containing attention and other transformations. A block in this course takes each token representation and returns a new representation with the same shape. Three other components matter: layer normalization, a feed-forward network, and a residual connection.

### What does layer normalization do?

Suppose a token’s features are `[1, 3]`. Their mean is two. Subtracting the mean gives `[-1, 1]`. We then divide by a value related to spread to control the feature scale.

**Layer normalization (LayerNorm)** performs this operation separately for each token’s features. It does not average across sentences or positions. It usually has two trainable parameter sets: one scales features and the other shifts them. Initially, the scale is one and the shift is zero.

**Code goal:** Compare a hand calculation for one token with the built-in layer.

<!-- cell -->

```python
feature_vector = torch.tensor([[1.0, 3.0]])
feature_mean = feature_vector.mean(dim=-1, keepdim=True)

# The mean squared distance from the mean is the variance of these two features.
feature_variance = ((feature_vector - feature_mean) ** 2).mean(dim=-1, keepdim=True)
epsilon = 1e-5  # A small number prevents division by zero for identical features.
normalized_by_hand = (feature_vector - feature_mean) / torch.sqrt(feature_variance + epsilon)

# LayerNorm operates on the two features of the last axis.
normalizer = nn.LayerNorm(2, eps=epsilon)
normalized_by_layer = normalizer(feature_vector)
print("Mean:", feature_mean)
print("Variance:", feature_variance)
print("Normalized:", normalized_by_layer.detach())
torch.testing.assert_close(normalized_by_hand, normalized_by_layer)
```

<!-- cell -->
The mean is two, the variance is one, and the output is approximately `[-1, 1]`. `epsilon` makes the final values differ slightly from negative one and one. In the model, this layer works on axis `C` of `(B, T, C)`, normalizing features independently for every batch item and position.

### What does the feed-forward network add to attention?

Attention combines information across positions. The **feed-forward network**, named `MLP` in code, transforms each position independently. In this course, one linear layer expands width from `C` to `4*C`, `GELU` is applied, and a second linear layer returns the width to `C`.

An **activation** is a numeric transformation between layers. If you stack linear projections without a nonlinear transformation, the result is still a linear projection. GELU, like many other activations, removes this limitation. You do not need to memorize its exact derivative for this course.

**Code goal:** Inspect the shape changes through the two feed-forward layers.

<!-- cell -->

```python
# This example has four input features; the middle layer creates sixteen features.
expand_features = nn.Linear(4, 16)
shrink_features = nn.Linear(16, 4)

# Each operation is applied independently to each position’s features.
expanded = expand_features(batch_x)
activated = F.gelu(expanded)
feedforward_output = shrink_features(activated)
print("Input:", tuple(batch_x.shape))
print("Expanded:", tuple(expanded.shape))
print("After GELU:", tuple(activated.shape))
print("Back to width 4:", tuple(feedforward_output.shape))
```

<!-- cell -->
The shapes are `(1, 3, 4)`, `(1, 3, 16)`, `(1, 3, 16)`, and `(1, 3, 4)`. The activation changes values, not shapes. The network weights are shared across positions, but one position’s calculation cannot directly access other positions’ content.

### A residual connection adds the input to the result

With a **residual connection**, we add a transformation’s output to its input: `new_x = x + transformation(x)`. This creates a direct path for representations and gradients. The transformation can also make a small change to the previous representation when needed.

This addition works directly only when the tensor shapes are compatible. This is why the feed-forward network returns to width `C` and the head output is projected back to `C`: both must be added to the previous representation.

**Code goal:** See that a zero transformation preserves the previous representation.

<!-- cell -->

```python
# If the transformation proposes no change, the residual path preserves the input.
zero_change = torch.zeros_like(batch_x)
unchanged = batch_x + zero_change
torch.testing.assert_close(unchanged, batch_x)

# Normally, a non-zero change is added to the previous representation.
updated = batch_x + feedforward_output
print("Residual output shape:", tuple(updated.shape))
print("Zero change preserves input:", torch.equal(unchanged, batch_x))
```

<!-- cell -->
The second check returns `True`. This example only shows what the addition means; it does not claim that residual connections alone solve every deep-learning problem.

## 14. Run a real block step by step

This course block uses **pre-norm**: normalization happens before attention and before the feed-forward network. The order is:

1. The input features are adjusted with layer normalization.
2. The attention heads combine information from positions.
3. The head outputs are joined, passed through a linear mapping, and added to the input.
4. The result is normalized again and passed through the feed-forward network.
5. The feed-forward output is also added to the representation from the previous step.

This same order appears in the `forward` method of `Block`. We first run it with clear intermediate names and then compare it with calling the block directly. Dropout is set to zero, so this example has no random removal.

**Code goal:** Open the attention half of a real block and inspect the result shape.

<!-- cell -->

```python
real_block = Block(head_config)
real_block.eval()

# # Each position is normalized independently along the feature axis.
normalized_before_attention = real_block.norm1(batch_x)

# # Each head operates on the same normalized input with independent weights.
head_results = []
for head in real_block.heads:
    head_results.append(head(normalized_before_attention))

# # Join the outputs and return to the original width.
joined_block_heads = torch.cat(head_results, dim=-1)
attention_change = real_block.project(joined_block_heads)
attention_change = real_block.attention_dropout(attention_change)
after_attention = batch_x + attention_change
print("After attention and residual:", tuple(after_attention.shape))
```

<!-- cell -->
The shape is still `(1, 3, 4)`. `attention_dropout` is a layer that may randomly remove components in training mode. In this configuration, the removal probability is zero. Lesson 04 explains the difference between training and evaluation modes with a separate example.

**Code goal:** Run the feed-forward half and compare the full staged result with the block itself.

<!-- cell -->

```python
# # The second half of the block receives the representation built by the first half.
normalized_before_mlp = real_block.norm2(after_attention)
mlp_change = real_block.mlp(normalized_before_mlp)
block_by_stages = after_attention + mlp_change

# # Calling the object runs all of these steps in its forward method.
block_direct = real_block(batch_x)
torch.testing.assert_close(block_by_stages, block_direct)
print("Whole block:", tuple(block_direct.shape))
print("Stage-by-stage and direct Block agree.")
```

<!-- cell -->
The comparison should pass. The block weights are still untrained, so its output has no language quality by itself. In the next lesson, you will stack several blocks, add token and position embeddings, and convert the final representation into vocabulary scores. Those pieces will form the complete model.

## 15. Why does text generation repeat computations?

During text generation, a starting text or **prompt** is available first. The model creates one new token, the input becomes longer, and the model creates the next token. The simple method processes the entire input again at every step. This project’s `generate` method uses that simple approach.

With causal attention, adding a new token should not change the representations of earlier positions as long as their inputs and position numbers stay fixed. Therefore, some intermediate results can be stored and reused.

A **key-value cache (KV cache)** stores the key and value tables already computed for the past. For a new token, we create only that token’s query, key, and value. The new query scores the stored keys and combines the stored values. The model weights do not change.

We first examine this idea with the same single head and fixed inputs. This is a teaching implementation; it does not add caching to the main course model.

**Code goal:** Store the keys and values for the first two positions.

<!-- cell -->

```python
# # The first two positions act as the initial prompt.
prefix_x = batch_x[:, :2, :]
cache_k = prefix_x @ Wk
cache_v = prefix_x @ Wv
prefix_output, _ = one_head_attention(prefix_x, Wq, Wk, Wv)
print("Stored keys:", tuple(cache_k.shape))
print("Stored values:", tuple(cache_v.shape))
print("Initial prompt output:", tuple(prefix_output.shape))
```

<!-- cell -->
The stored tables have shape `(1, 2, 2)`: one batch, two previous positions, and two features. The initial processing of the full prompt is usually called **prefill**. Processing new tokens afterward is called **decode**.

### Process only one new token

The new token is the third position in the same input. We append its key and value to the previous tables. This step has only one query, so the score table has shape `(1, 1, 3)`. For this query, every available source is in the past or is the query itself; there is no future entry to mask.

**Code goal:** Compare the new-token output from the cache with recomputing the full input.

<!-- cell -->

```python
# # Take one new position; its input shape is (1, 1, 4).
new_x = batch_x[:, 2:3, :]
new_q = new_x @ Wq
new_k = new_x @ Wk
new_v = new_x @ Wv

# # Axis 1 is the time axis; the past and the new token are joined on it.
cache_k = torch.cat([cache_k, new_k], dim=1)
cache_v = torch.cat([cache_v, new_v], dim=1)

cached_scores = new_q @ cache_k.transpose(-2, -1) / math.sqrt(2)
cached_weights = F.softmax(cached_scores, dim=-1)
cached_last_output = cached_weights @ cache_v

full_output, _ = one_head_attention(batch_x, Wq, Wk, Wv)
torch.testing.assert_close(cached_last_output, full_output[:, -1:, :])
print("Cached last output:", cached_last_output)
print("Full recomputation last output:", full_output[:, -1:, :])
```

<!-- cell -->
The two outputs are equal. For the final output, we did not rebuild the keys and values for the two previous positions. In a multi-layer model, a cache is required for every layer and head; each layer’s input representation also depends on the computations in earlier layers.

If several new tokens are supplied at once, the first new token must not see the second new token. In that case, the mask must place the positions correctly relative to the cached past length. The statement “a cache never needs a mask” is true only for the single-token step above.

### Check all steps with a loop

For extra confidence, we create a six-position input and compare the cached output at each step with the last output from processing the full prefix. A **prefix** is the part of the text from its beginning through the current position.

**Code goal:** Check that the two methods agree at several sequence lengths.

<!-- cell -->

```python
cache_rng = torch.Generator().manual_seed(19)
cache_input = torch.randn(1, 6, 4, generator=cache_rng)
stored_k = None
stored_v = None

for position in range(cache_input.shape[1]):
    current_x = cache_input[:, position:position + 1, :]
    current_q = current_x @ Wq
    current_k = current_x @ Wk
    current_v = current_x @ Wv

    if stored_k is None:
        stored_k = current_k
        stored_v = current_v
    else:
        stored_k = torch.cat([stored_k, current_k], dim=1)
        stored_v = torch.cat([stored_v, current_v], dim=1)

    step_scores = current_q @ stored_k.transpose(-2, -1) / math.sqrt(2)
    step_output = F.softmax(step_scores, dim=-1) @ stored_v

    # # The reference method processes the full prefix again at every step.
    whole_prefix = cache_input[:, :position + 1, :]
    whole_output, _ = one_head_attention(whole_prefix, Wq, Wk, Wv)
    torch.testing.assert_close(step_output, whole_output[:, -1:, :], rtol=1e-4, atol=1e-5)
    print("Matching prefix length:", position + 1)
```

<!-- cell -->
You should see a success message for lengths one through six. This experiment demonstrates numeric equality for one head with fixed input; it is not a speed measurement for a real service. `torch.cat` allocates memory in this simple implementation; production services usually manage cache memory more carefully.

**Important limitation for this small GPT:** The position table is absolute, and the generation function keeps the last 64 tokens and renumbers their positions from zero. When the window moves forward, the position numbers of the remaining tokens change. In that situation, simply dropping the first rows of an old cache is not necessarily equivalent to recomputation. Adding a cache requires a clear decision about positions and the window; this example does not change the model in that way.

## 16. Why does text length increase attention cost?

For each head and text segment of length `T`, the score table has `T*T` entries. If the length grows from 64 to 128, the number of entries grows from 4,096 to 16,384, which is four times larger. This relationship is called **quadratic growth**.

The score-table size is also multiplied by the number of batches and heads. The next table shows only one head and one batch item. Each `float32` uses four bytes of memory. One `MiB` equals `2**20` bytes.

**Code goal:** Observe how the score table grows as sequence length increases.

<!-- cell -->

```python
for length in [64, 128, 256, 512]:
    score_cells = length * length
    score_bytes = score_cells * 4  # # Each float32 uses four bytes.
    print("Length:", length,
          "score cells per head:", score_cells,
          "score matrix MiB:", round(score_bytes / 2**20, 4))
```

<!-- cell -->
Each time the length doubles, the number of entries quadruples. This table counts only the memory for the score table stored by the simple method; it excludes the full model, activations, gradients, and optimizer state. The cost of the attention products also depends on head dimension; with fixed dimension, it grows as `T²`.

### How is cache memory calculated?

For each layer, both keys and values are stored, so there is a factor of two. For every batch item, position, and key/value head, we keep a vector of head dimension. Multiplying these counts by the bytes per number gives the cache-tensor memory.

**Code goal:** Calculate the hypothetical cache memory for this course’s small configuration.

<!-- cell -->

```python
# # These values are for the memory calculation; they do not add caching to the main model.
batch_size = 1
layers = 2
context_length = 64
kv_heads = 4
head_dimension = 16
bytes_per_value = 4

cache_bytes = (2 * batch_size * layers * context_length
               * kv_heads * head_dimension * bytes_per_value)
print("K and V cache bytes:", cache_bytes)
print("K and V cache KiB:", cache_bytes / 2**10)
```

<!-- cell -->
The result is 65,536 bytes, or 64 KiB. This counts only the numeric data in the two key/value tensors for the hypothetical architecture. Cache memory grows linearly with text length, while the score table for full-text processing grows with the square of the length.

### Names you may hear in interviews

In **standard multi-head attention (MHA)**, the numbers of query, key, and value heads are equal; our model uses this type. In **grouped-query attention (GQA)**, several query heads share key and value heads. In **multi-query attention (MQA)**, there is one key/value head shared by several query heads. Sharing can reduce cache memory, but it is an architectural change; changing one number in the memory formula does not convert the current model into GQA.

**Code goal:** Observe the effect of the number of key/value heads on hypothetical memory, separately from other settings.

<!-- cell -->

```python
for number_of_kv_heads in [4, 2, 1]:
    memory_bytes = (2 * batch_size * layers * context_length
                    * number_of_kv_heads * head_dimension * bytes_per_value)
    print("KV heads:", number_of_kv_heads, "cache KiB:", memory_bytes / 2**10)
```

<!-- cell -->
With all other values fixed, the results are 64, 32, and 16 KiB respectively. Query-head count does not appear in this memory calculation because we store only keys and values.

**FlashAttention** refers to an attention implementation that computes in small tiles and reduces memory traffic. It can run exact dense attention without keeping the full score table in main memory. It does not by itself make the number of pairs in dense attention linear.

In **local attention** and some forms of **sparse attention**, the set of positions allowed to interact is restricted. This changes the number of pairs examined and differs from running the same dense attention faster. Our teaching model uses dense causal attention.

## 17. How do Transformer families differ?

So far we have focused on continuing text, but transformers have other uses. The main difference in the table below is the rule governing which positions can access information.

| Family | Typical information access | Example use |
|---|---|---|
| Encoder | Bidirectional access to the input text, with any required masks | Text representations, classification, and search |
| Decoder | The current position and the past in self-attention | Text continuation |
| Encoder–decoder | The encoder reads the input; the decoder builds the output step by step and accesses the input representation | Translation and text-to-text transformation |

**Self-attention** means that query, key, and value come from the representation of the same sequence, as in this lesson. **Cross-attention** means that the query comes from one sequence while the key and value come from another; for example, a decoder attends to the encoder’s input representation.

The `TinyGPT` model belongs to the `Decoder-only` family. The original Transformer paper described an encoder–decoder architecture for translation; reading the paper is not required to run this course. The primary source is the [original Transformer paper](https://arxiv.org/abs/1706.03762).

### Why is training parallel while generation is sequential?

During training, the text and the desired answers for all positions are already present in the data. A causal mask lets us compute predictions for all positions at once without allowing any position to see its future answer.

During generation, the next token does not exist yet. One token is selected first, then it becomes part of the next step’s input. This dependence on previous outputs is called **autoregressive generation**. Parallel training computation does not mean that the usual method generates all response tokens simultaneously.

<!-- cell -->
## 18. Exercises to check your understanding

Exercise 1: In the opening example, change the third value vector from `[10, 20]` to `[100, 200]`. Predict before running: do the attention weights change? What about the attention output? In this exercise, `q` and `k` stay fixed.

Exercise 2: Disable the mask in `one_head_attention`, then run the future-change experiment again. Compare the earlier outputs with the causal case.

Exercise 3: If `B=2`, `T=5`, `C=12`, and `H=3`, what are the shapes of each head’s output and the concatenated output? Answer without coding, then check with real heads.

Exercise 4: Suppose the text length doubles while head count and head dimension stay fixed. By what factor does the score table grow? What happens to key/value cache memory?

<!-- fa-lint-ignore-next-line -->
<details dir="rtl"><summary>After answering, open the suggested answer</summary>

In Exercise 1, the weights stay fixed because they depend only on the query and key. The output changes because we combine different values with those same weights. In a real model, changing the input usually changes all three mappings; keeping query and key fixed is a condition of this exercise.

In Exercise 2, earlier positions may receive information from the future without a mask, so their outputs change when the future changes.

In Exercise 3, each head dimension is `12/3=4`. Each head output is `(2, 5, 4)`, and the concatenated output is `(2, 5, 12)`.

In Exercise 4, the score table becomes four times larger and cache memory doubles, assuming all other settings stay fixed.

</details>

**Code goal:** Check the shape answer from Exercise 3 with real layers.

<!-- cell -->

```python
exercise_config = Config(vocab=7, context=5, width=12, heads=3, layers=1, dropout=0.0)
exercise_input = torch.randn(2, 5, 12)
exercise_head_outputs = []
for head_index in range(exercise_config.heads):
    exercise_head = Head(exercise_config)
    exercise_head.eval()
    one_output = exercise_head(exercise_input)
    exercise_head_outputs.append(one_output)
    print("Head:", head_index, "shape:", tuple(one_output.shape))

exercise_joined = torch.cat(exercise_head_outputs, dim=-1)
print("Joined shape:", tuple(exercise_joined.shape))
assert exercise_joined.shape == (2, 5, 12)
```

<!-- cell -->
The shapes should match the exercise answer. This check does not require data or a trained model; it checks the architecture’s shape contract.

## 19. Interview questions

Before opening the answers, explain each answer using one of this lesson’s cells. The goal is to state the definition together with code evidence.

1. Are the `Wq` weights and the attention-weight table the same thing?
2. Why do we need three query, key, and value mappings?
3. Why must the mask act while the weights are being constructed?
4. How do the feed-forward network and attention differ in their access to positions?
5. What does the cache store, and what conditions are required to reuse it?
6. Does the number of heads determine the number of tokens or the window size?

<!-- fa-lint-ignore-next-line -->
<details dir="rtl"><summary>Suggested answers for comparison</summary>

The `Wq` weights are trainable parameters shared across different inputs. The attention-weight table is built from each input and usually differs across texts.

The query and key determine how much each source contributes. Values are the numeric content combined with those weights. Separate mappings let the model learn scoring and transferred content independently. This is an architectural choice; equal mappings were not forbidden in the hand-written example.

A mask before softmax sets future weights to zero and makes the allowed weights sum to one. Simply removing weights after softmax without renormalizing does not preserve that sum.

The feed-forward network transforms each position’s features independently. Attention combines position representations and creates relationships between them.

The cache stores intermediate keys and values for the layers. It does not change model weights and is not permanent account memory. Reuse is valid when the past computation, including inputs, positions, and model settings, remains valid. Evaluation mode also prevents random dropout.

The number of heads is the number of attention paths. Text length is represented by `T`, and this model’s input limit is `context`; neither is the number of heads.

</details>

## Terms you should now be able to connect to code

| Term | Meaning in this lesson | Executable evidence |
|---|---|---|
| Query, key, and value | Three numeric mappings of the input representation | The calculations `x @ Wq`, `x @ Wk`, and `x @ Wv` |
| Attention weight | Each source’s contribution to the information combined for a query | Softmax output over score columns |
| Causal mask | Preventing access to future positions | Negative infinity before softmax |
| Attention head | An independent path for scoring and combining values | The teaching function and `Head` class |
| Position | The token’s position number in the input | Adding the position vector to the token vector |
| Layer normalization | Adjusting each position’s features independently | The `nn.LayerNorm` class |
| Feed-forward network | Transforming one position’s features without reading other positions | Linear, GELU, and linear layers |
| Residual connection | Adding a proposed change to the previous representation | The expression `x + change` |
| Key/value cache | Keeping past computations for a new generation step | The `cache_k` and `cache_v` tables |

**Next step:** In [Lesson 04, Build a Tiny GPT](04_build_tiny_gpt.ipynb), you will build a complete language model from embeddings, blocks, and an output layer. This lesson explained what each block does; the next lesson shows how to connect these blocks to token inputs and next-token scores.
