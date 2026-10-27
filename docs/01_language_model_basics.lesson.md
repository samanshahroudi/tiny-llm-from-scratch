# 01. Recognize the language model by counting and generating text

A language model answers the question: "Given the text I've seen so far, what could come next?" In this lesson, you turn that question into a small program. The program counts how often letters appear next to one another, converts those counts into probabilities, and then generates text one letter at a time.

In Lesson 00, you worked with numbers, tensors, character IDs, and a one-parameter learning example. If tensor shapes or the `print` command are still unclear, return to that lesson before continuing. This lesson assumes no prior knowledge of probability, neural networks, or language models.

**Practical result of the lesson:** You build a **bigram** language model. A bigram model uses only the last text unit to make its next-token prediction. It has no neural network, but its inputs and outputs prepare you for the model you will build later.

Read the description of each section, guess the output, and run the code cell with `Shift+Enter`. Cell variables are interdependent; The execution order is from top to bottom. If you restarted the kernel, run the cells again in the same way. After the text generation section, you can stop and read the evaluation section in the next session.

<!-- cell -->
## 1. Prepare the environment

The `torch` library is the same PyTorch that you worked with in lesson 00. The `math` tool provides mathematical functions and is included with the Python installation. You will use it a little later to calculate the logarithm; The meaning of this calculation is explained before use.

Setting `set_num_threads(2)` allows some calculations to use two processor execution paths. For the small size of this lesson, increasing this number will not necessarily make the execution faster.

**Code Purpose:** Import the required tools and see the version of the main library.

<!-- cell -->
```python
# Explain this step.
from pathlib import Path

# Explain this step.
import sys

# Explain this step.
import math

# Explain this step.
import torch

# Explain this step.
torch.set_num_threads(2)

print("PyTorch version:", torch.__version__)
```
<!-- cell -->
### How do we find the data folder?

The kernel working folder may be `notebooks` or the project root folder. The code below checks the current folder and then higher folders. The existence of the `src/tinyllm` folder is a sign that the project is found. This is the same method as lesson 00.

The variable `ROOT` is simply the name we chose for the project path. In `Path`, the `/` operator joins path components; it does not perform mathematical division.

<!-- cell -->
```python
# Explain this step.
current_folder = Path.cwd()

# Explain this step.
candidate_folders = [current_folder]
candidate_folders.extend(current_folder.parents)

# Explain this step.
ROOT = None

for folder in candidate_folders:
    # Explain this step.
    if (folder / "src" / "tinyllm").is_dir():
        ROOT = folder
        break  # Explain this step.

if ROOT is None:
    raise FileNotFoundError("Open this notebook from the tiny-llm-fa folder.")

print("Project folder:", ROOT.name)
```
<!-- cell -->
The output should show the name `tiny-llm-fa`. To use the shared course codes, you need to add the `src` folder to the Python search path. The `sys.path` directory specifies where Python should look for code when running `import`. This only changes the configuration of this kernel and does not install any libraries.

<!-- cell -->
```python
# Explain this step.
source_path = str(ROOT / "src")

# Explain this step.
if source_path not in sys.path:
    sys.path.insert(0, source_path)

print("Project code available:", source_path in sys.path)
```
<!-- cell -->
The output `True` means that Python can find the codes in the folder `src`. You import the functions of this project after the small example; First, know the problem itself.

<!-- cell -->
## 2. What do model, token and probability mean?

**Model** in this lesson is a program that obtains a pattern from several examples and creates an output for a new input. If he has seen many times that the letter "A" comes after the letter "B", he may consider "A" more likely. The model makes a statistical law from this repetition; We don't program it with commands like "always write a after b".

A token is a unit of text that the model works with. In this period, each Unicode character is a token. For starters, you can think of it as a letter, space, or symbol. Unicode details come in lesson 02. In many other models, a token can be a word or part of a word; So "token" does not always mean "letter" or "word".

The text `hello` has five tokens in the model of this course. To predict the last position, the text `hell` is the input, and the letter `o` is the response we expect in the tutorial example. The English name of the desired answer is **target**. The text given to the model is called **context** or `context`.

**Objective of the example:** See the difference between the number of characters and the number of words in a text.

<!-- cell -->
```python
# Explain this step.
example_text = "hello world"

# Explain this step.
characters = list(example_text)

# Explain this step.
words = example_text.split()

print("characters:", characters)
print("character count:", len(characters))
print("words:", words)
print("word count:", len(words))
```
<!-- cell -->
The number of characters is 11 and the number of words is 2. The space between two words is also listed in the list of characters. For this course tokenizer, distance is part of the model input.

### Start probability with an everyday example

Suppose there are five cards in a bag: three cards with the inscription "A" and two cards with the inscription "B". If you take a card without looking, three out of five choices will be A's. The probability of "A" is equal to `3 / 5`, i.e. 0.6. The probability of "b" is also 0.4.

The probability is between zero and one. Zero means, in this calculation, that choice cannot happen; One means that all possibilities reach that choice. The set of probabilities for all possible choices is called **probability distribution**. In the example of the bag, the sum of the probability of "A" and "B" is equal to one.

**Code Purpose:** Calculate these two probabilities by dividing the number of each card by the total number.

<!-- cell -->
```python
# Explain this step.
count_a = 3
count_b = 2
total_count = count_a + count_b

# Explain this step.
probability_a = count_a / total_count
probability_b = count_b / total_count

print("P(a):", probability_a)
print("P(b):", probability_b)
print("sum:", probability_a + probability_b)
```
<!-- cell -->
The outputs are 0.6, 0.4 and 1. The language model also creates such a set of possibilities for the next token; But the choices are changed based on the previous text.

### Conditional probability: Which information do we already know?

Suppose in the sample text, after "b" there is three times "a" and one time "b". When we know that the preceding letter is "b", we only count these four observations. The probability of "A" coming after "B" is equal to `3 / 4`.

The name of this calculation is **conditional probability**: the probability of an event, taking into account the information we already have. Read the statement `P(a | b)` "Probability A, subject to B". The symbol `|` in this text means "provided".

For real text, the language model usually wants to find the probability of the next token given the previous text. The Bigram model limits the bet to the last token. For example, for the entries `The ` and `A `, the last token is both a space; Therefore, the bigram model uses the same probability distribution to continue both.

<!-- cell -->
## 3. Make the first model with a small text

First, work with the text `ababac`. The goal is to keep the example small enough that you can check the result by hand. Meaningful word or sentence is not necessary; To know the count, these six letters are enough.

There are three different letters in this text: `a`, `b` and `c`. We call the list of all known tokens **vocabulary** or `vocabulary`. This name is also used in the letter model, even if its elements are not words. The number of these elements is usually represented by the letter `V`.

**Code Purpose:** Create a numeric identifier for each letter and convert the text into a list of identifiers. ID is just the letter number in the table; A bigger number does not mean a more important letter.

<!-- cell -->
```python
toy_text = "ababac"
toy_vocabulary = ["a", "b", "c"]

# Explain this step.
toy_to_id = {}
for index, character in enumerate(toy_vocabulary):
    toy_to_id[character] = index

# Explain this step.
toy_ids = []
for character in toy_text:
    toy_ids.append(toy_to_id[character])

print("character to ID:", toy_to_id)
print("text IDs:", toy_ids)
```
<!-- cell -->
Output IDs are `[0, 1, 0, 1, 0, 2]`. The `enumerate` function gives both the position number and the value of that position in each round. Python numbering starts at zero.

### Adjacent pairs are training examples

**Bigram** or `bigram` means a pair of adjacent tokens. The six-character example contains five adjacent pairs. The first pair is `a → b`, and the last pair is `a → c`.

Each pair gives us an input token and the next token that should be predicted. We will use the same idea in several small examples, and Lesson 05 will apply this one-position shift to GPT training.

**Purpose of the code:** Show all pairs of text so that the next count can be checked.

<!-- cell -->
```python
# Explain this step.
for position in range(len(toy_ids) - 1):
    previous_id = toy_ids[position]
    following_id = toy_ids[position + 1]

    # Explain this step.
    previous_character = toy_vocabulary[previous_id]
    following_character = toy_vocabulary[following_id]
    print(previous_character, "->", following_character)
```
<!-- cell -->
The pair `a → b` appears twice, the pair `b → a` twice, and the pair `a → c` once. The sample text has no pairs with the previous letter `c`; Because `c` is at the end of the text.

### count table: row is input, column is response

Create a three-by-three table. Row zero represents `a`, and column one represents `b`. Therefore, cell `[0, 1]` records how many times `b` follows `a`.

The tensor of this table has two axes: the first axis for the previous letter and the second axis for the next letter. In lesson 00, you recognized the shape of the tensor; The table format for this example is `(3, 3)`.

<!-- cell -->
```python
toy_V = len(toy_vocabulary)

# Explain this step.
toy_counts = torch.zeros(toy_V, toy_V, dtype=torch.float32)

for position in range(len(toy_ids) - 1):
    previous_id = toy_ids[position]
    following_id = toy_ids[position + 1]

    # Explain this step.
    toy_counts[previous_id, following_id] += 1

print("row and column order:", toy_vocabulary)
print("count table:\n", toy_counts)
print("shape:", tuple(toy_counts.shape))
```
<!-- cell -->
The line `a` is equal to `[0, 2, 1]`, the line `b` is equal to `[2, 0, 0]`, and the line `c` is equal to `[0, 0, 0]`. Each row summarizes the responses seen for an entry.

The model has "learned" so far, meaning that the information from the examples has been accumulated in the table. Learning does not always mean using a neural network or derivative. This table is a statistical model and is trained by counting.

### Convert count to probability

There are three observations after `a`. `b`'s share of these observations is two out of three and `c`'s share is one out of three. Conditional probability is obtained by dividing each house of the row by the total of the same row.

**Code Purpose:** Just split the line `a`. Row `c` still has a sum of zero and we are not dividing it for now.

<!-- cell -->
```python
# Explain this step.
row_a = toy_counts[toy_to_id["a"]]

# Explain this step.
row_a_total = row_a.sum()
row_a_probabilities = row_a / row_a_total

print("counts after a:", row_a.tolist())
print("total observations:", row_a_total.item())
print("probabilities after a:", row_a_probabilities.tolist())
print("probability sum:", row_a_probabilities.sum().item())
```
<!-- cell -->
The probabilities are approximately `[0.0, 0.667, 0.333]` and their sum is one. The `tolist` function converts a tensor to a Python list. The `item` function extracts the same number from a single-number tensor.

These probabilities are calculated only from the six-letter text. If the data changes, the table and probabilities also change. The result of this example is not the rule of all texts.

<!-- cell -->
## 4. Why do we give a count to the unseen couples?

The raw table has two problems. First, the line `c` has no observations, and dividing it by the total of the line will be divided by zero. Second, the model assumes zero probability for `a → a`, just because it didn't see it in this short text. Not seeing a pair in low data is not enough to consider it impossible.

One simple solution is **smoothing** or `smoothing`. In this example, we add one unit to each cell of the table. This method is called "add one" or `add-one smoothing`. This is not a unit, an actual observation; It is an auxiliary assumption so that no probability becomes zero.

**Code Purpose:** Smooth the counts and see how the unseen rows change.

<!-- cell -->
```python
# Explain this step.
toy_smoothed_counts = toy_counts + 1.0

# Explain this step.
toy_probabilities = torch.zeros_like(toy_smoothed_counts)
for row_index in range(toy_V):
    row_counts = toy_smoothed_counts[row_index]
    row_total = row_counts.sum()
    toy_probabilities[row_index] = row_counts / row_total

print("smoothed counts:\n", toy_smoothed_counts)
print("probability table:\n", toy_probabilities)
```
<!-- cell -->
The line `a` is now `[1, 3, 2]` and the sum is 6. The probabilities of this line are about `[0.167, 0.5, 0.333]`. `c` line changed from `[0, 0, 0]` to `[1, 1, 1]`; So three letters have the same probability of getting one third.

Smoothing also changes the proportion of true observations. For example, the probability of `b` after `a` has increased from two-thirds to one-half. So this work is not free; Adds an assumption to the calculation. In larger data, the effect of the same extra count is usually smaller.

### Do the same calculation for all rows at the same time

In a two-dimensional table, `sum(dim=1)` adds the cells in each row. With `keepdim=True`, each row sum remains a column, so the shape is `(3, 1)` for this three-by-three table.

In dividing the table `(3, 3)` by the column `(3, 1)`, PyTorch uses the number of each row of the column for all three houses of the same row. This compatibility of shapes is called **autoplay** or `broadcasting`. The number of numbers in the column does not change; The program uses the same number in several houses when calculating.

<!-- cell -->
```python
# Explain this step.
row_totals = toy_smoothed_counts.sum(dim=1, keepdim=True)

# Explain this step.
toy_probabilities_fast = toy_smoothed_counts / row_totals

print("row totals:\n", row_totals)
print("row totals shape:", tuple(row_totals.shape))

# Explain this step.
same_result = torch.allclose(toy_probabilities, toy_probabilities_fast)
print("same as the row loop:", same_result)

# Explain this step.
assert same_result
```
<!-- cell -->
The output of the comparison is `True`. Both methods perform the same calculation; The second method is shorter and will be more suitable for larger tables. The `assert` command checks the condition and returns an error if it is false.

**Short Exercise:** Add the value `0.5` to the raw table instead of one and recalculate the row `a`. First, write by hand how much the total line will be.

<!-- fa-lint-ignore-next-line -->
<details dir="rtl"><summary>Answer and method of exercise review</summary>

The line `a` becomes `[0.5, 2.5, 1.5]` and the sum is 4.5. The probabilities are around 0.111, 0.556 and 0.333. By decreasing the amount of smoothing, the most frequent pair `a → b` again gets a larger share. To continue the lesson, run the original cells with a value of one.

</details>

<!-- cell -->
## 5. Go from the small example to the data of this project

**dataset** or `dataset` are the examples we use to learn and check the model. In this project, the file `data/documents.json` has some short Persian and English texts. This data is created artificially and with common formats; It's good for understanding the code, but don't consider its quality as a measure of ability on real texts.

The data has three parts. The **training** section or `train` is used to build the model. The **validation** section or `validation` is for checking the model during development. We keep the **Exam** or `test` section for final evaluation. Here you only count on training part and measure the result on training and validation.

The common code `load_data` reads this file, puts together the text of each section, creates a table of character IDs from the training text, and converts the texts into tensors of IDs. Vocabulary building from training prevents the validation section's letter list from being used to build the model. However, sharing text formats can make evaluation easier than actual use.

**Code Purpose:** Load data and see the input shape instead of immediately starting the tutorial.

<!-- cell -->
```python
# Explain this step.
from tinyllm.data import load_data

# Explain this step.
documents, texts, tokenizer, data = load_data(ROOT / "data")

print("document count:", len(documents))
print("vocabulary size:", tokenizer.vocab_size)

for split_name in ["train", "validation", "test"]:
    # Explain this step.
    token_tensor = data[split_name]
    print(split_name, "token count:", token_tensor.numel())

# Explain this step.
print("training text preview:", repr(texts["train"][:100]))
```
<!-- cell -->
The variable `documents` is the list of documents. The variable `texts` is a dictionary of the text of each section and the variable `data` is a dictionary of tensor identifiers. The `repr` tool shows the technical representation of the string; For example, you see going to the next line as `\n`.

The vocabulary size of this current data is 83. ID zero is reserved for unknown letter; It means a word that is not in the vocabulary made from education. In lesson 02, you will check its behavior separately.

### See token generator agreement before counting

The `encode` function converts text to identifiers, and the `decode` function converts identifiers back to text. Now you get an ID for each sample English character. These identifiers are not the same as the previous three-letter example numbers; Because the words and order are different.

<!-- cell -->
```python
sample_text = "hello"

# Explain this step.
sample_ids = tokenizer.encode(sample_text)

# Explain this step.
recovered_text = tokenizer.decode(sample_ids)

print("text:", sample_text)
print("IDs:", sample_ids)
print("decoded:", recovered_text)

assert recovered_text == sample_text
```
<!-- cell -->
The retrieved text is `hello`. In the following code, these identifiers will be the row and column addresses of the counting table. Reading the table as `counts[i, j]` means "a house with the first letter of the identifier `i` and the next letter of the identifier `j`".

**Objective of the code:** On the training text, make the bigram table by smoothing one. This version performs the same calculation as the three-letter example for complete words.

<!-- cell -->
```python
# Explain this step.
V = tokenizer.vocab_size

# Explain this step.
counts = torch.ones(V, V, dtype=torch.float32)
train = data["train"]

for position in range(train.numel() - 1):
    # Explain this step.
    previous_id = train[position].item()
    following_id = train[position + 1].item()
    counts[previous_id, following_id] += 1

# Explain this step.
count_totals = counts.sum(dim=1, keepdim=True)
probabilities = counts / count_totals

print("counts shape:", tuple(counts.shape))
print("probabilities shape:", tuple(probabilities.shape))

# Explain this step.
assert torch.allclose(probabilities.sum(dim=1), torch.ones(V))
```
<!-- cell -->
The form of each table is `(83, 83)`. Each line has an independent probability distribution for the next letter. The code counts all adjacent pairs of continuous training text; This includes spaces and blank lines between documents. So the model here does not learn the end of the document with a special token.

Because smoothing is added to all houses, the unknown ID also has a non-zero probability. If this ID is selected in production, the tokenizer will display it as `�`. This behavior is allowed for the current training model; In a functional model, we design the vocabulary and selection of special tokens more precisely.

### What does the table say about the letter `e`?

First, find the ID `e`. With that ID, remove the appropriate row of the table. The `topk(8)` function returns the eight largest values ​​and their column IDs. This is a read-only table; You are not generating text yet.

<!-- cell -->
```python
# Explain this step.
e_id = tokenizer.encode("e")[0]
distribution_after_e = probabilities[e_id]

# Explain this step.
top_values, top_ids = distribution_after_e.topk(8)

print("Most likely tokens after e:")
for rank in range(8):
    next_id = top_ids[rank].item()
    next_character = tokenizer.decode([next_id])
    next_probability = top_values[rank].item()

    # Explain this step.
    print(repr(next_character), round(next_probability, 4))
```
<!-- cell -->
Each share number is the same letter as the following `e`, according to the training and smoothing data. If a space or newline is among the options, the `repr` utility will reveal it. These eight possibilities do not necessarily add up to one; The rest of the 75 options also have a share.

**Short Exercise:** Change the letter `e` to `a` and run the cell. Did the order of the options change? Explain the reason by referring to the counting of pairs. Then use `e` again to continue.

<!-- cell -->
## 6. Generating text means repeating the selection of a token

So far you have the probability table. To generate the text, you need to select a letter from it. There are two simple methods:

- **greedy decoding** or `greedy`: always choose the option with the highest probability.
- **Sampling** or `sampling`: choose randomly, but so that more likely options are chosen more often.

Sampling is like drawing a card from the same bag. If the odds are `[0.75, 0.25]`, the first option is usually chosen more often, but not necessarily the winner every time. The function `torch.multinomial` makes this selection from probabilities.

**random seed** or `seed` is the starting value of the pseudo-random number generator. With the same seed, you can repeat the experiment in the same environment. This does not change the odds. Exact replication is not guaranteed on all versions and hardware.

**Objective of the example:** Sample a two-choice distribution several times and see how it differs from always choosing the first choice.

<!-- cell -->
```python
two_choices = torch.tensor([0.75, 0.25])

# Explain this step.
example_rng = torch.Generator().manual_seed(7)

# Explain this step.
draws = torch.multinomial(
    two_choices,
    num_samples=20,
    replacement=True,
    generator=example_rng,
)

print("sampled IDs:", draws.tolist())
print("number of zeros:", (draws == 0).sum().item())
print("number of ones:", (draws == 1).sum().item())

# Explain this step.
print("greedy ID:", two_choices.argmax().item())
```
<!-- cell -->
In sampling, the observed ratio is not necessarily exactly three to one; You have selected only 20 times. The probability of 0.75 is the rule of the distribution, not the commitment that the first choice will come up exactly 15 times out of every 20 choices. Greedy selection in this example always gives zero id.

### Perform a production step by hand

The starting text is called **prompt** or `prompt`. Here the prompt is just the opening text, not necessarily the command or question. For bigrams, specifies the last prompt ID of the table row.

**Code Purpose:** Add a letter to the text `The `. Four tasks are performed: converting text to ID, selecting rows, sampling the next ID, and converting the result to text.

<!-- cell -->
```python
prompt = "The "
generated_ids = tokenizer.encode(prompt)

# Explain this step.
last_id = generated_ids[-1]
next_distribution = probabilities[last_id]

one_step_rng = torch.Generator().manual_seed(7)
sampled_id_tensor = torch.multinomial(
    next_distribution, num_samples=1, generator=one_step_rng
)
next_id = sampled_id_tensor.item()

# Explain this step.
generated_ids.append(next_id)

print("prompt:", repr(prompt))
print("last prompt character:", repr(tokenizer.decode([last_id])))
print("selected next character:", repr(tokenizer.decode([next_id])))
print("after one step:", repr(tokenizer.decode(generated_ids)))
```
<!-- cell -->
The list of identifiers has become one element longer. If the selected token is a space or a newline, quick glance text change is not enabled; Therefore, we show the output as `repr`.

### Repeat the same step several times

The following logic function places the same cell in a loop. The number `count` specifies how many new tokens to add to the text. Using the function helps to check multiple prompts and seeds without repeating the entire code.

**Code Purpose:** Create the bigram generation function. Each round of the loop reads only the last ID; The model does not use the rest of the text in the probability calculation.

<!-- cell -->
```python
def bigram_generate(prompt, count=120, seed=7):
    # Explain this step.
    if not prompt:
        raise ValueError("The prompt must contain at least one character.")
    if count < 0:
        raise ValueError("count must be non-negative.")

    ids = tokenizer.encode(prompt)
    rng = torch.Generator().manual_seed(seed)

    for step in range(count):
        last_id = ids[-1]  # Explain this step.
        distribution = probabilities[last_id]

        # Explain this step.
        sampled = torch.multinomial(distribution, 1, generator=rng)
        next_id = sampled.item()
        ids.append(next_id)

    # Explain this step.
    return tokenizer.decode(ids)
```
<!-- cell -->
The command `def` defines a function; Until you call the function, its body is not executed. The values ​​after `=` in the definition line are the default values ​​of the entries. The name `step` is the circle number; This function does not need the number itself and only uses the loop to repeat.

**NEXT CELL OBJECTIVE:** See the actual output and experiment with two different seeds.

<!-- cell -->
```python
output_seed_7 = bigram_generate("The ", count=120, seed=7)
output_seed_8 = bigram_generate("The ", count=120, seed=8)

print("seed 7:\n", output_seed_7)
print("seed 8:\n", output_seed_8)

# Explain this step.
repeated_output = bigram_generate("The ", count=120, seed=7)
print("same seed repeated:", output_seed_7 == repeated_output)

assert output_seed_7 == repeated_output
```
<!-- cell -->
You may see word-like fragments, but don't expect a coherent sentence. The model has only learned adjacent pairs and does not know what the subject of the sentence is. Farsi and English text may also be combined in the output; The data contains both languages ​​and the bigram does not have enough information to hold the language of the sentence.

Changing the seed did not change the `probabilities` table. Only the token selection path has been changed. In a neural model, it is necessary to separate the change of the model from the change of the output selection method.

### Prove the context restriction with a simple check

The two prompts below end with the same letter. Even though they mean different things, the bigram model reads the same table row when it predicts the first new token.

<!-- cell -->
```python
prompt_a = "The "
prompt_b = "A "

last_id_a = tokenizer.encode(prompt_a)[-1]
last_id_b = tokenizer.encode(prompt_b)[-1]

distribution_a = probabilities[last_id_a]
distribution_b = probabilities[last_id_b]

print("same last ID:", last_id_a == last_id_b)
print("same next-token distribution:", torch.equal(distribution_a, distribution_b))
```
<!-- cell -->
Both outputs are `True`. This result does not depend on the random seed; It is related to the structure of the model. The transformer you see in Lesson 03 combines the information of several previous tokens to build the next probability. This is one of the reasons for its greater ability to maintain the connection of text parts.

<!-- cell -->
## 7. Know logit and softmax with three options

Counting bigrams generate probabilities directly from counts. The neural network usually first produces a "raw score" for each token. We call this score **Logit** or `logit`. Logit is not a possibility; It can be negative or greater than one, and the sum of points is not necessarily one.

Suppose that for the three options, scores of `[3, 1, -1]` are generated. The first score is higher, but it still cannot be called probability 3. The **softmax** function or `softmax` converts these scores into positive numbers with a sum of one.

Its method is as follows: first, the exponential value of `exp(score)` is calculated from each score; Then each value is divided by the sum of all of them. The exponential function converts negative and positive numbers into positive values, and higher scores get a larger value. For this lesson, this behavior is enough; You do not need to memorize the mathematical definition of the exponential function.

**Purpose of the code:** First see the calculation in three lines and then compare the result with PyTorch ready function.

<!-- cell -->
```python
# Explain this step.
scores = torch.tensor([3.0, 1.0, -1.0])

# Explain this step.
positive_values = torch.exp(scores)
positive_total = positive_values.sum()
manual_probabilities = positive_values / positive_total

# Explain this step.
softmax_probabilities = torch.softmax(scores, dim=-1)

print("raw scores:", scores.tolist())
print("positive values:", positive_values.tolist())
print("probabilities:", softmax_probabilities.tolist())
print("sum:", softmax_probabilities.sum().item())

assert torch.allclose(manual_probabilities, softmax_probabilities)
```
<!-- cell -->
The probabilities are approximately 0.867, 0.117 and 0.016. A score of 3 has not been converted into a probability of 3; All points contribute to determining the probability of each option.

For very large scores, the direct calculation of `exp` may go beyond the numerical range. In the model code, we use the prepared function `torch.softmax`, which performs the calculation in a more stable way. Manual calculation here is just for understanding the concept.

### Temperature changes the intensity of preferences

**Temperature** or `temperature` is a positive number that we divide the points by before softmaxing. A temperature less than one makes the difference in scores larger and the probability distribution more concentrated. A temperature greater than one makes the differences smaller and increases the odds of the lower-scoring options.

Temperature does not retrain the model; It only changes the output selection from the same scores. High temperature can increase variety, but does not guarantee quality or accuracy.

**Code Objective:** See what probabilities the same three points become with three temperatures.

<!-- cell -->
```python
for temperature in [0.5, 1.0, 2.0]:
    # Explain this step.
    adjusted_scores = scores / temperature
    temperature_probabilities = torch.softmax(adjusted_scores, dim=-1)

    # Explain this step.
    rounded_values = []
    for value in temperature_probabilities.tolist():
        rounded_values.append(round(value, 3))

    print("temperature:", temperature, "probabilities:", rounded_values)

print("greedy choice ID:", scores.argmax().item())
```
<!-- cell -->
For a temperature of 0.5, the probability of the first option is about 0.982. For temperature 1, it is about 0.867 and for temperature 2, it is about 0.665. The order of the options does not change in this example; The intensity of their preference changes.

Do not put zero temperature in this formula, because it is divided by zero. In many applications, a value of zero is a convention for greedy selection: `argmax` is used directly. In the model building lesson, you will see this contract in the production code.

The high probability of a token does not mean "percentage of the sentence being correct". For example, the familiar continuation of an incorrect sentence may have a high probability. In this step, the model measures the statistical compatibility with the text.

<!-- cell -->
## 8. How do we measure the error of the language model?

The training text already has the next correct token. If the input is `hell`, the target in the example `hello` is `o`. To evaluate, we ask what probability the model has given to this goal. The higher the probability, the better the model performed in that instance.

In lesson 00, the error was the distance between the numerical output and the numerical target. Here is the output of a probability distribution. We need a number that shows the high probability of the target with a low error and the low probability of the target with a high error. A common choice is the **negative of the log likelihood of the target**: `-log(p_target)`.

You don't need to know the logarithm beforehand. The `log` function here converts a probability to a number that is zero for a probability of one and becomes more negative as the probability approaches zero. A negative sign in front of it turns the result into a positive error. A probability of one means zero error; Very low probability means high error.

**Example Objective:** Convert only three hypothetical probabilities to errors to clarify benchmark behavior. This calculation uses the natural logarithm.

<!-- cell -->
```python
for target_probability in [1.0, 0.5, 0.1]:
    # Explain this step.
    example_loss = -math.log(target_probability)
    print("target probability:", target_probability, "loss:", round(example_loss, 4))
```
<!-- cell -->
The errors are about 0, 0.6931 and 2.3026 respectively. This error is not infinite if the target probability is exactly zero. The smoothing of the bigram table has removed the zero probabilities and allows this measure to be calculated for all pairs.

The short name of the error in the code is **loss**. When the target is only one correct token per slot, the cross-entropy error or `cross-entropy` also computes the same negative logarithm of the probability of that token. In lesson 05, you use this concept for GPT output.

### Read a real example from the table

First, remove the probability of the pair `h → e` from the table. Then calculate the error of the same pair. This shows exactly how the probability table enters the evaluation.

<!-- cell -->
```python
h_id = tokenizer.encode("h")[0]
e_id = tokenizer.encode("e")[0]

# Explain this step.
pair_probability = probabilities[h_id, e_id].item()
pair_loss = -math.log(pair_probability)

print("P(e | h):", round(pair_probability, 4))
print("loss for h -> e:", round(pair_loss, 4))
```
<!-- cell -->
The numbers come from the current project data. If the data text is changed, this probability and error are also changed. To evaluate the entire model, you calculate and average the error of all pairs; One couple is not enough to judge.

### First, calculate the error of several pairs with a loop

The average is the sum of the values divided by their number. There are four pairs in the text `hello`: `h → e`, `e → l`, `l → l`, and `l → o`. The circle below shows the probability and error of each separately.

<!-- cell -->
```python
check_ids = tokenizer.encode("hello")
total_loss = 0.0
pair_count = len(check_ids) - 1

for position in range(pair_count):
    previous_id = check_ids[position]
    following_id = check_ids[position + 1]

    observed_probability = probabilities[previous_id, following_id].item()
    observed_loss = -math.log(observed_probability)
    total_loss += observed_loss

    previous_character = tokenizer.decode([previous_id])
    following_character = tokenizer.decode([following_id])
    print(previous_character, "->", following_character, "loss:", round(observed_loss, 4))

mean_loss_loop = total_loss / pair_count
print("mean loss:", round(mean_loss_loop, 4))
```
<!-- cell -->
Less error of a pair means that the model has given more probability to the correct answer of that pair. The average of the four errors shows how the model performed in general for this text. These four pairs are not independent tests of the model's ability; Just a small example to understand the calculation.

### Do the same calculation with tensors of the same length

The slice `source[:-1]` gives all IDs except the last one. The cut `source[1:]` gives all IDs except the first one. These two lists are of equal length, the input and the moved targets.

When you put two tensors of the same length in `table[rows, columns]`, PyTorch reads the corresponding houses: the first row with the first column, the second row with the second column, and so on. This command does not select all combinations of rows and columns.

**Objective of the code:** First, see this selection of houses with a five-letter text and compare it with the previous loop.

<!-- cell -->
```python
check_tensor = torch.tensor(check_ids, dtype=torch.long)

# Explain this step.
previous_ids = check_tensor[:-1]
following_ids = check_tensor[1:]

# Explain this step.
observed_probabilities = probabilities[previous_ids, following_ids]
individual_losses = -observed_probabilities.log()
mean_loss_tensor = individual_losses.mean().item()

print("previous IDs:", previous_ids.tolist())
print("following IDs:", following_ids.tolist())
print("probabilities:", observed_probabilities.tolist())
print("mean loss:", round(mean_loss_tensor, 4))

assert math.isclose(mean_loss_loop, mean_loss_tensor, rel_tol=1e-6)
```
<!-- cell -->
The two methods give almost the same number. A very small difference may be due to the accuracy of decimal numbers; For this reason, we check the approximate equality. In the tensor version, the `log` function applies to all probabilities and the `mean` function applies to all errors.

### Build the model evaluation function

The following code performs the same calculation for each ID tensor. The probability table is read only; We don't add any counts when evaluating and the model doesn't change.

<!-- cell -->
```python
def bigram_loss(source):
    # Explain this step.
    if source.numel() < 2:
        raise ValueError("Evaluation needs at least two tokens.")

    previous_ids = source[:-1]
    following_ids = source[1:]
    observed = probabilities[previous_ids, following_ids]

    # Explain this step.
    mean_loss = -observed.log().mean()
    return mean_loss.item()

for split_name in ["train", "validation"]:
    split_loss = bigram_loss(data[split_name])
    print(split_name, "mean loss:", round(split_loss, 4))
```
<!-- cell -->
The training error shows how the model performs on the data it has learned from. Validation error is calculated on texts that were not used for counting. A large difference between the two can be a sign of the model's high dependence on training examples. A small difference alone does not prove general ability, especially when the training and validation data formats are similar.

The bigram table is the **baseline** or `baseline` of this course: a simple method to compare with a more complex method. If a neural model fails to outperform this simple method, we need to examine its data, training, and evaluation. In Lesson 05, you will measure Bigram and GPT on the same targets from sample categories to make the comparison more fair.

<!-- cell -->
## 9. What is perplexity and what does it not say?

**Preplexity** or `perplexity` is equal to the exponential function of the previous average error: `exp(mean_loss)`. To understand this, consider a completely uninformed model that gives all tokens the same probability. If four tokens are possible, the probability of each is one fourth; Its error is `log(4)` and its complexity is 4.

For this reason, perplexity can be considered as the number of effective choices among which the model hesitates. This is an interpretation to understand the standard; The number of actual vocabulary choices does not change with model evaluation.

**Code Objective:** Check this relationship for 4 options and then calculate the unknown model number for the project vocabulary.

<!-- cell -->
```python
# Explain this step.
uniform_probability_four = 1.0 / 4
uniform_loss_four = -math.log(uniform_probability_four)
uniform_perplexity_four = math.exp(uniform_loss_four)

print("four-choice loss:", round(uniform_loss_four, 4))
print("four-choice perplexity:", round(uniform_perplexity_four, 4))

# Explain this step.
uniform_loss = math.log(V)
print("project uniform loss:", round(uniform_loss, 4))
print("project uniform perplexity:", round(math.exp(uniform_loss), 4))
```
<!-- cell -->
For four options, the multiplexity is 4. For the current vocabulary of the project, the multiplexity of the uniform model is 83. Bigram table usually outperforms this uninformed model; Check the result by running the next cell.

<!-- cell -->
```python
for split_name in ["train", "validation"]:
    split_loss = bigram_loss(data[split_name])
    split_perplexity = math.exp(split_loss)

    print(
        split_name,
        "loss:", round(split_loss, 4),
        "perplexity:", round(split_perplexity, 2),
    )
```

<!-- cell -->
Perplexity is not an error percentage. For example, a value of 5 does not mean 5% incorrect answers or 95% correct answers. It summarizes the probabilities assigned to text targets; it does not evaluate whether the claims in the text are true.

To compare two models, keep the data and tokenization method the same. Predicting one character and predicting one word do not use the same measurement unit. Do not rank perplexities from different tokenizers directly. Also report the number of evaluated positions and the sampling method.

### Where does the probability of a whole sequence come from?

At each generated token, the model assigns a probability to one step. The probability of the whole sequence is the product of the step probabilities. For example, if two consecutive steps have probabilities 0.5 and 0.25, the probability of seeing both choices in sequence is 0.125.

A bigram model conditions each step only on the last token. A model with a longer context uses several previous tokens to calculate each probability. The general formula below expresses this idea; you do not need to memorize it for the rest of the course.

$$p(x_1,\ldots,x_n)=\prod_{t=1}^{n}p(x_t\mid x_1,\ldots,x_{t-1})$$

In this notation, `x_t` is the token at the current position, `p` denotes probability, and `∏` means to multiply the values from all steps. There is no previous text for the first token, so its probability comes from the model's starting rule. In this lesson's generation function, the prompt is supplied externally, so we do not model the probability of starting the text.

For a long text, multiplying probabilities directly can produce a very small number. The logarithm turns this product into a sum of logarithms. This is one reason evaluation commonly works with log probabilities.

<!-- cell -->
## 10. Distinguish model weights, context, and application memory

In technical discussions, the word “memory” can refer to several different things. A programming example makes the distinction clear: an application's configuration file, the current request input, and database records can all affect the output, but they are not the same thing.

**Model parameters** are the values that determine the model's behavior. In this lesson's bigram model, the probability table plays that role and is built from counts. In a neural network, these trainable values are usually called **weights** or `weight`. In Lesson 00 you learned one numeric weight by reducing an error; GPT trains many tables and matrices.

**Context** is the set of tokens you make available to the model for the current request. Writing a new name in a prompt does not change the model weights during ordinary generation. The name is used as part of that input; it is not learned permanently by default.

**Application memory** is data outside the model, such as a file, database, or saved conversation history. The application can read this memory and place the required text into the context. If it does not do so, storing a record in a database does not make the model aware of it.

In a transformer, there is also **temporary computational memory** called the `KV cache`. It keeps part of the computations for previous tokens so generation does not repeat all of them. `K` and `V` are the names of two vector types explained in Lesson 03. For now, remember that this tool improves speed; it is not a database of user information. This lesson's bigram model has no such cache.

| Concept | Concrete example | How does it change? |
|---|---|---|
| Model parameter | Bigram probability table or GPT weights | When the model is built from data or the weights are trained |
| Request context | The question and a few previous sentences | When the input changes and output is appended |
| Application memory | A customer record in a database | When application code reads or writes it |
| Temporary computational memory | Cached computations for previous tokens | During generation as new tokens are added |

**Example goal:** Take one record from an application dictionary and turn it into context text. This cell does not train a model.

<!-- cell -->

```python
# Explain this step.
customer_records = {
    "Nila": {"remaining_credit": 250},
}

# Explain this step.
record = customer_records["Nila"]
remaining_credit = record["remaining_credit"]

# Explain this step.
context_record = f"Customer Nila has {remaining_credit} units of remaining credit."

print("record from application:", context_record)
```
<!-- cell -->
The number 250 comes from the record. The model did not guess it. This cell only shows valid data reading and context preparation; Bigram in this lesson is not able to answer the question about the client, and putting this sentence in the prompt does not make it a reliable assistant.

<!-- cell -->
## 11. Generating fluent text is not the same as correct answers

If you ask a model for the card number of a hypothetical person, it may produce a string similar to the card number. Being familiar with the number format does not mean having the actual record of that person. Making a claim that lacks sufficient support in the data or source provided is commonly called a **model illusion** or `hallucination`.

In a real application, you read a person's data from a valid source and perform detailed calculations with code. The model can help in writing the text, but the fluency of the answer does not take the place of checking the source and calculation.

**source fidelity** means that the answer is consistent with the document you have given. **world truth** means that the claim is consistent with the current reality. The model may accurately summarize the old document; The answer is faithful to the document, but its information may no longer be correct.

**Objective of the example:** Perform a deterministic calculation in Python. This result does not come from language-model sampling.

<!-- cell -->
```python
# Explain this step.
item_count = 173
unit_price = 249
total_price = item_count * unit_price

print("exact total:", total_price)
assert total_price == 43077
```
<!-- cell -->
The result is 43077. This example is meant to separate the calculation responsibility from the text writing. We do not use Bigram to test the responsiveness or measure the illusion of an assistant; This model is not designed for that task at all.

### What is the difference between the basic model and the commandable model?

**Basic model** or `base model` is trained on predicting the continuation of the text. For example, by seeing the beginning of a sentence, it learns which continuation is more likely in the data. It's not intended to be an "adequate response to a user's request" per se.

**Commandable model** or `instruct model` is adapted with appropriate request and response examples. One way to do this is **supervised fine-tuning** or `SFT`: the model has examples where the desired request and response are known, and its weights are trained for those responses.

Another method is "preference training": among several answers, the better answer is preferred and the training process uses it. The `RLHF` method is one of the methods based on human feedback; The details of its algorithm are not necessary in this lesson.

The model of this course is basic. Adding the sentence "You are a helper" to the prompt alone does not complete the command acceptance training step. Changes the input prompt; Fine-tuning changes the model weights. In lesson 07, you can see the path of learning in more detail.

<!-- cell -->
## 12. Token budget: how much text will fit?

A model typically cannot see unlimited text in a single calculation. We call the number of available slots **field length** or `context length`. In the model you create this course, the length of the field is 64 tokens. Because the tokens of this model are letters, 64 tokens are not the same as 64 words.

In a talk show, the input is not just the last question. Program command, chat history and added documents also consume tokens. For planning, it is necessary to calculate the contribution of each section and the space required for the output.

The following example belongs to a hypothetical service with a capacity of 16,000 tokens; This is not the capacity of TinyGPT of this course. All numbers, rates and speeds are hypothetical and do not represent the price or specifications of a provider. The local service of this billing period does not have an external API, but consumes CPU time and computer memory.

**Purpose of the code:** Add the hypothetical numbers and check if the input and output space fit in the capacity.

<!-- cell -->
```python
# Explain this step.
parts = {
    "instructions": 1000,
    "history": 2000,
    "documents": 6000,
    "question": 100,
}

# Explain this step.
input_tokens = sum(parts.values())

# Explain this step.
output_reserve = 1000
context_limit = 16000
total_reserved = input_tokens + output_reserve
fits = total_reserved <= context_limit

print("input tokens:", input_tokens)
print("input plus output reserve:", total_reserved)
print("fits the assumed context:", fits)
```
<!-- cell -->
The input is 9100 tokens and with the output space, the total reaches 10100. The output of `True` means that this value is below the assumed ceiling of 16000. The rules of the services may be different about the input and output ceilings; This example assumes only one common roof.

### Separate cost and time estimates from actual data

For a computational example, suppose one million input tokens cost $2 and one million output tokens cost $10. Calculate the cost ceiling assuming the full use of the output space. In practice, if the output is shorter, its contribution will be lower.

For a simple time estimate, we have three hypothetical values: a fixed time of 0.2 seconds, processing 5000 input tokens per second, and generating 50 output tokens per second. This relationship is not a performance measurement; It just shows how input and output lengths affect programming.

<!-- cell -->
```python
# Explain this step.
input_rate_per_million = 2.0
output_rate_per_million = 10.0

input_cost = input_tokens * input_rate_per_million / 1_000_000
output_cost_ceiling = output_reserve * output_rate_per_million / 1_000_000
cost_ceiling = input_cost + output_cost_ceiling

# Explain this step.
fixed_seconds = 0.2
assumed_input_tokens_per_second = 5000
assumed_output_tokens_per_second = 50

latency_seconds = (
    fixed_seconds
    + input_tokens / assumed_input_tokens_per_second
    + output_reserve / assumed_output_tokens_per_second
)

print("assumed cost ceiling, USD:", round(cost_ceiling, 4))
print("assumed total seconds:", round(latency_seconds, 2))
```
<!-- cell -->
The cap for this example is $0.0282 and the assumed simple time is 22.02 seconds. Actual time depends on hardware, model, request batch size and service conditions and should be measured. Do not confuse the hypothetical calculation with the measurement result for comparing implementation options.

### How does sliding window work with long output?

In this course's TinyGPT, when the text exceeds the field length, only the last 64 tokens are included in the calculation of the next token. We call this method **sliding window**. The model can generate more than 64 output tokens, but the very old text is no longer in its current input.

To see the behavior, consider a window size of 4 instead of 64. This example only shows the selection of the last part of the directory and does not run GPT.

<!-- cell -->
```python
# Explain this step.
all_ids = [10, 11, 12, 13, 14, 15, 16]
small_context_limit = 4

# Explain this step.
visible_context = all_ids[-small_context_limit:]

print("full generated sequence:", all_ids)
print("visible context:", visible_context)
print("visible count:", len(visible_context))
```
<!-- cell -->
The visible field is `[13, 14, 15, 16]`. The full list still has seven identifiers; Only the next calculation input is limited to four. Having a long output does not mean seeing it all at every step.

The bigram of this lesson is more limited: it always sees only the last token to calculate the probability. Neither of these two methods automatically save or retrieve the out-of-context record in the database; It is the responsibility of the program around the model.

<!-- cell -->
## 13. Interview exercises and answers

Before opening the answers, explain each question in your own language. If you have to use a term that you can't explain with an example, go back to the relevant section.

1. For the text `ababac`, without smoothing, what is the probability of `b` after `a`? What happens by adding one?

2. Why does Bigram create the same distribution for the two inputs `The ` and `A `?

3. What does changing the seed change? What does temperature change change? Is there any retraining?

4. If the model gives the target a probability of 0.1, is the error greater or the case that gave a probability of 0.5?

5. Why low error or low complexity does not prove the correctness of a generated card number?

6. What is the difference between prompting the model, adjusting the model and reading records from the database?

7. In the budget example, increase the history from 2000 to 12000 tokens. Does it fit in the ceiling with the same outlet space?

8. If you have a letter model and a model with multi-letter tokens, why not directly compare their complexity?

<!-- fa-lint-ignore-next-line -->
<details dir="rtl"><summary>Open suggested answers after your own answer</summary>

**Answer 1:** After `a`, we have seen `b` twice and `c` once; So the initial probability is two thirds. With smoothing, the line becomes `[1, 3, 2]` and the sum is 6; The probability of `b` is one in two.

**Answer 2:** The last token is a space in both prompts. The bigram model selects a table row using only that final token; it ignores the older text.

**Answer 3:** The seed determines the path of pseudo-random selections. Temperature makes the probabilities of points more concentrated or more spread out. The table or model weights are not trained with these two changes.

**Answer 4:** The probability of 0.1 error is higher; The negative of its logarithm is about 2.303, while for 0.5 it is about 0.693. The criterion gives the model with a higher probability for the desired answer, a lower error.

**Answer 5:** The measure examines the likelihood of textual targets. The correctness of the card number requires a valid record; Being familiar with the number format is not enough. Even exact repetition of a data number does not prove its current ownership.

**Answer 6:** The prompt modifies the request input. Fine-tuning changes model weights by training on examples. Reading the database fetches the external data with the application code and adds context as needed.

**Answer 7:** The input 19100 and the output space 2010 are tokens; It does not fit under the ceiling of 16,000. You must either truncate the history or documents with a clear rule, or reject the request. Indiscriminate deletion may make necessary information unavailable.

**Answer 8:** The prediction unit is different. One target may be one letter and another several letters. Without the same tokenizer and evaluation data, these two numbers have no directly comparable meaning.

</details>

<!-- cell -->
### Actionable exercise: Review the new budget

This cell makes a copy of the original dictionary so that it does not change the values of the previous cells. Its purpose is to check the answer to the seventh question with the code.

<!-- cell -->
```python
# Explain this step.
larger_parts = parts.copy()
larger_parts["history"] = 12000

larger_input_tokens = sum(larger_parts.values())
larger_reserved_tokens = larger_input_tokens + output_reserve
larger_fits = larger_reserved_tokens <= context_limit

print("new input tokens:", larger_input_tokens)
print("new reserved total:", larger_reserved_tokens)
print("fits:", larger_fits)

assert larger_input_tokens == 19100
assert larger_reserved_tokens == 20100
assert not larger_fits
```
<!-- cell -->
### Can you explain the path of the model?

Express the course of this lesson without looking at the code: the text is converted into an ID; Adjacent pairs were counted; The count of each row became a probability; A token is selected from the possibilities; And by repeating this choice, the text was made. In the evaluation, instead of random selection, the probability of the next true token was read from the table.

In the next lesson, you will dwell more on the first step: why the model does not take the text directly, how the tokenizer works, and how the identifier is converted to a vector of numbers? After that, you replace the enumeration table with a model that can use multiple context tokens.

**next lesson:** [02. tokenization and embeddings] (02_tokenization_and_embeddings.ipynb)
