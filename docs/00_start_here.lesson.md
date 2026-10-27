# 00. From the first cell to learning a number

To build a language model, you need to know how text is converted to numbers and how the program learns from examples. This lesson starts with the same numbers. Previous knowledge of tensor, neural network or derivative is not necessary.

First you run a simple cell. Then you work with numbers, lists, and tables and create their equivalents in the PyTorch library. In the end, you will have a program that changes the value of an adjustable number so that its answer is closer to the desired answer.

The name of the **PyTorch** library in the code is `torch`. This library provides tools for storing numbers, performing calculations, and calculating error variances. We explain each feature when it is needed in the example.

**Study method:** Read the explanation before each cell, run the code and then go to the output interpretation. You do not have to finish this lesson in one sitting. After the tensors section and after the text to number conversion section, you can stop.

If this file is already open in JupyterLab, the initial installation is probably done. Continue with the next section. The installation guide for a new device is at the end of this lesson.

<!-- cell -->

## 1. What is a notebook and where does the code run?

**Notebook** is a file with the extension `.ipynb` that keeps the tutorial text, the code and the result of the code execution together. Each separate piece on the screen is a **cell**.

A text cell, like this one, shows the explanation. Its text is written in a format called **Markdown**; For example, the symbol `#` at the beginning of a line converts that line to A code cell executes Python commands. For this lesson, you will mostly read only text cells and execute code cells.

**Kernel** is a program that executes Python code and keeps variables in memory. The browser displays the page; In this project, the kernel runs on the same computer.

To run a code cell, click on it and press `Shift+Enter`. The number next to the cell, such as `[1]`, is its execution number in the current kernel. This number is not the lesson number or line number. The sign `[*]` means that the execution of the cell has not finished yet.

The display setting cell at the beginning of the page is also a Python cell. The `HTML` and `display` tools give the browser text orientation settings. The long text inside is the instructions for the appearance of the page. To follow the lesson, it is enough to implement it; The details of appearance adjustment are not the subject of model learning.

**Goal of the next example:** Create a variable and see the result below the cell. A variable is the name by which you access a value. The `print` command shows the value in the output. The text after `#` is a comment; Python doesn't run it.

<!-- cell -->
```python
# Explain this step.
lesson_number = 2

# Explain this step.
print("lesson_number:", lesson_number)

# Explain this step.
print("lesson_number + 3:", lesson_number + 3)
```
<!-- cell -->

The first output is 2 and the second output is 5. English labels are only for identifying outputs. The second calculation did not change the value of the variable; Only the sum has been shown.

**Goal of the next example:** See that a cell can use a variable created in a previous cell.

<!-- cell -->
```python
# Explain this step.
print("still in memory:", lesson_number)

# Explain this step.
next_number = lesson_number + 1

# Explain this step.
print("next_number:", next_number)
```
<!-- cell -->

### Why is the execution order of the cells important?

If the previous cell has not been executed, Python does not recognize the name `lesson_number` and returns the error `NameError`. This error in such a case means "the variable has not been created yet".

Save file keeps saved text and output. **Variables in memory are not preserved by file saving.** If you restart the kernel with the `Restart Kernel` option, the variable memory is cleared. The old output may still be on the screen, but the corresponding variable no longer exists. To rebuild the variables, run the cells from top to bottom.

**Short exercise:** Change the number 2 in the first cell to 7. Run the same cell and then the second cell. The value of the `next_number` variable should be 8. Then return the initial number to 2 so that the lesson examples remain the same as the explanations.

<!-- cell -->

## 2. Are you running the correct Python?

There may be several versions of Python installed on Windows. Each version can also have different libraries. **virtual environment** is a folder that keeps Python and libraries specific to this project separate from other projects. In this course, the name of this folder is `.venv`.

The `import` command makes a library available for use in Python cells. The `sys` section contains information about running Python. The `sys.executable` path indicates exactly which Python file the kernel is using.

**Code target:** See Python version, its path and PyTorch version. This cell does not make any model and does not do training.

<!-- cell -->
```python
# Explain this step.
import sys

# Explain this step.
import torch

# Explain this step.
print("Python version:", sys.version.split()[0])

# Explain this step.
print("Python executable:", sys.executable)

# Explain this step.
print("PyTorch version:", torch.__version__)
```
<!-- cell -->

In this project, the python path should reach the folder `.venv` inside `tiny-llm-fa`. The PyTorch version of this course is `2.7.1`; An extension like `+cpu` may also be seen. The CPU is the main processor of the computer. The **GPU** processor performs many calculations at the same time and is used in training large models. For the examples in this lesson, the CPU is sufficient.

If the Python path does not point to this project's environment, select `Python (tiny-llm-fa)` from the kernel selection section in JupyterLab and run the previous cells again. If the message `No module named 'torch'` is seen, the library is not installed in the environment used by the kernel. The installation guide is at the end of the lesson.

<!-- cell -->

### Finding the project folder, step by step

Sometimes the kernel starts from the `notebooks` folder and sometimes from the main project folder. To access the data files, we need to find the root folder. The `Path` tool from the `pathlib` library simplifies working with file paths. This library is installed with Python.

In the following code, the value of `Path.cwd()` gives the path of the current working folder. The `parents` feature introduces higher folders. For each folder, we check if the path `src/tinyllm` exists inside it. The first folder that has this condition is the root of the project and we keep it with the name `ROOT`.

This section is only for accessing project files. The use of the name `ROOT` is a convention in the code of this course; Python has no special meaning for this name.

<!-- cell -->
```python
# Explain this step.
from pathlib import Path

# Explain this step.
current_folder = Path.cwd()

# Explain this step.
candidate_folders = [current_folder]

# Explain this step.
candidate_folders.extend(current_folder.parents)

# Explain this step.
ROOT = None

# Explain this step.
for folder in candidate_folders:
    # Explain this step.
    source_folder = folder / "src" / "tinyllm"
    # Explain this step.
    if source_folder.is_dir():
        ROOT = folder  # Explain this step.
        break  # Explain this step.

# Explain this step.
if ROOT is None:
    raise FileNotFoundError("Open this notebook from the tiny-llm-fa folder.")

# Explain this step.
print("Project folder:", ROOT.name)
```
<!-- cell -->

The output should show the name `tiny-llm-fa`. The command `break` stops the loop. The `raise` command stops normal execution with an error; Here the error helps not to confuse the path problem with the model calculation problem.

We don't need to change the search path for Python libraries yet. This tutorial only uses the installed libraries. In the model building lesson, it is explained how the code inside `src/tinyllm` enters the program.

<!-- cell -->

## 3. Know tensor from number, list and table

The language model works with numerical calculations. Before we get to the text, we need to know how the numbers fit together.

In Python, a number like 10 can be the value of a variable. How many numbers can be stored in a list? If you put several lists of the same size together, you will have a number table. In this example, each row of the table contains three numbers.

**Code Purpose:** See familiar shapes in regular Python. No tensors are created yet.

<!-- cell -->
```python
# Explain this step.
one_number = 10

# Explain this step.
number_list = [10, 20, 30]

# Explain this step.
number_rows = [[10, 20, 30], [40, 50, 60]]

# Explain this step.
print("one number:", one_number, "type:", type(one_number))

# Explain this step.
print("list:", number_list, "length:", len(number_list))

# Explain this step.
print("rows:", number_rows, "row count:", len(number_rows))
```
<!-- cell -->

### What is a tensor?

In this course, consider Tensor as a container for storing numbers in PyTorch. This container can hold a number, a numeric list, a table, or a combination with more axes. In addition to values, it also stores information such as the shape and type of numbers.

Tensor is built for numerical calculations. For example, you can multiply all the values ​​of a table by 2 with one command. Using PyTorch, it is possible to calculate error changes with respect to trainable tensors; You will find this application in the learning section.

The `torch.tensor(...)` tool builds a tensor from the values ​​you give it. The first example has only one number. A single number is called a scalar.

Know two features from the beginning:

- `shape` property says how many positions are there in each axis.
- `ndim` attribute tells the number of tensor axes. In the PyTorch code, the word "dimension" is also used for the same number of axes.

**Objective of the code:** Convert a normal number to a tensor and see what the tensor of a number looks like.

<!-- cell -->
```python
# Explain this step.
scalar = torch.tensor(10.0)

# Explain this step.
print("scalar:", scalar)

# Explain this step.
print("shape:", scalar.shape)

# Explain this step.
print("number of axes:", scalar.ndim)

# Explain this step.
print("ordinary Python number:", scalar.item())
```
<!-- cell -->

The scalar shape is seen as `torch.Size([])` and the number of its axes is zero. This empty form does not mean that the data is empty; The tensor still has the number 10. No need for row or column number just to find it.

The command `item()` is used for a tensor that has exactly one number. We use `tolist()` to convert a tensor with multiple numbers to a regular Python list.

<!-- cell -->

### Vector: several numbers on one axis

A one-dimensional numeric list is called a Vector. The following vector has three numbers and its form is `(3,)`. The comma in this representation is the Python convention for a one-member array of sizes; Here we have only one size, namely 3.

The term "three-dimensional vector" is different from the term "three-dimensional tensor". This vector has three numbers, but only **one axis**. Do not confuse the number of numbers in one axis with the number of axes of the whole tensor.

<!-- cell -->
```python
# Explain this step.
vector = torch.tensor([10.0, 20.0, 30.0])

# Explain this step.
print("vector:", vector)

# Explain this step.
print("shape:", vector.shape)

# Explain this step.
print("number of axes:", vector.ndim)

# Explain this step.
print("Python list:", vector.tolist())
```
<!-- cell -->

### Matrix: A table with rows and columns

A biaxial numerical table is called a Matrix. In this example, the first axis is rows and the second axis is columns. The form `(2, 3)` means two rows and three columns. The number of axes of this tensor is two; The total number of its numbers is six.

With the `numel()` tool, you can get the total number of numbers in the tensor. Its name comes from the English phrase "number of elements".

<!-- cell -->
```python
# Explain this step.
matrix = torch.tensor([
    [10.0, 20.0, 30.0],  # Explain this step.
    [40.0, 50.0, 60.0],  # Explain this step.
])

# Explain this step.
print("matrix:", matrix)

# Explain this step.
print("shape:", matrix.shape)

# Explain this step.
print("number of axes:", matrix.ndim)

# Explain this step.
print("total numbers:", matrix.numel())
```
<!-- cell -->

**Stop and practice:** Guess the shape of the tensor `torch.tensor([[1, 2], [3, 4], [5, 6]])` before running. How many axes does it have? How many numbers are in it?

<!-- fa-lint-ignore-next-line -->
<details dir="rtl"><summary>Open the answer after guessing</summary>

This tensor has three rows and two columns; Its form is `(3, 2)`. It has two axes and six numbers are stored in it. The number of rows is not the number of tensor axes.

</details>

<!-- cell -->

### Access a number, a row or a column

To select values, we use **index**; Index is the number one position. Python and PyTorch numbering starts at zero. In the previous matrix, the first row has the number zero and the second row has the number one.

In the expression `matrix[1, 2]`, the first number is the row number and the second number is the column number. The symbol `:` means to select all positions of that axis. So the statement `matrix[:, 0]` will remove the column number zero from all the rows.

**Code Purpose:** Before executing each line, find the selected value from the previous table.

<!-- cell -->
```python
# Explain this step.
print("first vector number:", vector[0].item())

# Explain this step.
print("second row:", matrix[1].tolist())

# Explain this step.
print("row 1, column 2:", matrix[1, 2].item())

# Explain this step.
print("first column:", matrix[:, 0].tolist())
```
<!-- cell -->

You will see the number 10, the list `[40.0, 50.0, 60.0]`, the number 60, and the list `[10.0, 40.0]`, respectively.

Selecting a row from the matrix gives a vector. Choosing a house gives a scalar. In the model code, the selection of values ​​is done in the same way; Only the number of axes of tensors will be more.

<!-- cell -->

### Type of numbers: identifier is different from calculable value

The **`dtype`** property specifies the type of numbers inside the tensor. The two types that are seen a lot in this period are:

- The `torch.long` type is the same as `torch.int64` and holds an integer. For the number of a place or the identifier of a letter, we need an integer.
- `torch.float32` type keeps a decimal number. For values ​​that are going to be set in small increments, like 1.2 or 1.21, we need a decimal type.

The number "64" in the name of the integer type and the number "32" in the name of the decimal type are the number of storage bits of each number. A bit is the basic unit of storage in a computer. These numbers do not express the number of axes of the tensor or the length of the vector.

**Purpose of the code:** Select the type of numbers explicitly and see the storage location of the tensor. The `device` attribute tells which processor the tensor resides on.

<!-- cell -->
```python
# Explain this step.
positions = torch.tensor([0, 1, 2], dtype=torch.long)

# Explain this step.
measurements = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32)

# Explain this step.
print("positions dtype:", positions.dtype)

# Explain this step.
print("measurements dtype:", measurements.dtype)

# Explain this step.
print("device:", measurements.device)
```
<!-- cell -->

In this lesson, the device output is `cpu`. A regular tensor is not necessarily trainable; It only holds numbers. When we get to learning, we specify which tensor to adjust. PyTorch does not enable gradient calculation for integer types such as `long`; The parameters of the training example will be decimal.

<!-- cell -->

### Calculation on several numbers with one command

**Code Purpose:** See how a command is applied to all numbers in a vector. Adding or multiplying a vector by a number, in this example, is done on individual houses. Multiplication of two isomorphic vectors marked `*` also multiplies the corresponding houses.

<!-- cell -->
```python
# Explain this step.
values = torch.tensor([1.0, 2.0, 3.0])

# Explain this step.
print("add 10:", (values + 10).tolist())

# Explain this step.
print("multiply by 2:", (values * 2).tolist())

# Explain this step.
print("elementwise product:", (values * values).tolist())

# Explain this step.
print("sum of all values:", values.sum().item())
```
<!-- cell -->

The result of the total sum is 6. The statement `values.sum().item()` does two things in a row: first it calculates the sum as a tensor, then it converts it to a regular Python number.

### Addition along an axis

In a matrix, the numbers of each row or each column can be added separately. The number of the first axis is zero and the number of the second axis is one. The `dim` option specifies **which axis to delete when collapsing**.

In the matrix `(2, 3)`, addition with `dim=0` adds two rows together, leaving three results, one for each column. Sum with `dim=1` sums the columns of each row, leaving two results, one for each row.

<!-- cell -->
```python
# Explain this step.
print("matrix:", matrix)

# Explain this step.
column_totals = matrix.sum(dim=0)

# Explain this step.
row_totals = matrix.sum(dim=1)

# Explain this step.
print("column totals:", column_totals.tolist(), "shape:", column_totals.shape)

# Explain this step.
print("row totals:", row_totals.tolist(), "shape:", row_totals.shape)
```
<!-- cell -->

The sum of the columns is equal to `[50.0, 70.0, 90.0]` and has the form `(3,)`. The sum of the rows is equal to `[60.0, 150.0]` and has the form `(2,)`. Follow the meaning of `dim` from the figure before and after the operation; Don't just keep the "row" or "column" names.

<!-- cell -->

### How is matrix multiplication different from house-to-house multiplication?

Models often use sum of products. Suppose each row of the matrix has two input values ​​and the vector of coefficients tells what number to multiply each value. For the line `[1, 2]` and coefficients `[10, 20]`, the result is equal to `1 * 10 + 2 * 20`, i.e. 50.

The symbol `@` performs this matrix calculation. In this example, the matrix has two columns; So the vector of coefficients must also have two numbers. The PyTorch tool performs this multiplication and addition for each row.

<!-- cell -->
```python
# Explain this step.
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Explain this step.
coefficients = torch.tensor([10.0, 20.0])

# Explain this step.
weighted_sums = inputs @ coefficients

# Explain this step.
print("matrix product:", weighted_sums.tolist())

# Explain this step.
print("first row by hand:", 1 * 10 + 2 * 20)
```
<!-- cell -->

So far, we have chosen the coefficients ourselves. In the learning section, the program sets the value of a coefficient from an example. In a large model, the same idea continues with many adjustable numbers and more calculations.

<!-- cell -->

### Tensor with three axes, without language model terms

Let's say you keep two days' worth of sales for two stores, and each day has three products. The order of axes in this example is as follows: **store, day, product**. The form `(2, 2, 3)` means two stores, two days for each store and three values ​​for each day.

Tensor does not know the meaning of "store" or "day". The programmer specifies what each axis means. The shape only holds the number of slots.

<!-- cell -->
```python
# Explain this step.
sales = torch.tensor([
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Explain this step.
    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # Explain this step.
])

# Explain this step.
print("shape:", sales.shape, "number of axes:", sales.ndim)

# Explain this step.
print("second store:", sales[1])

# Explain this step.
print("store 1, day 0, product 2:", sales[1, 0, 2].item())

# Explain this step.
print("total numbers:", sales.numel())
```
<!-- cell -->

The selected value is 9. The total number of values ​​is 12. Selecting `sales[1]` omits one axis and returns a matrix of the form `(2, 3)`.

**First stop:** If the shape of the tensor and the meaning of choosing a house are clear to you, continue. Otherwise, change the values ​​in the table and find some houses with their axis numbers. You don't have to memorize the language model terms yet.

<!-- cell -->

## 4. How is text converted to numbers?

A language model wants to predict the continuation of the text from the previous text. To perform calculations, the text is first divided into small units. Each unit is called a Token. In this course, to keep the code simple, each character is a token. In other models, the token can be a letter, part of a word, or a word.

A Tokenizer divides the text into tokens according to a rule and assigns a numerical identifier to each token. In the following example, we write this rule ourselves. We have four possible tokens: the letter "A", the letter "B", the letter "A" and the symbol "?". We call the set of possible tokens **vocabulary**; Even if the members are letters and not words.

For the text "Bob", the order of tokens is "B, A, B". If the identifier "a" is zero and the identifier "b" is one, the numerical representation of the text is equal to `[1, 0, 1]`. These identifiers are labels. A larger ID does not mean more importance or more similarity.

**Code Purpose:** See letter to ID conversion with a simple dictionary. The Python dictionary holds the value for each key; Here the key is the letter and the value is the identifier of that letter.

<!-- cell -->
```python
# Explain this step.
char_to_id = {"ا": 0, "ب": 1, "آ": 2, "؟": 3}

# Explain this step.
text = "باب"

# Explain this step.
ids_list = []

# Explain this step.
for character in text:
    token_id = char_to_id[character]  # Explain this step.
    ids_list.append(token_id)  # Explain this step.

# Explain this step.
print("token IDs:", ids_list)

# Explain this step.
ids = torch.tensor(ids_list, dtype=torch.long)

# Explain this step.
print("IDs tensor:", ids, "shape:", ids.shape)
```
<!-- cell -->

This token generator only recognizes the four defined characters. If you put another letter in the text, for example "S", the dictionary search will give the error `KeyError`; It means that a value is not defined for that key. Building a more complete tokenizer is the topic of the next Text and Tokens notebook.

### How is an identifier converted to a vector?

The ID just tells which token was selected. For model calculations, each token can be assigned a numeric vector. We call the table of these vectors **embedding table**. Each row corresponds to an identifier and each column holds one of the vector numbers.

In the exercise table below, we have chosen two numbers for each token. These numbers are set manually for the readability of the example and do not have a learned linguistic meaning. In the main model of the course, the numbers in such a table are among the teachable parameters and change during training.

**Code Purpose:** Select the line corresponding to an ID and then the lines corresponding to all IDs in the text.

<!-- cell -->
```python
# Explain this step.
embedding_table = torch.tensor([
    [1.0, 2.0],  # Explain this step.
    [3.0, 4.0],  # Explain this step.
    [5.0, 6.0],  # Explain this step.
    [7.0, 8.0],  # Explain this step.
], dtype=torch.float32)

# Explain this step.
letter_id = char_to_id["ب"]

# Explain this step.
print("vector for the letter:", embedding_table[letter_id].tolist())

# Explain this step.
text_vectors = embedding_table[ids]

# Explain this step.
print("vectors for the text:", text_vectors)
print("shape:", text_vectors.shape)
```
<!-- cell -->

For the text "Bob", the rows of the table are selected as one, zero and one respectively. The output has three lines: `[3.0, 4.0]`, then `[1.0, 2.0]`, and again `[3.0, 4.0]`. The resulting form is `(3, 2)`; Three tokens and two numbers for each token.

This example is just selecting rows from a static table. The model has not learned anything yet. In the embedding lesson, you will also check the difference between this table and the numerical display ready for text search.

<!-- cell -->

### Give some text together to the model

A group of samples that are processed together is called a batch. Here we have two three-character texts: "Bob" and "Water?". Equality of lengths simplifies the construction of the table. The method of working with unequal lengths is explained in the next lessons.

**Objective of the code:** Create an ID table with two rows and three columns from two texts. Then, for each house of that table, take a binary vector.

<!-- cell -->
```python
# Explain this step.
texts = ["باب", "آب؟"]

# Explain this step.
encoded_rows = []

# Explain this step.
for sample_text in texts:
    row_ids = []  # Explain this step.
    for character in sample_text:  # Explain this step.
        row_ids.append(char_to_id[character])  # Explain this step.
    encoded_rows.append(row_ids)  # Explain this step.

# Explain this step.
batch_ids = torch.tensor(encoded_rows, dtype=torch.long)

# Explain this step.
hidden = embedding_table[batch_ids]

# Explain this step.
print("batch IDs:", batch_ids, "shape:", batch_ids.shape)

# Explain this step.
print("batch vectors:", hidden, "shape:", hidden.shape)

# Explain this step.
print("first token vector:", hidden[0, 0].tolist())
```
<!-- cell -->

Identifiers have the form `(2, 3)` and vectors have the form `(2, 3, 2)`. These three axes have the same idea as the store example, but the meaning of their axes has changed: the text sample, the position of the token in the text, and the vector numbers of each token.

In model code, short names are usually used for these sizes. Now that you have seen their example, recognize these names:

| Name in the code Meaning in this example amount |
|---|---|---|
| Symbol `B` | The number of texts in a category 2 |
| Icon `T` | The number of tokens of each text 3
| Icon `C` | The number of vector numbers of each token 2 |
| Icon `V` | The number of different tokens in the vocabulary 4 |

The variable name `hidden` here refers to the numeric representation of the text. The term "hidden display" in the models means the intermediate values ​​of the calculation; This name does not mean the existence of hidden data.

**Exercise:** What will be the shape of the tensor for four five-token texts, with six numbers in each token vector?

<!-- fa-lint-ignore-next-line -->
<details dir="rtl"><summary>Open the answer after guessing</summary>

Identifiers have the form `(4, 5)` and vectors have the form `(4, 5, 6)`. Each unique identifier was a number; Now there are six numbers instead. Vocabulary size cannot be obtained from these three sizes and must be determined separately.

</details>

<!-- cell -->

## 5. Why are random numbers seen in the model code?

When the training is not yet done, we do not know the appropriate value of the model parameters. In many models, the initialization is done with random numbers. Then the training changes these numbers. A prime random number alone has no linguistic knowledge.

The `torch.randn` tool generates random numbers. In this example, the numbers come from a given distribution that has zero mean and positive and negative values. For now, it is enough to know that this tool creates a numerical table with the requested form.

To repeat an experiment, you can specify the starting point of generating random numbers. This amount is called **seed**. The number 7 in this example is an arbitrary choice; The number is not the quality of the model or the number of parameters.

**Objective of the code:** Create two separate producers with the same seed and see that they produce the same table in the same environment. The `torch.Generator` utility keeps the state of random number generation.

<!-- cell -->
```python
# Explain this step.
generator_a = torch.Generator()

# Explain this step.
generator_a.manual_seed(7)

# Explain this step.
random_a = torch.randn(2, 2, generator=generator_a)

# Explain this step.
generator_b = torch.Generator()

# Explain this step.
generator_b.manual_seed(7)

# Explain this step.
random_b = torch.randn(2, 2, generator=generator_b)

# Explain this step.
print("first table:", random_a)
print("second table:", random_b)
print("equal:", torch.equal(random_a, random_b))
```
<!-- cell -->

The label `equal` must show the value `True`; It means that two tables are equal in this comparison. If you change the seed of the second producer, you will usually see other values.

If you get numbers from the same generator again and do not reset the seed, the number generation sequence continues. The same seed alone does not guarantee that all model computations are exactly the same on every device and every software version.

**Second stop:** So far you have created tensors and converted text to identifiers and vectors. No table has been trained. In the next section, you will see the difference between "making a value" and "learning a value by example".

<!-- cell -->

## 6. See learning with an adjustable number

In this section, the model is a very small function: it multiplies the input by a factor. If the input is `x` and the coefficient is `w`, the output is obtained from the relation `prediction = w * x`. We use the name **Prediction** for the output of the model.

Suppose we have a training example: when the input is 2, the desired response is 6. The desired response value is called the target. The appropriate factor for this relationship is 3, but we start the program with a factor of 1. The initial output will be 2.

The adjustable number `w` is a **parameter**. A parameter is a value inside the model that the training modifies. The input and target come from the training example; The tutorial is supposed to set the parameter.

**Code Objective:** First, run the model in plain Python only. The symbol `*` here is the multiplication of two numbers.

<!-- cell -->
```python
# Explain this step.
x = 2.0

# Explain this step.
target = 6.0

# Explain this step.
weight = 1.0

# Explain this step.
prediction = weight * x

# Explain this step.
print("prediction:", prediction)
print("target:", target)
```
<!-- cell -->

### How to convert the error into a number?

To adjust the parameter, we need to measure the error rate of the model. **Error function or Loss** converts the difference between prediction and target into a number. In this example, we raise the difference to the power of 2: `loss = (prediction - target) ** 2`.

The difference can be positive or negative. By squaring, the error value in this example does not become negative and is zero only when the prediction is exactly equal to the target. The symbol `** 2` in Python means "to the power of two".

For prediction 2 and target 6, the difference is negative 4 and the squared error becomes 16. Reducing this error will be the goal of parameter adjustment.

<!-- cell -->
```python
# Explain this step.
difference = prediction - target

# Explain this step.
loss_value = difference ** 2

# Explain this step.
print("difference:", difference)

# Explain this step.
print("squared error:", loss_value)
```
<!-- cell -->

### What is the effect of changing the parameter?

**Code Purpose:** Try multiple values ​​for the multiplier. The following function just keeps the error calculation in a reusable section. The command `def` defines the function and the command `return` returns the result.

<!-- cell -->
```python
# Explain this step.
def squared_error(candidate_weight):
    candidate_prediction = candidate_weight * x  # Explain this step.
    error = candidate_prediction - target  # Explain this step.
    return error ** 2  # Explain this step.

# Explain this step.
for candidate_weight in [1.0, 2.0, 2.5, 3.0, 4.0]:
    candidate_prediction = candidate_weight * x  # Explain this step.
    candidate_loss = squared_error(candidate_weight)  # Explain this step.
    print("weight:", candidate_weight, "prediction:", candidate_prediction, "loss:", candidate_loss)
```
<!-- cell -->

You see 16, 4, 1, 0, and 4 errors, respectively. From factor 1 to factor 3, the error decreases. If the coefficient goes higher than 3, the error increases again. Therefore, always increasing the coefficient is not the right answer; The direction of change must depend on the current value.

### See this relationship on the graph

The horizontal axis of the graph is the value of the coefficient and the vertical axis is the squared error. The curve reaches the lowest point at a factor of 3. The `matplotlib` library is the plotting tool and the `pyplot` section provides the plotting commands. The short name `plt` is just a convention name for this section.

**Objective of the code:** See that the error increases by a factor of 3 on both sides. The `linspace` tool here generates 51 equally spaced numbers from 0 to 5. The names of the axes of the diagram are in English so that their display does not depend on the direction of the Persian text.

<!-- cell -->
```python
# Explain this step.
import matplotlib.pyplot as plt

# Explain this step.
candidate_weights = torch.linspace(0.0, 5.0, 51)

# Explain this step.
candidate_losses = (candidate_weights * x - target) ** 2

# Explain this step.
plt.figure(figsize=(6, 3))

# Explain this step.
plt.plot(candidate_weights.tolist(), candidate_losses.tolist(), color="black")

# Explain this step.
plt.xlabel("Weight (w)")
plt.ylabel("Squared error (loss)")

# Explain this step.
plt.grid(True)

# Explain this step.
plt.show()
```
<!-- cell -->

### The gradient shows the direction of error change

Instead of trying all possible values, we can check how a very small change in the parameter changes the error. In this single-parameter example, **Gradient** is the slope of the error with respect to the coefficient; That is, the ratio of the error change to the very small change of the coefficient.

If the gradient is negative, a small increase in the coefficient will reduce the error. If the gradient is positive, a small decrease in the coefficient will reduce the error. The value of the gradient tells how sensitive the error near this point is to the change of the coefficient. This explanation is about small and close changes of the current value; A large jump may increase the error.

In a model with several parameters, such a number is calculated for each parameter, and the set of these numbers forms the gradient. For now, we only have one parameter.

**Code Objective:** Estimate the slope without the derivative formula, with two values ​​close to 1. The expression `epsilon` is the name of the selected small interval.

<!-- cell -->
```python
# Explain this step.
current_weight = 1.0

# Explain this step.
epsilon = 0.01

# Explain this step.
loss_right = squared_error(current_weight + epsilon)

# Explain this step.
loss_left = squared_error(current_weight - epsilon)

# Explain this step.
estimated_gradient = (loss_right - loss_left) / (2 * epsilon)

# Explain this step.
print("loss just below:", loss_left)
print("loss just above:", loss_right)
print("estimated gradient:", round(estimated_gradient, 4))
```
<!-- cell -->

The error of the coefficient of 0.99 is about 16.1604 and the error of the coefficient of 1.01 is about 15.8404. So, close to the coefficient of 1, increasing the error coefficient decreases. Estimated slope is equal to negative 16. The `round` function only limits the number display to four decimal places.

This calculation can be done for a small example. PyTorch has another tool that derives the derivative of the computation from the operation itself. In the next cell, you get the same slope with that tool.

<!-- cell -->

### Automatic derivation: role of forward calculation and `backward`

The `requires_grad=True` option tells PyTorch to track error changes relative to this tensor. This option does not change the tensor automatically and does not start training.

When the prediction and error are calculated, PyTorch keeps the relation of operations: the coefficient and the input are multiplied together, the target is subtracted from the result, and the difference is raised to the power of 2. We call this connection of operations **calculation graph**; Here "graph" means the relation of the calculations, not the graph drawn on the screen.

The calculation from the input and parameter to the prediction and error is called forward pass. The command `loss.backward()` returns from the error to the previous operation and calculates the derivative. This feature is called **automatic derivative or Autograd**. The resulting gradient is saved in the `w.grad` property.

**Objective of the code:** See the calculation of the gradient is different by changing the value of the parameter.

<!-- cell -->
```python
# Explain this step.
x_tensor = torch.tensor(2.0)
target_tensor = torch.tensor(6.0)

# Explain this step.
w = torch.tensor(1.0, requires_grad=True)

# Explain this step.
prediction_tensor = w * x_tensor

# Explain this step.
loss = (prediction_tensor - target_tensor) ** 2

# Explain this step.
print("gradient before backward:", w.grad)

# Explain this step.
loss.backward()

# Explain this step.
print("loss:", loss.item())
print("gradient:", w.grad.item())

# Explain this step.
print("weight after backward:", w.item())
```
<!-- cell -->

Before running `backward`, the gradient value is `None`; It means that the gradient has not been saved yet. After execution, the gradient is negative 16. The value of the parameter remains 1. Therefore, the calculation of the gradient provides only the necessary information to change the parameter.

For this simple relation, the exact derivative formula is `2 * (w * x - target) * x`. By placing the initial values, the result is negative 16. Memorizing this formula is not necessary to continue the lesson; It is important that the result of the automatic derivative is consistent with the estimation of the previous section.

<!-- cell -->

### A learning step, by changing the parameter

To reduce the error, we change the parameter in the opposite direction of the gradient. The rule for this example is: `new_weight = old_weight - learning_rate * gradient`.

The learning rate value controls the step size. In this example, we choose it as 0.1. For a coefficient of 1 and a negative gradient of 16, the new value of the coefficient is 2.6. Calculate this value by hand before running the cell.

When changing the parameter directly, we use `torch.no_grad()`. This command temporarily stops the creation of the derivative path in the part related to the change itself; The gradient calculation has been done before. The parameter change is not part of the prediction calculation of this step.

**Code Purpose:** See a complete step: prediction, error calculation, gradient calculation, parameter change and error remeasurement. The parameter is rebuilt at the beginning of the cell so that the re-execution of the same cell starts from the same point.

<!-- cell -->
```python
# Explain this step.
w_one_step = torch.tensor(1.0, requires_grad=True)

# Explain this step.
learning_rate = 0.1

# Explain this step.
loss_before = (w_one_step * x_tensor - target_tensor) ** 2

# Explain this step.
loss_before.backward()

# Explain this step.
with torch.no_grad():
    # Explain this step.
    w_one_step -= learning_rate * w_one_step.grad
    # Explain this step.
    prediction_after = w_one_step * x_tensor
    loss_after = (prediction_after - target_tensor) ** 2

# Explain this step.
print("loss before:", loss_before.item())
print("new weight:", round(w_one_step.item(), 4))
print("new prediction:", round(prediction_after.item(), 4))
print("loss after:", round(loss_after.item(), 4))
```
<!-- cell -->

The new coefficient is about 2.6 and the new forecast is about 5.2. The desired answer was 6; Therefore, the prediction is closer to the target and the error is reduced to about 0.64. Decimal numbers in a computer do not always have an accurate representation of decimals; Small differences in the last printed digits are normal in this example.

The `w_one_step` parameter has been changed from the example. We have chosen the learning rate ourselves. Such a setting that controls the training method is called **hyperparameter**. Here it is necessary to know the difference between the roles of these two numbers: training changes the coefficient, but the learning rate remains constant.

**Exercise:** Change the learning rate value to 1 and run the same cell again. The new coefficient reaches 17 and the error increases. The direction of change was correct, but the step was too large. Then set the rate back to 0.1 and run the cell.

<!-- cell -->

### Several steps in a row, by erasing the previous gradient

One step brings the parameter closer to the correct answer. Now we repeat the same process five times. At each step, the prediction and error must be calculated with **new parameter value**.

The `backward()` command adds the new gradient to the existing gradient. To have only the gradient of the current step in this example, we erase the previous value at the beginning of each step with `w_train.grad = None`. In other applications, this aggregation may be intentional; We do not intend to do so in this exercise.

**Objective of the code:** See the trend of the coefficient approaching 3 and the prediction to 6. The function `range(5)` returns the numbers zero to four; For readable numbering, we add one unit to the step number at the output.

<!-- cell -->
```python
# Explain this step.
w_train = torch.tensor(1.0, requires_grad=True)

# Explain this step.
learning_rate = 0.1

# Explain this step.
initial_training_loss = ((w_train * x_tensor - target_tensor) ** 2).item()

# Explain this step.
loss_history = []

# Explain this step.
for step in range(5):
    w_train.grad = None  # Explain this step.
    prediction_now = w_train * x_tensor  # Explain this step.
    loss_now = (prediction_now - target_tensor) ** 2  # Explain this step.
    loss_now.backward()  # Explain this step.

    with torch.no_grad():
        w_train -= learning_rate * w_train.grad  # Explain this step.
        new_prediction = w_train * x_tensor  # Explain this step.
        new_loss = (new_prediction - target_tensor) ** 2  # Explain this step.

    loss_history.append(new_loss.item())  # Explain this step.
    # Explain this step.
    # Explain this step.
    print(f"step {step + 1}: weight={w_train.item():.4f}, prediction={new_prediction.item():.4f}, loss={new_loss.item():.6f}")
```
<!-- cell -->

The coefficient after the first step is about 2.6 and after the fifth step is about 2.9994. The final error is less than the initial error. This is learning a parameter from an example.

We chose the multiplication rule ourselves, so reducing the error here does not mean that the program understands language or can answer questions. A language model has different inputs, targets, and parameter counts, and its quality must be checked on separate data. The basic training loop is the same, though: compute an output and its error, calculate gradients, and update the parameters.

### Using the learned parameter

After training, the model can be given input and output without changing the parameters. This work is called **Inference**.

**Code Purpose:** Use the learned multiplier for input 4. In this training relationship, the coefficient is close to 3; So the output should be close to 12. This result comes from the simple relationship we have chosen for the model.

<!-- cell -->
```python
# Explain this step.
new_input = torch.tensor(4.0)

# Explain this step.
with torch.no_grad():
    new_output = w_train * new_input  # Explain this step.

# Explain this step.
print("new input:", new_input.item())
print("output with learned weight:", round(new_output.item(), 4))
```
<!-- cell -->

### The optimizer does the parameter change for us

So far, you have written the command to change the parameter yourself. **Optimizer** is a tool that makes this change with the selected rule. When the model has a large number of parameters, the optimizer can apply the same rule to the list of parameters.

The `torch.optim` section contains optimization tools. The `SGD` method is one of them; Its name is short for "stochastic gradient descent". In training with multiple data, samples or categories are usually selected. This example has only one sample and does not perform random sample selection. For this example and the configuration below, the optimizer does the same subtraction of the learning rate multiplied by the gradient.

Separate the three commands: the `zero_grad()` command erases the previous gradients, the `backward()` command calculates the new gradient, and the `step()` command changes the parameter. The name `step` here is "optimizer step" and is different from the cell run number.

**Code Objective:** Execute the same five steps, this time with the optimizer. The parameter is reconstructed so that comparison with the manual method is possible.

<!-- cell -->
```python
# Explain this step.
w_optimizer = torch.tensor(1.0, requires_grad=True)

# Explain this step.
# Explain this step.
optimizer = torch.optim.SGD([w_optimizer], lr=0.1)

# Explain this step.
for step in range(5):
    optimizer.zero_grad()  # Explain this step.
    prediction_now = w_optimizer * x_tensor  # Explain this step.
    loss_now = (prediction_now - target_tensor) ** 2  # Explain this step.
    loss_now.backward()  # Explain this step.
    optimizer.step()  # Explain this step.

# Explain this step.
print("manual final weight:", round(w_train.item(), 4))
print("optimizer final weight:", round(w_optimizer.item(), 4))

# Explain this step.
print("same result:", torch.allclose(w_train, w_optimizer))
```
<!-- cell -->

Both final coefficients are about 2.9994 and the comparison result is `True`. The optimizer did the part you already wrote yourself. The calculation of prediction and error is still defined in the code.

In the original model, you'll see the same few lines with lots of parameters and text samples. Before that, it is necessary to know exactly how the desired input and response are constructed for a language model. Notebook 01 deals with the same issue.

<!-- cell -->

## 7. Check the end of the lesson

The **`assert`** command checks for a condition. If the condition is true, the execution continues. If the condition is false, the error `AssertionError` is displayed. In this cell, the conditions check only the output of the examples of the same lesson; The skill score or quality of a model is not real.

If you have changed the values ​​of the exercise, first return them to the original values ​​and run the corresponding cells in order. The condition that the coefficient is close to 3 is written with the same five steps and a learning rate of 0.1.

<!-- cell -->
```python
# Explain this step.
assert scalar.ndim == 0 and scalar.numel() == 1

# Explain this step.
assert matrix.shape == (2, 3)

# Explain this step.
assert batch_ids.shape == (2, 3)

# Explain this step.
assert hidden.shape == (2, 3, 2)

# Explain this step.
assert batch_ids.dtype == torch.long

# Explain this step.
assert loss_history[-1] < initial_training_loss

# Explain this step.
assert abs(w_train.item() - 3.0) < 0.01

# Explain this step.
assert torch.allclose(w_train, w_optimizer)

# Explain this step.
print("All lesson checks passed.")
```
<!-- cell -->

### Answer in your own language

1. How many axes does a vector with ten numbers have? What is the difference between the number of numbers and the number of axes?

2. In the `(2, 3, 2)` figure related to the texts in this lesson, what does each size count for?

3. Why do we take the identifier of a letter as an integer, but we take the teachable parameter as a decimal?

4. Executing `backward()` alone, which value does it count and which value does it not change?

5. Why does increasing the learning rate not always lead to better learning?

6. What is the difference between training and getting output from the trained model?

<!-- fa-lint-ignore-next-line -->
<details dir="rtl"><summary>Open suggested answers after your own answer</summary>

1. A decimal vector has one axis with ten positions. The number of axes is determined by the number of dimensions of the figure.

2. We have two texts; Each text has three tokens; Each token is converted into a vector with two numbers.

3. ID is a line number. The parameter is an adjustable value and needs to hold small decimal changes.

4. The gradient is calculated and stored in the `grad` property; The value of the parameter itself remains unchanged. Parameter change is done by manual command or optimizer step.

5. A large step may cross the low error zone and increase the error. In the example of this lesson, rate 1 behaved like this.

6. Training uses target and error to change the parameter. When taking the output, we use the existing parameters and do not change them in this process.

</details>

### Words that you have now seen examples of

| The term | Practical meaning in this lesson
|---|---|
| Tensor | A container of numbers in PyTorch, along with their shape and type information
| Figure or `shape` | The number of positions in each axis
| Type or `dtype` | The method of maintaining the numbers inside the tensor, such as integers or decimals
| Token | the unit into which the text is divided; In this period, a character |
| Embedding Numeric vector corresponding to a token; In our example, | was removed from the table row
| parameter | A value inside the model that changes its training
| Error or `loss` | A number to measure the output difference with the target |
| gradient | Change of error compared to small change of parameter |
| Learning rate A setting to control the size of the parameter change
| Optimizer A tool to change the parameters based on the selected gradient and rule
| Inference Use the available parameters to get the output

### The path after this lesson

Notebook 01 is about predicting the continuation of the text. Notebook 02 examines text splitting and token-to-vector conversion. In Notebook 03, you see a calculation that considers the relevance of text tokens; The name of this mechanism is "attention".

In Notebook 04, the components are put together to make a small model of the GPT family. This family of models produces the continuation of the text step by step. Notebook 05 deals with model training, evaluation and storage. In Notebook 06, the model becomes a service that another program can output from; There is also the way to publish on GitHub and run the service.

Notebook 07 is optional and demonstrates the use of ready-made data and models. The Hugging Face website is a place to publish and receive such data and models. You do not need an account on that website, dataset download, or internet service to run the 00 Notebook examples.

**Next step:** After the examples of this lesson are clear, open the [notebook 01] (01_language_model_basics.ipynb) file.

### Additional resources, to refer to after practice

The examples and explanations of this lesson are written for this educational path. For details on the behavior of PyTorch tools, you can refer to [Tensors Official Guide](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) and [Automatic Derivation Official Guide](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html). Reading these English resources is not a prerequisite for doing the exercises of this course.

<!-- cell -->

## Installation guide for new device

If the environment check cell is run and the python path points to `.venv` of this project, do not run this section. Reinstallation is not required to study this lesson.

The commands in this section are **PowerShell commands and do not run in the Python cell**. Open the PowerShell program in the `tiny-llm-fa` folder. The `uv` tool manages the Python environment and installs the libraries; The following commands assume that this tool is installed on the device.

Versions are fixed for compatibility with the rest of the project. This list does not suggest the latest versions available. You need internet for initial installation. All the calculation examples in this lesson run without internet after installing the libraries.```powershell
# Explain this step.
uv venv --python 3.11 .venv

# Explain this step.
uv pip install --python .venv/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Explain this step.
uv pip install --python .venv/Scripts/python.exe -e ".[notebooks,dev]"

# Explain this step.
.venv/Scripts/python.exe -m ipykernel install --user --name tiny-llm-fa --display-name "Python (tiny-llm-fa)"

# Explain this step.
.venv/Scripts/python.exe -m jupyterlab```The `--python` option in the `uv` commands specifies which Python to use. The `-e` option installs the project in such a way that its code files can be edited in the environment. The `[notebooks,dev]` section is the name of the two dependency groups in the `pyproject.toml` file; Notebook and code review tools are defined in these groups.

The `-m` option tells Python to run an installed utility. `ipykernel` is the Python core for Jupyter. The `--name` option is the internal name of the kernel and the `--display-name` option is the name you see in the kernel selection menu.

If you don't have the `uv` tool but have Python version 3.11 and the `py` executor installed on Windows, the complete replacement of the commands is:```powershell
# Explain this step.
py -3.11 -m venv .venv

# Explain this step.
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Explain this step.
.venv/Scripts/python.exe -m pip install -e ".[notebooks,dev]"

# Explain this step.
.venv/Scripts/python.exe -m ipykernel install --user --name tiny-llm-fa --display-name "Python (tiny-llm-fa)"

# Explain this step.
.venv/Scripts/python.exe -m jupyterlab```The notebooks folder in JupyterLab can be seen as `notebooks`. Open the `00_start_here.ipynb` file and select the `Python (tiny-llm-fa)` kernel. Then start from the display setup cell and the examples at the beginning of this lesson.

If JupyterLab is already open on this device, running the last command again may open a page with a different port. To continue reading, the same active page is enough. Docker is not required to run Notebook 00; Docker is used in the service execution part.
