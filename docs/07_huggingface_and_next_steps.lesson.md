# 07. Understand real data and pretrained models, then choose your next step

So far, you have followed the full path with visible code and project data: text became IDs, a model was built and trained, and it was exposed through a service interface. None of this required downloading a pretrained model or creating a Hugging Face account.

This lesson contains three optional experiments: counting tokens with a ready-made tokenizer, comparing sentences with an embedding model, and training TinyGPT on natural text. First you learn what each tool does, then you can enable the relevant section if you wish. Downloads are disabled by default, so the core sections and exercises still run without internet access.

**Prerequisites:** Lessons 00–06. The optional sections require internet access and additional libraries. If your current goal is to understand the foundations, leave the experiments disabled and read the definitions and small examples.

By the end, you should be able to explain when you need training from scratch, fine-tuning, or document retrieval. This lesson introduces those paths; it does not build a complete fine-tuning or retrieval system.

<!-- cell -->

## 1. Prepare this lesson's independent environment

Each experiment has its own on/off switch. Python's `bool` type has only two values, `True` and `False`; `False` means that the corresponding section is skipped. All three switches are off by default so that opening the lesson never starts a download automatically.

Keep this as the first code cell. The script `scripts/verify_hf.py` locates the first `RUN_REAL_TOKENIZER` statement and creates a temporary copy with only the requested experiment enabled. For a manual run, change only the switch for the experiment you want to `True`.

<!-- cell -->

```python
RUN_REAL_TOKENIZER = False
RUN_REAL_EMBEDDINGS = False
RUN_WIKITEXT = False
```

<!-- cell -->

Every notebook should run after starting a fresh kernel, so we do not assume variables from earlier lessons exist. The next cell locates the project folder and imports the basic tools.

The `ROOT` variable stores the project root. The `src/tinyllm` path is used to identify it. The `os` module configures the working directory and environment variables later, while `json` stores dataset metadata.

<!-- cell -->

```python
from pathlib import Path  # Work with filesystem paths.
import sys  # Add the project source path to Python.
import os  # Configure the working directory and environment variables.
import json  # Store and read structured data.
import math  # Mathematical functions for small examples.
import torch  # Tensor computation and TinyGPT training.

current_folder = Path.cwd()
candidate_folders = [current_folder]
candidate_folders.extend(current_folder.parents)
ROOT = None

for folder in candidate_folders:
    if (folder / "src" / "tinyllm").is_dir():
        ROOT = folder
        break  # The first matching folder is the project root.

if ROOT is None:
    raise FileNotFoundError("Open this notebook from the tiny-llm folder.")

source_path = str(ROOT / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)

torch.set_num_threads(2)  # Two CPU threads are enough for these small computations.
torch.manual_seed(7)  # Make random examples reproducible in this environment.
print("Project folder:", ROOT.name)
```

<!-- cell -->

The output should show the name `tiny-llm`. This cell downloads no external data or pretrained model.

<!-- cell -->

## 2. What does Hugging Face provide?

**Hugging Face** is a place for sharing model and data files and related tools. Think of the **Hub** as a set of repositories for models, datasets, and demo applications. Three repository types matter for this course:

- A **model repository** (`Model`) can contain weights, configuration, a tokenizer, and documentation. Downloading weights alone does not run a model; compatible code must load them.
- A **dataset repository** (`Dataset`) contains data examples and information about their source, structure, and license.
- A **demo environment** (`Space`) is a repository that can also run a demo application. For example, the Docker service from Lesson 06 can run there.

The [official Hub repository guide](https://huggingface.co/docs/hub/repositories) explains this split. A model description is a **model card**, and a dataset description is a **dataset card**. Read these pages before use; they contain language, intended use, limitations, and license information.

Public models and data selected here normally require no account or access token. Creating a personal repository, uploading files, or using restricted files requires an account and appropriate permission. This lesson creates no account or publication; Lesson 06 covers publication.

### How GitHub and Hugging Face relate in this project

GitHub is the right place for the course code, notebooks, setup guide, and experiment reports. If you want to publish model weights separately, a model repository on the Hub can help. If you want an online demo, a Space is another option. You do not need all three for local study.

For a Docker-based Space, set `sdk: docker` and, for this service, `app_port: 7860` in the Space metadata. The application must listen on that port. The [official Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker) documents container-user details; Lesson 06 covers the Docker file and publication guide.

<!-- cell -->

## 3. Install optional libraries only when needed

The project's core libraries are enough to run TinyGPT. The three new experiments use additional libraries. The `pyproject.toml` file groups them under `hf`; writing `.[hf]` means “this project plus the optional dependencies in the `hf` group.”

The project pins the versions in this group; installation does not silently upgrade them to the newest releases. This makes the teaching environment more reproducible.

| Tool | Pinned project version | Use in this lesson |
|---|---|---|
| `tiktoken` | `0.9.0` | Count tokens with a ready-made method |
| `sentence-transformers` | `4.1.0` | Load a sentence-embedding model |
| `datasets` | `3.6.0` | Read WikiText data |
| `huggingface-hub` | `0.32.4` | Communicate with and manage Hub files |
| `transformers` | `4.52.4` | Pretrained-model code; dependency of the embedding path |

To run the online experiment, execute the following command in PowerShell from the `tiny-llm` root. It is a terminal command, not Python code.

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[hf]"
```

If `uv` is not installed, use this environment's Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hf]"
```

The `-e` option installs the project in editable mode, so the `src` code remains the code you see in the project. Restart the notebook kernel after installation and rerun cells from the top. Installing the group does not enable downloads; later cells control downloads.

<!-- cell -->

# Explain the corresponding operation in this teaching example.

`importlib.metadata` reads installed package metadata. This check imports no model or dataset and makes no network request. If the `hf` group is absent, you see `not installed` instead of a stopping error.

<!-- cell -->

```python
from importlib.metadata import version, PackageNotFoundError

optional_packages = [
    "tiktoken", "sentence-transformers", "datasets",
    "huggingface-hub", "transformers",
]

for package_name in optional_packages:
    try:
        installed_version = version(package_name)
    except PackageNotFoundError:
        installed_version = "not installed"

    print(package_name, ":", installed_version)
```

<!-- cell -->

If versions match the table, the recommended environment is available. Without the optional tools, the foundational examples and planning section still run.

<!-- cell -->

## 4. Where are downloaded files stored?

Download tools usually keep files in a **file cache** so later runs do not download everything again. This is different from a transformer `KV cache`: here we mean files on disk, not temporary token computations in memory.

To keep the course organized, these files go under `data/external`. This folder is excluded from the default Git publication and course bundle. The original `data/documents.json` is not changed.

An **environment variable** is a named setting that a program and its libraries can read. For example, `HF_HOME` identifies the location for Hub files. We set these values before importing Hugging Face libraries so they use the selected location.

On Windows, the current project folder has a deep path, and some dataset lock files become long. This project uses the short relative path `data/external/ds` and sets the working directory to the root. This reduces path length, but it cannot guarantee that every long-path error disappears on every machine.

<!-- cell -->

```python
# Interpret relative paths from the project root from this point onward.
os.chdir(ROOT)

# Use an absolute path for model and Hub caches.
os.environ["HF_HOME"] = str(ROOT / "data" / "external" / "hf-cache")

# Use a short relative dataset path for this Windows project.
os.environ["HF_DATASETS_CACHE"] = "data/external/ds"

# tiktoken uses a separate cache for tokenizer files.
os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / "data" / "external" / "tiktoken-cache")

print("working folder:", Path.cwd().name)
print("dataset cache:", os.environ["HF_DATASETS_CACHE"])
```

<!-- cell -->

The working folder is `tiny-llm`, and the dataset path is the configured relative path. This cell only configures paths; no download has started.

## 5. Count tokens with a ready-made tokenizer

Lesson 02 showed that tokenization is a contract between text and IDs. **tiktoken** provides a ready-made tokenizer from the BPE family. In BPE, frequent pieces can receive one ID, so a token is not necessarily one character. The official [tiktoken repository](https://github.com/openai/tiktoken) documents the tool and the `cl100k_base` example.

In this example, we load the named `cl100k_base` encoding for comparison. This does not turn TinyGPT into another model; TinyGPT still uses the course's character tokenizer.

**Code goal:** Compare character and token counts for several texts and verify that IDs decode back to the original text. If the switch is off, the cell only prints a skipped message.

<!-- cell -->

```python
if RUN_REAL_TOKENIZER:
    import tiktoken  # Needed only when this experiment is enabled.

    # Encoding files may be downloaded on the first run.
    encoding = tiktoken.get_encoding("cl100k_base")
    token_examples = [
        "The model learns patterns.",
        "مدل الگوها را می‌آموزد.",
        "word", " word",
        "123456789", '{"answer":42}',
        "ی", "ي",
    ]

    for text in token_examples:
        token_ids = encoding.encode(text)
        recovered_text = encoding.decode(token_ids)

        print("text:", repr(text))
        print("characters:", len(text), "tokens:", len(token_ids), "IDs:", token_ids)

        # This tokenizer must decode its own IDs back to the original text.
        assert recovered_text == text
else:
    print("Real tokenizer experiment skipped: RUN_REAL_TOKENIZER is False.")
```

<!-- cell -->

In an enabled run, notice the difference between `word` and ` word`: a leading space can affect segmentation. The two forms of the Persian letter yeh also have different Unicode code points and may receive different IDs. These examples show real behavior; do not assume a fixed token count in advance.

The ratio of Persian to English token counts is not universal. Both the text and tokenizer matter. To calculate a service budget, use the tokenizer of the target model.

Do not pass `cl100k_base` IDs directly to TinyGPT's embedding table. The same number does not necessarily mean the same token in two vocabularies, and some IDs may exceed the model table size. Changing a tokenizer usually requires a compatible vocabulary, model configuration, and training run.

<!-- cell -->

## 6. Distinguish sentence embeddings from a model's internal embeddings

In GPT, the embedding table converted a token ID into a vector. That vector was an input to the model's computation. For document search, we usually need a different vector: a numeric representation of a whole sentence or text chunk, designed so related texts are close when their vectors are compared.

We call this representation a **sentence embedding**. A vector is a list of numbers; the meaning of a sentence is not stored in one named slot such as “publishing permission.” Its behavior comes from model training and from how it is compared with other vectors.

The pretrained **paraphrase-multilingual-MiniLM-L12-v2** model is designed to produce these vectors. According to its [official model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), each text produces 384 numbers. This lesson also reads the actual shape and length limit from the model. It creates text embeddings; it does not replace TinyGPT as a response generator.

### First, see vector similarity with two numbers

A vector length is the square root of the sum of squared values. The length of `[3, 4]` is 5. **Normalization** divides by that length, preserving direction and making the length one.

**Cosine similarity** compares the direction of two vectors. For normalized vectors, multiply and sum corresponding values; this is the `dot product`. One means the same direction, zero means perpendicular directions, and negative one means opposite directions. These are vector properties, not percentages of sentence correctness.

**Example goal:** Run this calculation on synthetic vectors so the comparison method is clear before downloading anything.

<!-- cell -->

```python
# These vectors are synthetic; they do not come from text or a trained model.
toy_vectors = torch.tensor([
    [3.0, 4.0],
    [6.0, 8.0],
    [-4.0, 3.0],
])

# Sum squares in each row and take the square root; the result has one column.
toy_lengths = torch.sqrt((toy_vectors ** 2).sum(dim=1, keepdim=True))
toy_normalized = toy_vectors / toy_lengths

# Compare every row with every row; T swaps rows and columns.
toy_similarities = toy_normalized @ toy_normalized.T

print("lengths:", toy_lengths.squeeze(1).tolist())
print("normalized vectors:\n", toy_normalized)
print("cosine similarities:\n", toy_similarities)
```

<!-- cell -->

The lengths are 5, 10, and 5. The first two vectors point in the same direction, so their similarity is one. The third is perpendicular to both, so its similarity is zero. The same operation is applied to 384-dimensional text vectors; semantic quality depends on model training.

# Explain the corresponding operation in this teaching example.

`SentenceTransformer` reads model files from the named repository. `device="cpu"` selects the CPU, and `normalize_embeddings=True` returns unit-length vectors.

Two sentences about publication permission differ only by negation, plus one billing sentence. We compare them with “publish the service.” An embedding alone does not implement permission logic.

<!-- cell -->

```python
if RUN_REAL_EMBEDDINGS:
    from sentence_transformers import SentenceTransformer

    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

    # Explain the corresponding operation in this teaching example.
    embedding_sentences = [
        "انتشار سرویس مجاز است.",
        "انتشار سرویس مجاز نیست.",
        "صورتحساب مشتری را بررسی کن.",
        "سرویس را منتشر کن.",
    ]

    # Explain the corresponding operation in this teaching example.
    vectors = embedding_model.encode(
        embedding_sentences,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    print("embedding table shape:", tuple(vectors.shape))
    print("embedding width:", vectors.shape[1])
    print("model max_seq_length:", embedding_model.max_seq_length)

    # Explain the corresponding operation in this teaching example.
    lengths = torch.sqrt((vectors ** 2).sum(dim=1))
    print("vector lengths:", lengths.tolist())
else:
    print("Embedding download skipped: RUN_REAL_EMBEDDINGS is False.")
```

<!-- cell -->

When enabled, the expected shape is `(4, 384)`: four texts and 384 features per vector. `max_seq_length` is this embedding model's input limit, not TinyGPT's 64-character context. Long documents can exceed it and need separate chunking.

This experiment reads the repository's current model revision, so results can change if files are replaced. For research or production, record the exact model revision and library versions. The data section demonstrates selecting a fixed revision.

# Explain the corresponding operation in this teaching example.

The first three rows are documents and the fourth is the query. Compare the query only with document vectors; including it among its own candidates makes retrieval artificially easy.

**Code goal:** Compute three scores and rank documents from highest to lowest similarity.

<!-- cell -->

```python
if RUN_REAL_EMBEDDINGS:
    document_vectors = vectors[:3]  # Keep only the three documents, not the query.
    query_vector = vectors[3]  # The query row.

    # Explain the corresponding operation in this teaching example.
    document_scores = document_vectors @ query_vector

    # Explain the corresponding operation in this teaching example.
    ranked_document_ids = torch.argsort(document_scores, descending=True)

    for rank_index, document_id_tensor in enumerate(ranked_document_ids):
        document_id = document_id_tensor.item()
        score = document_scores[document_id].item()
        print(
            "rank:", rank_index + 1,
            "score:", round(score, 4),
            "document:", embedding_sentences[document_id],
        )
else:
    print("Embedding comparison skipped.")
```

<!-- cell -->

Read the order from the real output; do not assume the permitted sentence ranks above the prohibited one. The negative sentence shares words and topic, so it may score highly. Semantic similarity does not prove permission.

In real use, numbers, negation, time, and document version matter. A “you are not allowed” document may be the relevant source for a permission question even though the final answer must obey the prohibition. Ranking and content decisions are separate tasks.

<!-- cell -->

# Explain the corresponding operation in this teaching example.

**Retrieval** finds relevant documents for a query. One similarity table is not enough; define multiple queries and relevant documents in advance.

**Recall@k** asks what fraction of relevant documents appear in the first `k` results. `k` is the number of retained results. With one relevant document per query, each query scores zero or one; then average across queries.

**MRR** also considers the rank of the first relevant document. `Mean Reciprocal Rank` gives ranks one, two, and three scores of one, one-half, and one-third; no relevant result scores zero, then queries are averaged.

**Example goal:** Compute these metrics on synthetic rankings. No model or download is used; inspect the relevant ranks by hand.

<!-- cell -->

```python
# Explain the corresponding operation in this teaching example.
ranked_examples = [
    ["A", "B", "C"],  # Relevant document A is ranked first.
    ["A", "B", "C"],  # Relevant document B is ranked second.
    ["A", "B", "C"],  # Relevant document C is ranked third.
]

# Explain the corresponding operation in this teaching example.
relevant_examples = [ {"A"}, {"B"}, {"C"} ]
k = 2
recall_values = []
reciprocal_ranks = []

for question_index in range(len(ranked_examples)):
    ranked_ids = ranked_examples[question_index]
    relevant_ids = relevant_examples[question_index]

    # Explain the corresponding operation in this teaching example.
    top_ids = set(ranked_ids[:k])
    found_relevant = top_ids.intersection(relevant_ids)
    recall = len(found_relevant) / len(relevant_ids)
    recall_values.append(recall)

    # Explain the corresponding operation in this teaching example.
    reciprocal_rank = 0.0
    for zero_based_rank, document_id in enumerate(ranked_ids):
        if document_id in relevant_ids:
            reciprocal_rank = 1.0 / (zero_based_rank + 1)
            break

    reciprocal_ranks.append(reciprocal_rank)

mean_recall = sum(recall_values) / len(recall_values)
mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("Recall@2 by question:", recall_values)
print("mean Recall@2:", round(mean_recall, 4))
print("reciprocal ranks:", reciprocal_ranks)
print("MRR:", round(mean_reciprocal_rank, 4))
```

<!-- cell -->

Recall values are `[1.0, 1.0, 0.0]` with a mean of about 0.6667. Reciprocal ranks are one, one-half, and one-third, averaging about 0.6111.

These values do not measure MiniLM; they come from synthetic ranks. A real evaluation applies this logic to labeled queries and documents and states the Recall `k` and MRR list. It does not alone measure generated-answer correctness.

**Short exercise:** Change `k` to 3. Mean Recall becomes one, but MRR stays unchanged because it uses the full three-item list. Set `k` back to 2.

<!-- cell -->

## 8. Why do we want natural data for the second experiment?

The main course data contains 150 short synthetic documents: 114 training, 18 validation, and 18 test documents. It is useful for understanding the program and seeing local pattern learning. Because templates and sentences overlap, good results do not demonstrate general ability on natural text.

**WikiText** is an English language-modeling dataset with training, validation, and test splits. This lesson uses `wikitext-2-raw-v1`. A configuration identifies a specific repository version or subset; it is not the same as a larger version or one with unknown replacements. The [official Salesforce/wikitext card](https://huggingface.co/datasets/Salesforce/wikitext) describes its splits and licenses.

This is not a Persian dataset and will not produce a useful Persian model. Here it demonstrates connecting external data to the same TinyGPT code. The raw version is better for character-tokenization experiments than one that replaces words with `<unk>` because you see the text characters directly.

The dataset card lists `CC BY-SA 3.0` and `GFDL`; these differ from the course code's MIT license. Do not apply the code license to external data. When redistributing data or derivatives, check and record its source and license terms. This lesson stores downloads under `data/external`, excluded from the default publication.

<!-- cell -->

## 9. Download a precise split, revision, and amount

Repository name, configuration, and split each have a separate meaning:

- **Repository name** `Salesforce/wikitext` identifies the data location.
- **Configuration name** `wikitext-2-raw-v1` identifies the requested dataset variant.
- **Data split** uses `train[:5000]` and `validation[:1000]`: the first 5,000 training and first 1,000 validation rows.
- A **fixed revision** is a specific repository-history identifier. We use it instead of moving `main` so the dataset cannot change silently.

A row is not a complete document or a token. It may be a title, blank line, or text fragment, and a slice can cut an article in the middle. Selecting a slice does not guarantee that only those rows are downloaded; the library may read larger files to prepare them. This is not a full WikiText evaluation.

We do not read the test split to choose settings. The loading code requests only training and validation. The project-pinned [Datasets loading guide](https://huggingface.co/docs/datasets/v3.6.0/en/loading) explains configurations, revisions, and slices.

**Code goal:** Load the data with this precise contract and inspect the row counts.

<!-- cell -->

```python
if RUN_WIKITEXT:
    import datasets
    from datasets import load_dataset

    # A fixed data-repository revision; it does not move with main.
    revision = "b08601e04326c79dfdd32d625aee71d232d685c3"
    wiki_slices = {
        "train": "train[:5000]",
        "validation": "validation[:1000]",
    }

    wiki = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        revision=revision,
        cache_dir="data/external/ds",  # Short relative path after setting the working directory.
        split=wiki_slices,
    )

    for split_name, rows in wiki.items():
        print(split_name, "rows:", len(rows), "columns:", rows.column_names)
else:
    print("WikiText download skipped: RUN_WIKITEXT is False.")
```

<!-- cell -->

When enabled, you receive 5,000 training rows and 1,000 validation rows. Each split has a `text` column. `wiki` is a dictionary of both splits; the next cell writes non-empty text to files.

### Store external data in a separate folder

`mkdir` creates a folder. `parents=True` creates required parent folders, and `exist_ok=True` accepts an existing folder. External files go under `data/external/wikitext2` here.

This example removes empty rows and joins the rest with newlines. Removing blank rows changes the data representation, so record it in the report. To reproduce an official benchmark, follow its expected preprocessing.

<!-- cell -->

```python
if RUN_WIKITEXT:
    external = ROOT / "data" / "external" / "wikitext2"
    external.mkdir(parents=True, exist_ok=True)

    for split_name, rows in wiki.items():
        nonempty_texts = []
        for row in rows:
            row_text = row["text"]
            if row_text.strip():  # Use strip only to detect an empty row.
                nonempty_texts.append(row_text)  # Keep the original text without stripping.

        joined_text = "\n".join(nonempty_texts)
        output_file = external / f"{split_name}.txt"
        output_file.write_text(joined_text, encoding="utf-8")

        print(split_name, "nonempty rows:", len(nonempty_texts), "characters:", len(joined_text))
else:
    print("External text files were not written.")
```

<!-- cell -->

`train.txt` and `validation.txt` are created only when enabled. The original `data/documents.json` remains unchanged. Read character counts from the actual output; they are not row counts.

### Record data provenance alongside the files

A metadata file or `manifest` records where data came from, which revision was used, and how it was prepared. In Datasets, a **fingerprint** identifies the dataset state and transformations; we record it for traceability. It does not replace source and license documentation.

**Code goal:** Save experiment-tracking information beside the text files. This file is not uploaded to your repository unless you explicitly choose it during publication.

<!-- cell -->

```python
if RUN_WIKITEXT:
    fingerprints = {}
    for split_name, rows in wiki.items():
        # This field records state in the project's pinned datasets version.
        fingerprints[split_name] = rows._fingerprint

    manifest = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "revision": revision,
        "datasets_version": datasets.__version__,
        "slices": wiki_slices,
        "fingerprints": fingerprints,
        "preparation": "keep nonempty text rows; join original rows with newline",
        "license_card": ["cc-by-sa-3.0", "gfdl"],
        "source": "https://huggingface.co/datasets/Salesforce/wikitext",
    }

    manifest_path = external / "manifest.json"
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    print("manifest file:", manifest_path.relative_to(ROOT))
    print("revision:", manifest["revision"])
else:
    print("External manifest was not written.")
```

<!-- cell -->

When enabled, the manifest is saved in the same external folder. If you report the experiment, also record slice sizes and the empty-row policy. A fixed revision prevents silent dataset changes, but cannot by itself guarantee identical computations on every machine.

<!-- cell -->

# Explain the corresponding operation in this teaching example.

Using new data does not require rewriting the architecture. Keep the same contract: text becomes IDs, inputs and targets are shifted by one position, and the model computes next-token loss.

The vocabulary must match the data. Here you build a new vocabulary only from WikiText training text. A character ID in this vocabulary may differ from the same character in the course vocabulary, so you do not attach the old weights to this tokenizer; you create a new model with the new `vocab` setting.

Validation text may contain unseen characters. `strict=False` maps them to ID zero, the unknown token. We report its rate so the evaluation does not hide lost characters.

**Code goal:** Read the saved text and convert it into two tensors using the correct ID contract.

<!-- cell -->

```python
if RUN_WIKITEXT:
    from tinyllm.data import CharTokenizer, get_batch
    from tinyllm.model import Config, TinyGPT

    train_text = (external / "train.txt").read_text(encoding="utf-8")
    val_text = (external / "validation.txt").read_text(encoding="utf-8")

    # Build the vocabulary from training text only.
    wiki_tokenizer = CharTokenizer.fit(train_text)
    wiki_train_ids = wiki_tokenizer.encode(train_text)
    wiki_val_ids = wiki_tokenizer.encode(val_text, strict=False)

    # long is the integer type expected by the embedding table for IDs.
    wiki_train = torch.tensor(wiki_train_ids, dtype=torch.long)
    wiki_val = torch.tensor(wiki_val_ids, dtype=torch.long)

    unknown_fraction = (wiki_val == 0).float().mean().item()
    print("new vocabulary size:", wiki_tokenizer.vocab_size)
    print("training tensor shape:", tuple(wiki_train.shape))
    print("validation unknown fraction:", unknown_fraction)
else:
    print("WikiText token tensors were not created.")
```

<!-- cell -->

The training tensor is a long one-dimensional list of IDs; batches with shape `(B, T)` have not been created yet. `(wiki_val == 0)` produces a Boolean value for each position. Converting it to `float` gives ones and zeros, and the mean is the fraction of unknown IDs.

Do not confuse this rate with the loss. Unknown IDs may be rare while the model predictions are still poor.

### Build a fresh model and one input batch

This TinyGPT uses a context length of 64, embedding width 64, four heads, and two blocks; only the vocabulary comes from the new data. A batch of 16 means 16 text chunks per computation, each with 64 input and 64 target IDs.

The `get_batch` targets are shifted one position forward: each input must predict its next token. A local random generator chooses the starting positions.

<!-- cell -->

```python
if RUN_WIKITEXT:
    # Make the fresh model's random initialization reproducible.
    torch.manual_seed(8)
    wiki_config = Config(vocab=wiki_tokenizer.vocab_size)
    wiki_model = TinyGPT(wiki_config)

    # The optimizer will update this fresh model's weights.
    wiki_optimizer = torch.optim.AdamW(wiki_model.parameters(), lr=0.002)
    wiki_rng = torch.Generator().manual_seed(8)

    sample_x, sample_y = get_batch(wiki_train, wiki_config.context, 16, wiki_rng)
    print("input shape:", tuple(sample_x.shape))
    print("target shape:", tuple(sample_y.shape))
    print("shifted positions match:", torch.equal(sample_x[:, 1:], sample_y[:, :-1]))

    assert torch.equal(sample_x[:, 1:], sample_y[:, :-1])
else:
    print("WikiText model creation skipped.")
```

<!-- cell -->

The input and target shapes are `(16, 64)`, and the shift check is `True`. This cell creates a fresh model; it does not use the previous Persian weight file.

### Twenty steps to verify the connection, not to claim quality

The next loop follows Lesson 05: choose a batch, compute loss, clear old gradients, compute gradients, clip their size, and update weights. A **gradient** measures how a weight change affects loss; the **optimizer** uses it to update weights.

Only 20 steps run here. That is enough to observe the execution path, but not enough to build a high-quality language model on natural text.

<!-- cell -->

```python
if RUN_WIKITEXT:
    wiki_model.train()  # Enable training behavior, including dropout.
    wiki_training_losses = []

    for step in range(20):
        x, y = get_batch(wiki_train, wiki_config.context, 16, wiki_rng)

        # The second output is loss for this batch's targets.
        logits, loss = wiki_model(x, y)

        wiki_optimizer.zero_grad(set_to_none=True)  # Clear the previous step's gradients.
        loss.backward()  # Compute gradients for the model weights.

        # Clip the total gradient size to reduce unusually large updates.
        torch.nn.utils.clip_grad_norm_(wiki_model.parameters(), 1.0)
        wiki_optimizer.step()  # Update the weights here.

        wiki_training_losses.append(loss.item())
        if step in [0, 9, 19]:
            print("step:", step + 1, "training batch loss:", round(loss.item(), 4))

    print("completed steps:", len(wiki_training_losses))
else:
    print("WikiText 20-step training skipped.")
```

<!-- cell -->

The final number is the loss of the random batch at the last training step. Batches and dropout change, so loss need not decrease every step. Do not treat this number as a final quality evaluation.

The next cell reads one validation batch without changing weights. For a serious report, add several fixed batches, a plot, best-weight saving, and a bigram comparison as in Lesson 05.

<!-- cell -->

```python
if RUN_WIKITEXT:
    wiki_model.eval()  # Disable dropout during evaluation.
    wiki_val_rng = torch.Generator().manual_seed(9)
    val_x, val_y = get_batch(wiki_val, wiki_config.context, 16, wiki_val_rng)

    # no_grad avoids building a gradient path; weights do not change.
    with torch.no_grad():
        val_logits, val_loss = wiki_model(val_x, val_y)

    print("one validation batch loss:", round(val_loss.item(), 4))
    print("validation unknown fraction:", round(unknown_fraction, 6))
else:
    print("WikiText validation demonstration skipped.")
```

<!-- cell -->

One validation batch is not enough to conclude general quality. This run only shows that external data connects to the same model contract. Do not compare its loss directly with the Persian synthetic-data model or a multi-character tokenizer model; the data and prediction units differ.

At the end of this experiment, no original course weights or data files were replaced. To save and serve this external model, package its weights, configuration, and **this new tokenizer** together as an independent artifact; Lessons 05 and 06 describe the general method.

<!-- cell -->

## 11. Training from scratch versus continued training

**Training from scratch** starts with random weights and learns from that point. The original TinyGPT and this lesson’s WikiText model both use next-token prediction.

**Pretraining** is learning general text structure from a large corpus. This course uses the same objective at a small scale; its data volume and diversity are not comparable to production pretraining.

**Continued pretraining** starts from previous weights and reads more text with the same objective. A base model may continue on permitted, domain-specific technical text. Tokenizer, objective, and data compatibility matter; new data alone is not enough.

With a fixed tokenizer, you can start from previous weights. If ID meanings change, the old embedding table no longer follows the same contract. The WikiText experiment intentionally creates a fresh model because it builds a new vocabulary; it is not continued training of the earlier Persian model.

<!-- cell -->

## 12. Supervised fine-tuning with one example

**Fine-tuning** trains an existing model further with a specific dataset and objective. **Supervised fine-tuning** (`SFT`) uses examples containing a prompt and desired answer. For example, “Describe the service status in one sentence” with “The service is available” is one training example.

In a generative model, prompt and answer enter together so the prompt remains in context while the answer is predicted. Mark their boundary with a stable format or special tokens compatible with the model.

Sometimes loss should apply only to answer tokens. This is a **loss mask** and differs from the causal attention mask: the causal mask controls visibility, while the loss mask controls which targets contribute to training loss.

# Explain the corresponding operation in this teaching example.

The following IDs are synthetic. In `[10, 11, 12, 21, 22, 23]`, the first three IDs are the prompt and boundary, and the final three are the answer. Shifting by one position creates five inputs and five targets.

Targets 11 and 12 still belong to the prompt, so exclude them from loss. Target 21 is the first answer token and remains in loss. In PyTorch cross-entropy, `-100` with `ignore_index` means “ignore this target.”

**Code goal:** Inspect only the mask contract. Scores are random; this cell does not perform real fine-tuning.

<!-- cell -->

```python
import torch.nn.functional as F

# These IDs do not belong to the course tokenizer; they are only a small example.
combined_ids = torch.tensor([10, 11, 12, 21, 22, 23], dtype=torch.long)
prompt_length = 3

sft_inputs = combined_ids[:-1]
sft_targets = combined_ids[1:].clone()  # Copy so the original sequence is unchanged.

# Explain the corresponding operation in this teaching example.
sft_targets[:prompt_length - 1] = -100

# Explain the corresponding operation in this teaching example.
toy_logits = torch.randn(5, 32)
answer_only_loss = F.cross_entropy(toy_logits, sft_targets, ignore_index=-100)

print("inputs:", sft_inputs.tolist())
print("masked targets:", sft_targets.tolist())
print("answer target count:", (sft_targets != -100).sum().item())
print("demonstration loss:", answer_only_loss.item())

assert sft_targets.tolist() == [-100, -100, 21, 22, 23]
```

<!-- cell -->

The final targets are `[-100, -100, 21, 22, 23]`, so three answer targets contribute to loss. The prompt remains in the input; excluding it from loss does not remove it from context.

<!-- fa-lint-ignore-next-line -->
This example runs no gradient path or optimizer, so no weights are trained. Real SFT needs a suitable model, conversation format, answer data, batching method, and response evaluation. The current `TinyGPT.forward` averages loss over all targets by default; coordinate target preparation and the mask contract deliberately for this use.

<!-- cell -->

## 13. Why does LoRA train fewer values?

With full fine-tuning, you may update every model weight. For a large model, storing gradients and optimizer state is expensive. **Parameter-efficient fine-tuning** (`PEFT`) trains only a small subset of values; PEFT is both a method family and a library.

One such method is **LoRA**. For selected weight matrices, the base weight stays frozen and an update from two smaller matrices is added. Instead of updating every cell of a large weight, you train two small tables. See the [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora).

**Low rank** means the update path has a small intermediate dimension. The next example shows only multiplication shapes; it does not train or implement the full LoRA library.

<!-- cell -->

```python
# Explain the corresponding operation in this teaching example.
base_weight = torch.zeros(3, 4)

# Explain the corresponding operation in this teaching example.
small_B = torch.tensor([[1.0], [2.0], [3.0]])  # Shape (3, 1).
small_A = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # Shape (1, 4).

delta_weight = small_B @ small_A  # Resulting shape is (3, 4).
effective_weight = base_weight + delta_weight

print("base weight shape:", tuple(base_weight.shape))
print("small matrix shapes:", tuple(small_B.shape), tuple(small_A.shape))
print("change shape:", tuple(delta_weight.shape))
print("base values:", base_weight.numel())
print("small-matrix values:", small_B.numel() + small_A.numel())
```

<!-- cell -->

The base weight has 12 values, while the two small matrices have 7. Real savings depend on layer size and rank; the base weight and model computation are still required, so LoRA does not eliminate total model memory.

For the next practical step, choose a small pretrained model compatible with HF and PEFT. Installing PEFT alone does not turn the current teaching model into a strong assistant. Capacity, language, tokenizer format, and data quality still matter; record library versions and experiment settings.

<!-- cell -->

## 14. How does retrieval differ from training weights?

Suppose the answer to “How many annual-leave days are available?” is in the latest internal company document. Making that document available does not necessarily require retraining the model on it. The program can find the relevant document and place it in the model context with the query.

This method is **retrieval-augmented generation** (`RAG`). The simple path is: split documents into chunks, create vectors, retrieve relevant chunks with the query vector, then pass chunks and query to the answering model. Embedding and retrieval do not change the answering model weights.

**Chunking** divides a long document into usable pieces. An **index** stores vectors and chunk IDs for search. A small example can compare every vector; a larger collection needs an index to manage search.

**Grounding** means that an answer claim can be traced to information in retrieved chunks. Good retrieval is necessary but not sufficient; the model must respect negation, versions, and document numbers. Evaluate the final answer separately from document ranking.

### See only the connection from a retrieved document to context

The next example selects a document in advance. It runs no embedding, index, or answer model; it shows that RAG passes document information from the application into model input rather than changing weights during the request.

<!-- cell -->

```python
# This synthetic document was selected in advance; it is not a real search result.
retrieved_document = {
    "id": "policy-demo-v2",
    "text": "در این مثال فرضی، سقف مرخصی سالانه ۲۰ روز است.",
}
question = "در سند مثال، سقف مرخصی سالانه چند روز است؟"

# Put the query and document into one text input with clear boundaries.
rag_context = (
    "فقط بر اساس سند زیر پاسخ بده. اگر پاسخ در سند نبود، نبودن اطلاعات را بیان کن.\n"
    + "شناسهٔ سند: " + retrieved_document["id"] + "\n"
    + "متن سند: " + retrieved_document["text"] + "\n"
    + "پرسش: " + question
)

print(rag_context)
print("context character count:", len(rag_context))
```

<!-- cell -->

The output contains the instruction, document ID, document content, and query. This cell does not generate an answer or guarantee instruction following. Clear instructions are part of input design; model quality and behavioral evaluation are also required.

This context easily exceeds 64 characters. The current character-level TinyGPT is not suitable for real answers over such a document; with a sliding window, older input leaves the context. For RAG, keep the application architecture and service contract, then choose a suitable answering model and retrieval tool.

<!-- cell -->

## 15. Which project parts survive a model swap?

Lesson 06 defined a clear boundary between application and model: it accepted text and generation settings and returned a structured response. This boundary lets you replace the model implementation while keeping parts of the application.

We call the internal model implementation the **model backend**. Here it is TinyGPT, the character tokenizer, and its weight file; a future version may use a pretrained model and its tokenizer.

| Next path | Reusable parts | Changes needed for the same goal |
|---|---|---|
| Continued text training | Next-token objective, training loop, and evaluation | Compatible data and before/after report with a fixed tokenizer |
| Prompt/answer fine-tuning | Causal block and loss computation | Conversation format, answer data, and loss-mask contract |
| LoRA fine-tuning | Evaluation and model-output handling | Compatible base model, PEFT tools, layer and rank settings |
| Document retrieval | Request/response boundary and part of the application logic | Chunking, separate embeddings, index, and document/answer evaluation |
| Application service | Health checks, request validation, and service startup | Task-appropriate model, access control, monitoring, and capacity measurement |

Keeping the service boundary does not make every limitation identical. A new model may use a different tokenizer, request format, input length, or memory budget; validation and error contracts must be updated accordingly.

Increasing TinyGPT’s context length alone is not enough. Position tables, weight files, and training experience for different lengths must match the new choice. This small model is useful for learning because you can read almost all its code; production model selection is a separate decision based on language, resources, and task evaluation.

<!-- cell -->

## 16. Complete a project report with evidence

For an interview, a service link alone does not show what you understand. A short report of the decision and result of a controlled experiment is stronger evidence. A **controlled experiment** changes one declared setting while keeping comparison conditions as fixed as possible.

For example, keep the tokenizer and data fixed, change the number of training steps, and compare validation loss, runtime, and a generated sample using the same method. If you change data, tokenizer, and model size together, you cannot attribute the result to one cause.

**Cell goal:** Create a report template without inventing results. This template writes no files and publishes nothing.

<!-- cell -->

```python
experiment_report = {
    "question": "What changes when training steps increase with fixed data and tokenizer?",
    "changed_setting": "training_steps",
    "fixed_items": ["data split", "tokenizer", "model config", "evaluation batches"],
    "before": None,  # Fill this after a real measurement.
    "after": None,
    "measured_seconds": None,
    "limitations": "Synthetic template overlap; small model; character-level context.",
}

print(json.dumps(experiment_report, indent=2, ensure_ascii=False))
```

<!-- cell -->

`null` values in JSON correspond to Python `None`: no result has been entered yet. Fill them only with measured results. Keep the synthetic-data and small-model limitations in the report so readers understand the actual scope.

For GitHub publication, keep notebooks, code, setup notes, the code license, and valid reports in the `tiny-llm` folder. Download caches and WikiText are excluded from the default bundle. A Hub account is needed to create a personal model repository or Space, not to read and run the local path.

<!-- cell -->

## 17. Review and interview questions

1. How do model repositories, dataset repositories, and Spaces differ? Which account is needed for local study?

2. Why should `cl100k_base` IDs not be passed to TinyGPT’s embedding table?

3. How does GPT token embedding differ from a sentence embedding used for search?

4. If “permitted” and “not permitted” sentences are highly similar, has the model granted publication permission?

5. Why were Recall@2 about 0.667 and MRR about 0.611 in the three-query example?

6. Why record both the WikiText revision and its preprocessing?

7. Why are 20 steps not a final quality metric, and why should their loss not be compared directly with the earlier Persian model?

8. In SFT, why does the prompt remain in the input while some of its targets are removed from loss?

9. Which paths change weights: prompting, SFT, LoRA, or retrieval during a request?

10. After replacing the answering model, which service parts can remain and which limits must be checked again?

<!-- fa-lint-ignore-next-line -->
<details><summary>Read the suggested answers after answering yourself</summary>

**Answer 1:** A model repository stores model files, a dataset repository stores dataset examples, and a Space runs a demo application. Local study needs no Hub account; personal Hub publication requires an account and suitable access.

**Answer 2:** An ID is an index in one tokenizer's vocabulary. The same number can represent different text pieces in another vocabulary or exceed the model table size. Model and tokenizer must be selected as one compatible contract.

**Answer 3:** GPT's internal token embedding is an input to token computation. A sentence embedding represents a whole text for query/document comparison. Averaging a random table does not automatically create a reliable semantic embedding.

**Answer 4:** No. Similarity shows that topic or sentence structure is close; permission and negation require content inspection. A prohibited sentence can still be the relevant document for a permission question.

**Answer 5:** Two relevant documents appeared in the first two results and the third did not, so Recall averaged one, one, and zero: two-thirds. The first relevant ranks were one, two, and three, so MRR averaged one, one-half, and one-third: about 0.611.

**Answer 6:** The revision identifies which repository files were read. Preprocessing identifies the final text constructed from them; removing blank rows and slicing rows can change the model input.

**Answer 7:** Only a few batches were read to verify the connection. Quality evaluation needs more data and examples. WikiText data and vocabulary differ from the earlier model, so prediction units and difficulty are not directly comparable.

**Answer 8:** The model needs the prompt to predict the answer. The training objective may score only the answer, so prompt tokens remain in context while some prompt targets are masked from loss. This differs from causal attention masking.

**Answer 9:** SFT changes weights through training. LoRA changes trainable additions while the base weights usually remain frozen. A prompt and retrieval change the input during normal generation and do not automatically train weights.

**Answer 10:** Request-schema validation, health checks, and service startup/shutdown logic can be reused. The tokenizer, input format, length limit, memory needs, and generation method may change and must be aligned with the new model.

</details>

<!-- cell -->

### Study path after this lesson

If you cannot yet explain `text → IDs → vectors → attention → logits → loss` with an example, return to the relevant lesson and rerun its cells with one small change. First complete a controlled-experiment report; then study pretrained-model selection, fine-tuning, and retrieval in separate projects.

**Interview review:** `docs/INTERVIEW_FA.md` brings the earlier topics together. Answer with evidence from your own runs and clearly distinguish this small demo model from a production system.
