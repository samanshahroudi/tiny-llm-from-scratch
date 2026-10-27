# Tiny language model: from concepts to code and service

This repository is a practical course built from eight Jupyter notebooks for software engineers moving into AI engineering. Read the explanation, predict the output, run each cell, and inspect the small experiments.

The course starts from Python and tensors, then follows `text → IDs → vectors → attention → logits → loss/generation → API`. It covers the foundations behind N01–N04 and prepares a clean portfolio project for later fine-tuning, RAG, and deployment work. No previous AI specialization is assumed; every new term is introduced before it is used.

The model is a small character-level GPT with two blocks, width `64`, four heads, a context window of `64`, and 114,515 parameters. It is intentionally small so attention, MLPs, training, and serving remain readable on CPU. The main path needs no pretrained model or external API.

## Start here

Read [notebook 00](notebooks/00_start_here.ipynb) first. It contains complete environment setup. From PowerShell in this directory:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e ".[notebooks,dev]"
.venv/Scripts/python.exe -m ipykernel install --user --name tiny-llm-fa --display-name "Python (tiny-llm-fa)"
.venv/Scripts/python.exe -m jupyterlab
```

Select the `Python (tiny-llm-fa)` kernel and run cells from top to bottom with `Shift+Enter`. The first cell in each notebook applies the display stylesheet; code and formulas remain left-to-right. No Windows or Chrome language change is required.

## Study order

| Step | Notebook | Observable result |
|---|---|---|
| 00 | [Environment and tensor basics](notebooks/00_start_here.ipynb) | shape and gradient of one weight |
| 01 | [Language-model basics](notebooks/01_language_model_basics.ipynb) | bigram, generation, loss, temperature, and budget |
| 02 | [Tokens and embeddings](notebooks/02_tokenization_and_embeddings.ipynb) | tokenizer, BPE from scratch, Unicode, and similarity |
| 03 | [Attention and transformers](notebooks/03_attention_and_transformers.ipynb) | Q/K/V, causal mask, future-leakage test, and KV cache |
| 04 | [Build GPT](notebooks/04_build_tiny_gpt.ipynb) | classes, tensor shapes, parameter count, and generation |
| 05 | [Train, evaluate, and checkpoint](notebooks/05_train_evaluate_and_checkpoint.ipynb) | two learning rates, curves, best model, test, and round-trip |
| 06 | [Serve and publish](notebooks/06_serve_deploy_and_publish.ipynb) | API, HTTP tests, Docker bundle, GitHub, and HF Space |
| 07 | [Hugging Face and next steps](notebooks/07_huggingface_and_next_steps.ipynb) | real tokenizer, embeddings, and optional WikiText |

Every notebook bootstraps itself. Notebook 06 can use the demo artifact without running notebook 05. Notebook 07 downloads nothing by default; each optional experiment is enabled explicitly inside the notebook.

## Data and limits

The main corpus contains 150 short English/Persian documents authored for this project under MIT. It is split by document into 114 train, 18 validation, and 18 test documents. The tokenizer is built from train only. Because templates are intentionally shared, a low loss demonstrates local pattern learning, not general language ability. The model is a base next-token model and has no instruction tuning.

Optional real-data experiments use [Salesforce/WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext). It is English, disabled by default, and has a license separate from the MIT code. An HF account is unnecessary for local training but required to host a Space.

## Demo artifact and service

`artifacts/demo/` contains a real 400-step CPU run, selected checkpoint, logs, and samples. Run the service with the prepared artifact and inspect `/`, `/docs`, and `/health`. For a notebook-selected checkpoint, set `TINYLLM_CHECKPOINT=artifacts/runs/notebook/selected.pt`.

## Verification

Run a lightweight smoke check across all active notebook cells:

```powershell
.venv/Scripts/python.exe scripts/execute_notebooks.py --smoke
```

Omit `--smoke` for the full run. Outputs are written under `artifacts/notebook_execution/`, while lesson sources remain output-free. The environment report is in [VALIDATION](docs/VALIDATION_FA.md).

## Publishing

The repository contains the files needed for GitHub and Hugging Face Docker Spaces. Notebook 06 contains the exact commit, push, Docker, and Space commands. No public repository or Space is created automatically.

## Structure

```text
notebooks/       # step-by-step lessons and executable cells
src/tinyllm/     # reusable model, data, training, artifacts, and API
data/             # training corpus and manifest
artifacts/demo/   # trained demo and execution evidence
docs/            # editable lesson sources, coverage, interview practice, and validation
 tests/           # model and service contracts
scripts/          # data/notebook builds, execution, verification, and packaging
.github/          # CI for tests and notebook smoke runs
```

The editable source for each lesson is `docs/<notebook-name>.lesson.md`. `<!-- cell -->` separates cells; a fenced `python` block becomes an executable cell. Build lessons with:

```powershell
.venv/Scripts/python.exe scripts/build_lessons.py --write
```

For a complete coverage map, see [docs/COVERAGE_FA.md](docs/COVERAGE_FA.md). Interview prompts are in [docs/INTERVIEW_FA.md](docs/INTERVIEW_FA.md).

