"""Build lessons 01-07 from editable Markdown, preserving notebook 00.

Blocks are separated by <!-- cell -->. A standalone Python fence is a code
cell; every other block is Markdown. No learner code runs during generation.
"""
import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

import nbformat

ROOT = Path(__file__).resolve().parents[1]
LESSONS = (
    "01_language_model_basics",
    "02_tokenization_and_embeddings",
    "03_attention_and_transformers",
    "04_build_tiny_gpt",
    "05_train_evaluate_and_checkpoint",
    "06_serve_deploy_and_publish",
    "07_huggingface_and_next_steps",
)


def build_lesson(name, root=ROOT):
    if name not in LESSONS:
        raise ValueError(f"Unknown lesson: {name}")
    source = (root / "docs" / f"{name}.lesson.md").read_text(encoding="utf-8")
    # English notebooks use Jupyter's normal left-to-right layout.
    cells = []
    for index, block in enumerate(source.split("\n<!-- cell -->\n")):
        block = block.strip()
        if not block:
            continue
        match = re.fullmatch(r"```python\n(.*?)\n```", block, flags=re.DOTALL)
        cell_id = f"{name[:2]}-foundation-{index:03d}"
        if match:
            code = match.group(1)
            compile(code, f"{name}:{cell_id}", "exec")
            cells.append(nbformat.v4.new_code_cell(code, id=cell_id))
        else:
            if re.search(r"^```python\s*$", block, flags=re.MULTILINE):
                raise ValueError(f"Python fence must occupy its whole cell: {cell_id}")
            cells.append(nbformat.v4.new_markdown_cell(block, id=cell_id))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (tiny-llm-fa)",
                       "language": "python", "name": "tiny-llm-fa"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=LESSONS, default=list(LESSONS))
    parser.add_argument("--write", action="store_true", help="Back up and write selected notebooks")
    args = parser.parse_args()
    # Validate every selected source before replacing any notebook.
    notebooks = {name: build_lesson(name) for name in args.only}
    if args.write:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = ROOT / "artifacts" / "runs" / "notebook_backups" / f"lessons-{stamp}"
        backup.mkdir(parents=True)
        for name, notebook in notebooks.items():
            target = ROOT / "notebooks" / f"{name}.ipynb"
            if target.exists():
                shutil.copy2(target, backup / target.name)
            nbformat.write(notebook, target)
        print("Backup:", backup)
    for name, notebook in notebooks.items():
        print(name, "cells=", len(notebook.cells), "code=",
              sum(cell.cell_type == "code" for cell in notebook.cells))


if __name__ == "__main__":
    main()
