"""Build only lesson 00; its editable source lives in docs/00_start_here.lesson.md."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build_notebook(root=ROOT):
    lesson = (root / "docs" / "00_start_here.lesson.md").read_text(encoding="utf-8")
    # English notebooks use Jupyter's normal left-to-right layout.
    cells = []
    for index, block in enumerate(lesson.split("\n<!-- cell -->\n")):
        block = block.strip()
        if not block:
            continue
        match = re.fullmatch(r"```python\n(.*?)\n```", block, flags=re.DOTALL)
        cell_id = f"00-foundation-{index:03d}"
        if match:
            source = match.group(1)
            compile(source, f"00_start_here:{cell_id}", "exec")
            cell = nbformat.v4.new_code_cell(source, id=cell_id)
        else:
            cell = nbformat.v4.new_markdown_cell(block, id=cell_id)
        cells.append(cell)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (tiny-llm-fa)",
                       "language": "python", "name": "tiny-llm-fa"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Back up and replace only notebook 00")
    args = parser.parse_args()
    notebook = build_notebook()
    if args.write:
        destination = ROOT / "notebooks" / "00_start_here.ipynb"
        if destination.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = ROOT / "artifacts" / "runs" / "notebook_backups" / f"00-before-foundations-{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup / destination.name)
            print("Backup:", backup)
        nbformat.write(notebook, destination)
        print("Written:", destination)
    print("Cells:", len(notebook.cells),
          "Code:", sum(c.cell_type == "code" for c in notebook.cells))


if __name__ == "__main__":
    main()
