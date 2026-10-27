"""Build the course from its editable .lesson.md sources.

Usage: python scripts/build_notebooks.py --only 01 02
The original entry point is retained, but no old lesson prose is embedded here.
Generation backs up existing notebooks and never executes their code.
"""
import argparse
from datetime import datetime
from pathlib import Path
import shutil

import nbformat
from build_start_here import build_notebook
from build_lessons import LESSONS, build_lesson

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[f"{i:02d}" for i in range(8)],
                        default=[f"{i:02d}" for i in range(8)])
    args = parser.parse_args()
    selected = {}
    if "00" in args.only:
        selected["00_start_here"] = build_notebook(ROOT)
    for name in LESSONS:
        if name[:2] in args.only:
            selected[name] = build_lesson(name, ROOT)
    # Validate/build all before writing any of them.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = ROOT / "artifacts/runs/notebook_backups" / f"rebuild-{stamp}"
    backup.mkdir(parents=True)
    for name, notebook in selected.items():
        target = ROOT / "notebooks" / f"{name}.ipynb"
        if target.exists():
            shutil.copy2(target, backup / target.name)
        nbformat.write(notebook, target)
        print("Written:", target.name, "cells:", len(notebook.cells))
    print("Backup:", backup)


if __name__ == "__main__":
    main()
