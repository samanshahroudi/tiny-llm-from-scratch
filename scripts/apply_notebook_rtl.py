"""Add/update presentation cells while preserving every existing learning cell."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import nbformat
from notebook_presentation import presentation_cells

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_IDS = {"tinyllm-rtl-help", "tinyllm-rtl-style"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-label", default="rtl")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = ROOT / "artifacts" / "runs" / "notebook_backups" / f"{args.backup_label}-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy2(path, backup / path.name)
        nb = nbformat.read(path, as_version=4)
        learning_cells = [c for c in nb.cells if c.get("id") not in PRESENTATION_IDS]
        nb.cells = presentation_cells(ROOT) + learning_cells
        nbformat.validate(nb)
        nbformat.write(nb, path)
        restored = nbformat.read(path, as_version=4)
        assert restored.cells[2:] == learning_cells, f"Learning cells changed: {path.name}"
        print("Updated:", path.name)
    print("Backup:", backup)


if __name__ == "__main__":
    main()
