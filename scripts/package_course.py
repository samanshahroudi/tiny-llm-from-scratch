"""Make a portable standalone repository archive with an explicit file allowlist."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = ["README.md", "LICENSE", "pyproject.toml", "Dockerfile", ".gitignore",
             ".dockerignore", "environment-lock.txt"]
DIRECTORIES = ["src", "notebooks", "tests", "scripts", "docs", ".github", "artifacts/demo"]


def main():
    files = [ROOT / name for name in TOP_FILES if (ROOT / name).is_file()]
    files += [ROOT / "data" / name for name in ["documents.json", "manifest.json", "README.md"]]
    for directory in DIRECTORIES:
        files += [p for p in (ROOT / directory).rglob("*") if p.is_file()
                  and not any(part in ("__pycache__", ".ipynb_checkpoints") or part.endswith(".egg-info")
                              for part in p.relative_to(ROOT).parts)
                  and p.suffix not in (".pyc", ".pyo")]
    target = ROOT / "dist" / "tiny-llm-fa.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path.is_file():
                archive.write(path, str(Path("tiny-llm-fa") / path.relative_to(ROOT)))
    with zipfile.ZipFile(target) as archive:
        assert not any(".venv" in name or "/.git/" in name or "data/external" in name
                       for name in archive.namelist())
    print(target)
    print("Files:", len(set(files)), "MiB:", round(target.stat().st_size / 2**20, 2))


if __name__ == "__main__":
    main()
