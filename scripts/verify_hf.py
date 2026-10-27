"""Opt-in execution of online cells without changing the teaching notebook."""
import argparse
import asyncio
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", action="store_true")
    parser.add_argument("--wikitext", action="store_true")
    parser.add_argument("--embeddings", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("Choose at least one online experiment")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    path = ROOT / "notebooks" / "07_huggingface_and_next_steps.ipynb"
    nb = nbformat.read(path, as_version=4)
    flags = next(c for c in nb.cells if c.cell_type == "code" and c.source.startswith("RUN_REAL_TOKENIZER"))
    flags.source = (f"RUN_REAL_TOKENIZER = {args.tokenizer}\n"
                    f"RUN_REAL_EMBEDDINGS = {args.embeddings}\nRUN_WIKITEXT = {args.wikitext}")
    specs = KernelSpecManager().find_kernel_specs()
    kernel = "tiny-llm-fa" if "tiny-llm-fa" in specs else "python3"
    client = NotebookClient(nb, timeout=600, kernel_name=kernel,
                            resources={"metadata": {"path": str(ROOT)}})
    selection = "-".join(name for name, enabled in vars(args).items() if enabled)
    folder = ROOT / "artifacts" / "runs" / "hf_verification" / selection
    folder.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
    finally:
        nbformat.write(nb, folder / path.name)
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream" and output.name == "stdout":
                print(output.text)


if __name__ == "__main__":
    main()
