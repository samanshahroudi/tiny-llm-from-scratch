"""Execute each notebook in a fresh real kernel; save outputs separately."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="16 updates per experiment; not a quality run")
    parser.add_argument("--only", nargs="*", help="Notebook prefixes, e.g. 03 05")
    args = parser.parse_args()
    out = ROOT / "artifacts" / "notebook_execution"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TINYLLM_SMOKE"] = "1" if args.smoke else "0"
    os.environ["TINYLLM_RUN_DIR"] = str(ROOT / "artifacts" / "runs" / ("smoke" if args.smoke else "full"))
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # python3 kernel is installed inside this project's environment by ipykernel.
    specs = KernelSpecManager().find_kernel_specs()
    kernel = "tiny-llm-fa" if "tiny-llm-fa" in specs else "python3"
    mode = "smoke" if args.smoke else "full"
    report_path = out / f"execution_report_{mode}.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "mode": mode, "kernel": kernel, "notebooks": []}
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if args.only and path.name[:2] not in args.only:
            continue
        started = time.perf_counter()
        print("Executing", path.name, flush=True)
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        NotebookClient(nb, timeout=600, kernel_name=kernel,
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        output_folder = out / mode
        output_folder.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, output_folder / path.name)
        report["notebooks"] = [r for r in report["notebooks"] if r["name"] != path.name]
        report["notebooks"].append({"name": path.name,
                                    "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                                    "seconds": round(time.perf_counter() - started, 2), "status": "passed"})
        report["notebooks"].sort(key=lambda r: r["name"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
