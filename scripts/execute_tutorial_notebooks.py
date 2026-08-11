#!/usr/bin/env python3
"""Execute tutorial notebooks from the repository root and save outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def execute(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"PASS {path.relative_to(ROOT)} ({code_cells} code cells)")


def main() -> None:
    requested = [ROOT / item for item in sys.argv[1:]]
    paths = requested or sorted((ROOT / "notebooks").glob("tut*.ipynb"))
    for path in paths:
        execute(path)


if __name__ == "__main__":
    main()
