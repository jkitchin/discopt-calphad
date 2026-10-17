"""Build the example notebooks from their percent-format sources, and run them.

Each notebook is written as a plain Python file in ``notebooks/src/``, with cells
separated by ``# %%`` (code) and ``# %% [markdown]`` (markdown, every line
commented with ``# ``). The source is what gets reviewed and diffed; the
``.ipynb`` beside this script is built from it, executed, with its outputs
stored, so it reads on GitHub and on the docs site without running anything.

    uv run python notebooks/build.py                  # build and run every notebook
    uv run python notebooks/build.py 02_magnetic      # just one
    uv run python notebooks/build.py --no-execute     # convert only

A notebook that raises fails the build.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def cells_of(text: str) -> list:
    """Split a percent-format source into notebook cells."""
    cells, kind, lines = [], None, []

    def flush():
        body = "\n".join(lines).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif kind == "code" and body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def build(src: Path, execute: bool = True, timeout: int = 3600, out_dir: Path | None = None) -> Path:
    """Build ``src`` into ``out_dir`` (default: beside this script); the notebook runs in ``notebooks/``."""
    nb = nbformat.v4.new_notebook()
    nb.cells = cells_of(src.read_text())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out = (out_dir or HERE) / f"{src.stem}.ipynb"
    if execute:
        t = time.time()
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
        try:
            client.execute()
        except CellExecutionError:
            nbformat.write(nb, out)  # keep the partly run notebook, to read the error in place
            raise
        print(f"{src.stem}: ran in {time.time() - t:.0f} s")
    nbformat.write(nb, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="notebook names (default: all)")
    ap.add_argument("--no-execute", action="store_true", help="convert without running")
    args = ap.parse_args(argv)
    sources = sorted(SRC.glob("*.py"))
    if args.names:
        sources = [s for s in sources if s.stem in args.names]
        if not sources:
            ap.error(f"no notebook source named {args.names}")
    for src in sources:
        print("wrote", build(src, execute=not args.no_execute).relative_to(HERE.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
