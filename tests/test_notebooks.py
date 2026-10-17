"""The example notebooks: each is built from its source, stored with outputs, and runs without error."""

import importlib.util
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SOURCES = sorted((NB / "src").glob("*.py"))


def _build_module():
    spec = importlib.util.spec_from_file_location("notebooks_build", NB / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_there_is_a_notebook_per_feature():
    stems = {s.stem for s in SOURCES}
    for feature in ("certified_equilibrium", "magnetic", "order_disorder", "ionic", "third_generation", "fitting"):
        assert any(feature in s for s in stems), feature


@pytest.mark.parametrize("src", SOURCES, ids=lambda p: p.stem)
def test_stored_notebook_matches_source_and_ran(src):
    """The stored .ipynb has the source's cells (rebuild after editing a source), with outputs and no errors."""
    built = _build_module().cells_of(src.read_text())
    stored = nbformat.read(NB / f"{src.stem}.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in stored.cells] == [(c.cell_type, c.source) for c in built], (
        f"notebooks/{src.stem}.ipynb is stale: run `uv run python notebooks/build.py {src.stem}`"
    )
    code = [c for c in stored.cells if c.cell_type == "code"]
    assert all(c.get("execution_count") for c in code), "stored without running"
    assert not [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]


@pytest.mark.slow
@pytest.mark.parametrize("src", SOURCES, ids=lambda p: p.stem)
def test_notebook_runs(src, tmp_path):
    """Execute the notebook from its source (a few minutes each); a cell that raises fails the test."""
    pytest.importorskip("nbclient")
    assert _build_module().build(src, out_dir=tmp_path).exists()
