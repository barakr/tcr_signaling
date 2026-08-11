"""Execute the KS tutorial notebooks and prove they did real work.

Nothing executed these before, which is how `01_explore_models.ipynb` came to sit
broken for months passing `--contact_fraction` and `--cd45_bulk_density`, neither of
which has existed since the 2026-03 rewrite.

Checking the exit status alone is not enough. A notebook whose cells all skip, or whose
simulation silently produces nothing, still exits cleanly -- Status.md records exactly
that failure mode ("reported PASS for a tutorial that did no useful work"). So each
notebook ends in a self-check cell that asserts its own scientific claims and prints
`[KS_N self-check OK]`, and this test requires that beacon to appear.

Marked `slow`: the five notebooks together run a few hundred simulations, which is fine
for CI but not for a pre-commit hook. CI runs them in a dedicated step.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_NOTEBOOK_DIR = _REPO / "notebooks" / "models" / "kinetic_segregation"

# Generous by default, because a developer's first run compiles the model. CI
# lowers it via KS_NOTEBOOK_TIMEOUT so that a stuck cell fails naming the cell,
# rather than letting five notebooks sit at 30 minutes each.
_TIMEOUT = int(os.environ.get("KS_NOTEBOOK_TIMEOUT", "1800"))

# Substrings that mean a cell reported trouble without raising.
_FAILURE_MARKERS = ("Traceback", "ModuleNotFoundError", "CellExecutionError")


def _notebooks() -> list[Path]:
    return sorted(_NOTEBOOK_DIR.glob("KS_*.ipynb"))


def test_notebook_directory_is_populated():
    """Guard the guard: an empty glob would make every test below vacuously pass."""
    found = _notebooks()
    assert found, f"no KS_*.ipynb found in {_NOTEBOOK_DIR}"
    assert len(found) >= 5, f"expected the full KS series, found only {[p.name for p in found]}"


@pytest.mark.slow
@pytest.mark.parametrize("notebook", _notebooks(), ids=lambda p: p.stem)
def test_notebook_executes_and_self_check_passes(notebook: Path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(notebook, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=_TIMEOUT,
        kernel_name="python3",
        # cwd = the notebook's own directory, so `import ks_tutorial` resolves the
        # same way it does for a reader who opened it in Jupyter.
        resources={"metadata": {"path": str(_NOTEBOOK_DIR)}},
    )
    client.execute()

    texts: list[str] = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                raise AssertionError(f"{notebook.name}: {out.get('ename')}: {out.get('evalue')}")
            texts.append(out.get("text", "") or "")

    combined = "\n".join(texts)
    for marker in _FAILURE_MARKERS:
        assert marker not in combined, f"{notebook.name} printed a failure marker: {marker}"

    # Notebooks print the short form, e.g. "[KS_1 self-check OK]" from
    # KS_1_Kinetic_Segregation.ipynb.
    short = "_".join(notebook.stem.split("_")[:2])
    beacon = f"[{short} self-check OK]"
    assert beacon in combined, (
        f"{notebook.name} ran without raising but never printed {beacon!r}. "
        "Either the self-check cell was removed or it did not reach the end."
    )
