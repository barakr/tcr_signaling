"""The contract every model under `models/` must satisfy, checked for all of them.

This project exists to combine independently-built models, so the property that
matters most is that they stay independent. The composition seam is deliberately
a **process boundary, not a Python API**: each `specs/model.<name>.json` declares
an entrypoint of the form

    python -m projects.tcr_signaling.models.<name>

and the model reads CLI flags and writes one JSON object to stdout. That is why a
model can be fully autonomous — kinetic segregation imports nothing from the
`bayesian-metamodeling` framework at all — and still be composable. Nothing needs
to import a model to use it.

These tests protect that seam. They are written to discover models rather than
list them, so the next one is held to the same rules on the day it lands.

(When a second compiled model appears, move this module and `test_docs.py` to
`models/tests/` — they live under kinetic_segregation only because it is
currently the sole model with a build.)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MODELS = _REPO / "models"
_SPECS = _REPO / "specs"


def model_dirs() -> list[Path]:
    """Every model package under `models/`, found rather than listed."""
    return sorted(
        p
        for p in _MODELS.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and (p / "__init__.py").is_file()
    )


def _imported_modules(path: Path) -> list[str]:
    """Absolute and relative import targets in a file, via AST.

    AST rather than a substring scan, so a docstring or comment that *mentions*
    an import — as these modules do when explaining the rule — is not mistaken
    for one. That exact false positive has already been hit twice here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    return names


def test_models_are_discovered():
    """Guard the guard: an empty list makes every test below vacuous."""
    found = model_dirs()
    assert len(found) >= 4, f"expected the four partial models, found {[p.name for p in found]}"


@pytest.mark.parametrize("model", model_dirs(), ids=lambda p: p.name)
class TestEveryModel:
    def test_is_runnable_as_a_module(self, model: Path):
        """`__main__.py` is the composition contract: `python -m ...<name>`."""
        assert (model / "__main__.py").is_file(), (
            f"{model.name} has no __main__.py, so the ModelSpec entrypoint "
            "`python -m projects.tcr_signaling.models."
            f"{model.name}` cannot run it"
        )

    def test_has_its_own_tests(self, model: Path):
        assert (model / "tests").is_dir(), f"{model.name} ships no tests/ directory"

    def test_imports_siblings_relatively(self, model: Path):
        """Absolute `models.<name>` imports break under the real entrypoint.

        A model runs under two package paths: `models.<name>` when pytest runs
        from this repo, and `projects.tcr_signaling.models.<name>` when the
        framework runs it from the parent repo — which is what every spec
        declares. An absolute import naming `models` resolves only under the
        first, so the failure is invisible to this repo's own test suite and
        shows up as a broken sweep in the parent. That happened once, to
        kinetic_segregation; the other three were already correct.
        """
        offenders = []
        for py in model.rglob("*.py"):
            if "tests" in py.parts or "build" in py.parts:
                continue
            for name in _imported_modules(py):
                if name.startswith("models."):
                    offenders.append(f"{py.relative_to(_MODELS)}: {name}")
        assert not offenders, (
            f"{offenders} import via the absolute `models.` path. Use a relative "
            "import (`from . import x`), which resolves under both package paths."
        )

    def test_does_not_import_another_model(self, model: Path):
        """Independence is the whole premise: models compose by subprocess, not import.

        A shared helper between two models would make them one model with extra
        steps, and would break the promise that each can be developed, tested and
        published on its own.
        """
        siblings = {p.name for p in model_dirs()} - {model.name}
        offenders = []
        for py in model.rglob("*.py"):
            if "build" in py.parts:
                continue
            for name in _imported_modules(py):
                bare = name.lstrip(".")
                head = bare.split(".")[0]
                if head in siblings or any(bare.startswith(f"models.{s}") for s in siblings):
                    offenders.append(f"{py.relative_to(_MODELS)}: {name}")
        assert not offenders, (
            f"{model.name} imports another model ({offenders}). Models are combined "
            "through their CLI + JSON contract, never by importing each other."
        )


class TestSpecsAndModelsAgree:
    def test_every_spec_entrypoint_names_a_real_model(self):
        """A renamed directory silently breaks its spec otherwise."""
        names = {p.name for p in model_dirs()}
        problems = []
        for spec in sorted(_SPECS.glob("model.*.json")):
            data = json.loads(spec.read_text(encoding="utf-8"))
            entry = data.get("model", {}).get("artifact", {}).get("entrypoint")
            if not entry:
                problems.append(f"{spec.name}: no entrypoint")
                continue
            target = entry[-1].split(".")[-1]
            if target not in names:
                problems.append(f"{spec.name}: entrypoint targets {target!r}, which does not exist")
        assert not problems, problems

    def test_entrypoints_use_the_parent_qualified_path(self):
        """All specs must agree on how a model is addressed.

        The framework runs from the parent repo, so the path is
        `projects.tcr_signaling.models.<name>`. A spec using the short
        `models.<name>` form would work only when run from inside this repo.
        """
        problems = []
        for spec in sorted(_SPECS.glob("model.*.json")):
            data = json.loads(spec.read_text(encoding="utf-8"))
            entry = data.get("model", {}).get("artifact", {}).get("entrypoint") or []
            if len(entry) >= 3 and entry[1] == "-m":
                if not entry[2].startswith("projects.tcr_signaling.models."):
                    problems.append(f"{spec.name}: {entry[2]}")
        assert not problems, (
            f"specs address models inconsistently: {problems}. Use the "
            "parent-qualified path, which is what the framework invokes."
        )
