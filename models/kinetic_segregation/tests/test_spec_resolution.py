"""Audit every checked-in spec for Phase-2 mesh resolution.

A spec that leaves `patch_size`/`grid_size` at the binary defaults gets dx = 31.25 nm
against sigma_r = 2 nm, which under-resolves the pMHC influence field by ~16x in total
deposited weight (see test_pmhc_deposition.py for the mechanism, and the KS_3 notebook
for what it does to a contact).

This test does NOT assert that every spec is resolved -- several are not, and changing
their mesh would change the physical system they describe, which is a scientific
decision rather than a code fix. What it does assert is that the situation is
*declared*: an under-resolved spec must appear in KNOWN_UNRESOLVED with a reason. A new
spec that drifts into the degenerate regime fails this test instead of silently
producing contact-free results.

Same principle as the CI skip guard: exceptions are allowed, but never blank cheques.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

# Binary defaults from src/simulation.h, used when a spec is silent.
_DEFAULT_PATCH = 2000.0
_DEFAULT_GRID = 64
_SIGMA_R = 2.0

# Point-sampled deposition holds up while dx is within a few sigma_r. Beyond ~4x the
# deficit is already large; 15.6x (the default mesh) loses ~94% of the weight.
_MAX_DX_OVER_SIGMA_R = 4.0

# Specs known to be under-resolved, each with the reason it has not been changed.
# Removing an entry from here is how you certify a spec has been fixed.
_PRODUCTION = (
    "production spec feeding the metamodel; changing patch_size/grid_size changes the "
    "modelled system and would invalidate fitted surrogates. Pending a scientific "
    "decision -- see Status.md 2026-08-07."
)
_DEMO = (
    "speed-first demo fixture; kept coarse deliberately because it demonstrates the "
    "workflow rather than the physics."
)
_SWEEP = (
    "the DOE definition only; run.py overrides with --patch_size 500 --grid_size 100 "
    "(dx = 5 nm), so the executed runs ARE resolved."
)

KNOWN_UNRESOLVED = {
    "specs/model.kinetic_segregation.json": _PRODUCTION,
    "examples/specs/model.kinetic_segregation.fast.json": _DEMO,
    "examples/specs/model.kinetic_segregation.regular.json": _DEMO,
    "examples/specs/model.kinetic_segregation.extensive.json": _DEMO,
    "experiments/ks_behavior_sweep/spec.json": _SWEEP,
}


def _spec_paths():
    globs = (
        "specs/model.kinetic_segregation*.json",
        "examples/specs/model.kinetic_segregation*.json",
        "experiments/*/spec.json",
    )
    out = []
    for g in globs:
        out += sorted(_REPO.glob(g))
    return out


def _first(grid, key, default):
    v = grid.get(key)
    if isinstance(v, list) and v:
        return v[0]
    return v if v is not None else default


def _dx_of(path: Path) -> float:
    spec = json.loads(path.read_text(encoding="utf-8"))
    grid = spec.get("design", {}).get("grid", {}) or {}
    patch = float(_first(grid, "patch_size", _DEFAULT_PATCH))
    n = float(_first(grid, "grid_size", _DEFAULT_GRID))
    return patch / n


def test_specs_were_found():
    """Guard the guard -- an empty glob would make the audit vacuous."""
    assert _spec_paths(), "no KS specs found; the audit globs are wrong"


@pytest.mark.parametrize("spec", _spec_paths(), ids=lambda p: p.name)
def test_spec_is_resolved_or_declared(spec: Path):
    rel = spec.relative_to(_REPO).as_posix()
    ratio = _dx_of(spec) / _SIGMA_R
    resolved = ratio <= _MAX_DX_OVER_SIGMA_R

    if resolved:
        assert rel not in KNOWN_UNRESOLVED, (
            f"{rel} is now resolved (dx/sigma_r = {ratio:.1f}); remove it from "
            "KNOWN_UNRESOLVED so the audit keeps its teeth."
        )
        return

    assert rel in KNOWN_UNRESOLVED, (
        f"{rel} has dx/sigma_r = {ratio:.1f}, above the {_MAX_DX_OVER_SIGMA_R:.0f}x "
        "limit, so its pMHC influence field is badly under-resolved and the Phase-2 "
        "TCR attraction is largely absent. Either give it a finer mesh "
        "(patch_size/grid_size such that dx ~ 2 nm), pass --pmhc_deposition area, or "
        "add it to KNOWN_UNRESOLVED with the reason it is acceptable."
    )


def test_allowlist_has_no_stale_entries():
    """Every declared exception must still exist, so the list cannot rot."""
    present = {p.relative_to(_REPO).as_posix() for p in _spec_paths()}
    stale = set(KNOWN_UNRESOLVED) - present
    assert not stale, f"KNOWN_UNRESOLVED names specs that no longer exist: {sorted(stale)}"
