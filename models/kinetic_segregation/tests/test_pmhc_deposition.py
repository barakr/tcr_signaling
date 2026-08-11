"""Tests for the Phase-2 pMHC deposition scheme and its resolution diagnostics.

Background. The per-cell binding weights consumed by the Phase-2 grid update are a
particle-mesh deposition of Gaussian kernels of width `sigma_r` (default 2 nm). The
historical scheme samples each kernel at the CELL CENTRE, which is only valid while
`sigma_r >= dx`. Once cells are wider than the kernel, most pMHC have no cell centre
inside the 3*sigma_r cutoff and deposit nothing, so the total deposited weight collapses
as the mesh coarsens and the TCR attraction quietly leaves the membrane update.

`--pmhc_deposition area` instead deposits the exact cell AVERAGE of the same kernel
(closed form, via erf), which is mesh-independent. `point` remains the default so that
existing results and tests/reference_values.json stay bit-identical; these tests pin
both behaviours and the diagnostics that make the difference visible.

Phase 1 (molecular moves) is unaffected by either scheme -- it evaluates the field at
continuous positions via `pmhc_influence_at`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from models.kinetic_segregation import ks_build

_PKG = Path(__file__).resolve().parents[1]
_BINARY = ks_build.find_binary() or (_PKG / ks_build.binary_name())

# A pMHC count and sigma_r for which the resolved target is easy to reason about.
_COMMON = [
    "--time_sec",
    "1",
    "--rigidity_kT",
    "5",
    "--n_steps",
    "150",
    "--n_tcr",
    "30",
    "--n_cd45",
    "60",
    "--n_pmhc",
    "30",
    "--pmhc_mode",
    "inner_circle",
    "--no-gpu",
]


def _run(tmp_path, *, patch, grid, deposition=None, label="run"):
    if not _BINARY.exists():
        try:
            ks_build.build()
        except RuntimeError as exc:
            pytest.skip(f"Failed to build binary: {exc}")
    if not _BINARY.exists():
        pytest.skip("ks_gpu not available")
    cmd = [
        str(_BINARY),
        "--run-dir",
        str(tmp_path / label),
        "--patch_size",
        str(patch),
        "--grid_size",
        str(grid),
        *_COMMON,
    ]
    if deposition:
        cmd += ["--pmhc_deposition", deposition]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads(proc.stdout)["diagnostics"], proc.stderr


class TestDiagnostics:
    def test_diagnostics_are_reported(self, tmp_path):
        d, _ = _run(tmp_path, patch=500, grid=64)
        for key in (
            "pmhc_influence_max",
            "pmhc_influence_sum",
            "pmhc_influence_expected",
            "pmhc_deposition",
        ):
            assert key in d, f"missing diagnostic {key}"
        assert d["pmhc_deposition"] == "point", "point must remain the default"

    def test_expected_is_the_mesh_independent_target(self, tmp_path):
        """expected = n_pmhc * 2*pi*sigma_r^2 / dx^2, so it grows as dx shrinks."""
        coarse, _ = _run(tmp_path, patch=2000, grid=64, label="c")  # dx = 31.25
        fine, _ = _run(tmp_path, patch=500, grid=64, label="f")  # dx = 7.8125
        ratio = fine["pmhc_influence_expected"] / coarse["pmhc_influence_expected"]
        assert ratio == pytest.approx(16.0, rel=0.02), (
            "expected weight should scale as 1/dx^2; dx dropped 4x so it should rise 16x"
        )


class TestPointModeUnderResolves:
    def test_coarse_mesh_loses_most_of_the_weight(self, tmp_path):
        """The defect this whole module exists to document."""
        d, _ = _run(tmp_path, patch=2000, grid=64)  # dx/sigma_r = 15.6
        assert d["pmhc_influence_sum"] < 0.25 * d["pmhc_influence_expected"], (
            "point sampling at dx >> sigma_r should deposit far less than the target"
        )

    def test_coarse_mesh_warns(self, tmp_path):
        _, err = _run(tmp_path, patch=2000, grid=64)
        assert "WARN-PMHC" in err, "an under-resolved field must warn on stderr"

    def test_resolved_mesh_does_not_warn(self, tmp_path):
        d, err = _run(tmp_path, patch=500, grid=100)  # dx = 5 nm, the sweep's choice
        assert "WARN-PMHC" not in err
        assert d["pmhc_influence_sum"] == pytest.approx(d["pmhc_influence_expected"], rel=0.05), (
            "at dx ~ sigma_r point sampling should already be close to the target"
        )


class TestAreaMode:
    def test_area_hits_the_target_on_a_coarse_mesh(self, tmp_path):
        d, err = _run(tmp_path, patch=2000, grid=64, deposition="area")
        assert d["pmhc_deposition"] == "area"
        assert d["pmhc_influence_sum"] == pytest.approx(d["pmhc_influence_expected"], rel=0.05), (
            "area deposition must be mesh-independent"
        )
        assert "WARN-PMHC" not in err

    def test_area_recovers_what_point_loses(self, tmp_path):
        point, _ = _run(tmp_path, patch=2000, grid=64, label="p")
        area, _ = _run(tmp_path, patch=2000, grid=64, deposition="area", label="a")
        assert area["pmhc_influence_sum"] > 10 * point["pmhc_influence_sum"]

    def test_schemes_converge_as_the_mesh_is_refined(self, tmp_path):
        """The correctness argument: area -> point as dx -> 0, so they are the same
        physics, differing only in how faithfully the mesh represents it."""
        coarse_gap, fine_gap = None, None
        for patch, grid, slot in ((2000, 64, "coarse"), (250, 128, "fine")):
            p, _ = _run(tmp_path, patch=patch, grid=grid, label=f"p{grid}{patch}")
            a, _ = _run(
                tmp_path, patch=patch, grid=grid, deposition="area", label=f"a{grid}{patch}"
            )
            gap = abs(a["pmhc_influence_sum"] - p["pmhc_influence_sum"]) / a["pmhc_influence_sum"]
            if slot == "coarse":
                coarse_gap = gap
            else:
                fine_gap = gap
        assert coarse_gap > 0.5, "the two schemes must disagree badly on a coarse mesh"
        assert fine_gap < 0.1, f"they must agree on a fine mesh, got {fine_gap:.3f}"
        assert fine_gap < coarse_gap


class TestFlagValidation:
    def test_typo_is_rejected_not_silently_remapped(self, tmp_path):
        """Unlike --binding_mode/--step_mode/--pmhc_mode, which silently select the
        other option on a typo (see KS_5), this flag validates its argument."""
        if not _BINARY.exists():
            pytest.skip("ks_gpu not available")
        proc = subprocess.run(
            [
                str(_BINARY),
                "--run-dir",
                str(tmp_path / "x"),
                "--patch_size",
                "500",
                "--grid_size",
                "64",
                "--pmhc_deposition",
                "aera",
                *_COMMON,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode != 0, "a misspelled deposition mode must fail loudly"
        assert "must be 'point'/'0' or 'area'/'1'" in proc.stderr

    def test_numeric_aliases_work(self, tmp_path):
        """A ModelSpec DOE grid is float-only, so a spec can only send "1.0"."""
        for value, expected in (("0", "point"), ("1", "area"), ("1.0", "area")):
            d, _ = _run(tmp_path, patch=500, grid=64, deposition=value, label=f"n{value}")
            assert d["pmhc_deposition"] == expected, f"{value!r} should mean {expected}"

    def test_wrapper_forwards_the_flag(self, tmp_path):
        """The Python wrapper must expose it, or specs cannot opt in."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.kinetic_segregation",
                "--run-dir",
                str(tmp_path / "w"),
                "--grid_size",
                "64",
                "--pmhc_deposition",
                "area",
                *_COMMON,
            ],
            cwd=str(_PKG.parents[1]),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert json.loads(proc.stdout)["diagnostics"]["pmhc_deposition"] == "area"
