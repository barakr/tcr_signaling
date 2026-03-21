"""Tests for new binary defaults: gaussian binding, brownian step mode, auto-pMHC.

These tests exercise the execution paths introduced when the binary defaults
changed from forced/paper/n_pmhc=0 to gaussian/brownian/auto-pMHC-from-density.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"

# Physical constants matching simulation.h
PMHC_DENSITY_PER_UM2 = 300.0
PMHC_RADIUS_FRAC_DEFAULT = 1.0 / 3.0
PATCH_SIZE_DEFAULT = 2000.0


def _ensure_binary():
    if _BINARY.exists():
        return
    result = subprocess.run(
        ["make"], cwd=str(_PKG_DIR),
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to build binary: {result.stderr}")


def _run(tmp_path, label="run", **kwargs):
    """Run the C binary with given params, return parsed JSON output."""
    rd = tmp_path / label
    cmd = [str(_BINARY), "--run-dir", str(rd), "--no-gpu"]
    defaults = {
        "time_sec": 10.0,
        "rigidity_kT": 20.0,
        "seed": 42,
        "n_steps": 100,
        "grid_size": 32,
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        cmd.extend([f"--{k}", str(v)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Binary failed: {result.stderr}"
    return json.loads(result.stdout.strip()), result.stderr


# ─── Gaussian binding mode tests ─────────────────────────────────────────


class TestGaussianBinding:
    """Tests for gaussian binding mode (binding_mode=0)."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_gaussian_binding_positive_depletion(self, tmp_path):
        """Gaussian binding mode should produce positive depletion with enough steps."""
        data, _ = _run(
            tmp_path, label="gauss_depl",
            binding_mode="gaussian",
            step_mode="paper",
            n_pmhc=50,
            pmhc_mode="inner_circle",
            pmhc_seed=1,
            n_steps=500,
            rigidity_kT=30.0,
        )
        assert data["depletion_width_nm"] > 0, (
            f"Gaussian binding should produce positive depletion, got {data['depletion_width_nm']}"
        )

    def test_gaussian_binding_bound_tcrs_can_move(self, tmp_path):
        """In gaussian binding, the accept rate for bound TCRs should be > 0.

        Unlike forced binding (all-or-nothing), gaussian binding uses a smooth
        potential well, so molecules near pMHC can still move.
        """
        data, _ = _run(
            tmp_path, label="gauss_move",
            binding_mode="gaussian",
            step_mode="paper",
            n_pmhc=50,
            pmhc_mode="inner_circle",
            pmhc_seed=1,
            n_steps=200,
        )
        rate = data["diagnostics"]["accept_rate"]
        assert rate > 0.0, "Accept rate should be positive with gaussian binding"


# ─── Brownian step mode tests ────────────────────────────────────────────


class TestBrownianMode:
    """Tests for brownian step mode (step_mode=0)."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_brownian_with_auto_pmhc_produces_results(self, tmp_path):
        """Brownian mode with auto-pMHC density should produce reasonable results."""
        data, stderr = _run(
            tmp_path, label="brown_auto",
            binding_mode="gaussian",
            step_mode="brownian",
            # n_pmhc omitted → auto-compute from density
            n_steps=200,
        )
        assert data["depletion_width_nm"] >= 0
        rate = data["diagnostics"]["accept_rate"]
        assert 0.0 < rate < 1.0, f"Accept rate out of range: {rate}"

    def test_brownian_step_size_h_not_fixed(self, tmp_path):
        """Brownian mode derives step_size_h from diffusion, not fixed at 1.0 nm."""
        data, _ = _run(
            tmp_path, label="brown_step",
            binding_mode="forced",
            step_mode="brownian",
            n_pmhc=-1,
            n_steps=50,
        )
        step_h = data["diagnostics"]["step_size_h_nm"]
        assert step_h != pytest.approx(1.0, abs=0.01), (
            f"Brownian step_size_h should not be 1.0 nm (paper mode), got {step_h}"
        )

    def test_paper_step_size_h_is_fixed(self, tmp_path):
        """Paper mode has fixed step_size_h = 1.0 nm."""
        data, _ = _run(
            tmp_path, label="paper_step",
            binding_mode="forced",
            step_mode="paper",
            n_pmhc=-1,
            n_steps=50,
        )
        step_h = data["diagnostics"]["step_size_h_nm"]
        assert step_h == pytest.approx(1.0, abs=1e-6), (
            f"Paper step_size_h should be 1.0 nm, got {step_h}"
        )

    def test_brownian_vs_paper_step_h_scaling(self, tmp_path):
        """Brownian step_size_h = sqrt(2*D_h*dt), paper = 1.0 nm."""
        d_paper, _ = _run(
            tmp_path, label="scale_paper",
            binding_mode="forced",
            step_mode="paper",
            n_pmhc=-1,
            n_steps=20,
        )
        d_brown, _ = _run(
            tmp_path, label="scale_brown",
            binding_mode="forced",
            step_mode="brownian",
            n_pmhc=-1,
            n_steps=20,
        )
        assert d_paper["diagnostics"]["step_size_h_nm"] == pytest.approx(1.0)
        assert d_paper["diagnostics"]["step_size_h_nm"] != pytest.approx(
            d_brown["diagnostics"]["step_size_h_nm"], abs=0.01
        )


# ─── Auto-pMHC from density tests ───────────────────────────────────────


class TestAutoPmhc:
    """Tests for auto-computation of n_pmhc from paper density."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_auto_pmhc_uniform_default_patch(self, tmp_path):
        """With default patch_size=2000nm and uniform mode, n_pmhc = 300 * 4.0 = 1200."""
        # patch_size = 2000 nm → area = 4e6 nm² = 4.0 µm²
        # density = 300/µm² → n_pmhc = 1200
        data, stderr = _run(
            tmp_path, label="auto_uni",
            binding_mode="gaussian",
            step_mode="brownian",
            pmhc_mode="uniform",
            # n_pmhc omitted → auto-compute
            n_steps=50,
        )
        assert "AUTO-PMHC" in stderr, "Expected auto-pMHC log message"
        expected_n = int(PMHC_DENSITY_PER_UM2 * (PATCH_SIZE_DEFAULT / 1000.0) ** 2 + 0.5)
        assert f"n_pmhc={expected_n}" in stderr, (
            f"Expected n_pmhc={expected_n} in stderr, got: {stderr}"
        )

    def test_auto_pmhc_small_patch(self, tmp_path):
        """For patch_size=250nm, uniform, density=300/µm², expect n_pmhc ≈ 19."""
        # patch_size = 250 nm → area = 62500 nm² = 0.0625 µm²
        # n_pmhc = 300 * 0.0625 = 18.75 → rounds to 19
        patch_size = 250.0
        area_um2 = (patch_size / 1000.0) ** 2
        expected_n = int(PMHC_DENSITY_PER_UM2 * area_um2 + 0.5)
        assert expected_n == 19, f"Sanity check: expected 19, computed {expected_n}"

        data, stderr = _run(
            tmp_path, label="auto_small",
            binding_mode="gaussian",
            step_mode="brownian",
            pmhc_mode="uniform",
            patch_size=patch_size,
            grid_size=16,
            n_steps=50,
        )
        assert "AUTO-PMHC" in stderr
        assert f"n_pmhc={expected_n}" in stderr, (
            f"Expected n_pmhc={expected_n} for patch={patch_size}nm, got: {stderr}"
        )

    def test_explicit_n_pmhc_overrides_density(self, tmp_path):
        """Explicit --n_pmhc should override auto-computation from density."""
        data, stderr = _run(
            tmp_path, label="override",
            binding_mode="gaussian",
            step_mode="brownian",
            n_pmhc=7,
            pmhc_mode="uniform",
            pmhc_seed=1,
            n_steps=50,
        )
        # Should NOT see AUTO-PMHC message when n_pmhc is explicitly set
        assert "AUTO-PMHC" not in stderr, (
            "Should not auto-compute when n_pmhc is explicitly provided"
        )
        assert data["depletion_width_nm"] >= 0


# ─── Execution path combination tests ────────────────────────────────────


class TestExecutionPaths:
    """Test different combinations of binding_mode and step_mode."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    @pytest.mark.parametrize(
        "binding_mode, step_mode",
        [
            ("forced", "paper"),
            ("gaussian", "brownian"),
            ("gaussian", "paper"),
            ("forced", "brownian"),
        ],
        ids=["forced+paper", "gaussian+brownian", "gaussian+paper", "forced+brownian"],
    )
    def test_combination_runs_successfully(self, tmp_path, binding_mode, step_mode):
        """Each valid combination of binding + step mode should run without error."""
        data, _ = _run(
            tmp_path,
            label=f"{binding_mode}_{step_mode}",
            binding_mode=binding_mode,
            step_mode=step_mode,
            n_pmhc=20,
            pmhc_mode="inner_circle",
            pmhc_seed=1,
            n_steps=100,
        )
        assert data["depletion_width_nm"] >= 0
        rate = data["diagnostics"]["accept_rate"]
        assert 0.0 < rate < 1.0, f"Accept rate out of range: {rate}"

    @pytest.mark.parametrize(
        "binding_mode, step_mode",
        [
            ("forced", "paper"),
            ("gaussian", "brownian"),
            ("gaussian", "paper"),
            ("forced", "brownian"),
        ],
        ids=["forced+paper", "gaussian+brownian", "gaussian+paper", "forced+brownian"],
    )
    def test_combination_deterministic(self, tmp_path, binding_mode, step_mode):
        """Each combination should be deterministic with same seed."""
        common = dict(
            binding_mode=binding_mode,
            step_mode=step_mode,
            n_pmhc=20,
            pmhc_mode="inner_circle",
            pmhc_seed=1,
            n_steps=50,
            seed=99,
        )
        d1, _ = _run(tmp_path, label=f"det1_{binding_mode}_{step_mode}", **common)
        d2, _ = _run(tmp_path, label=f"det2_{binding_mode}_{step_mode}", **common)
        assert d1["depletion_width_nm"] == d2["depletion_width_nm"]
        assert d1["diagnostics"]["accept_rate"] == d2["diagnostics"]["accept_rate"]
