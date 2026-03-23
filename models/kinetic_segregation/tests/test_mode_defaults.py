"""Tests for changed binary defaults (brownian step, gaussian binding, auto-pMHC, new auto-dt).

These tests verify that the NEW defaults produce expected behavior:
- step_mode defaults to brownian (step_size_h scales with sqrt(dt))
- binding_mode defaults to gaussian (bound TCRs can still move)
- auto-dt with MAX_BENDING_DE_PER_STEP=0.05 gives dt_auto inversely proportional to kappa
- pMHC density auto-computed from literature (300/um^2)
- gaussian vs forced binding produces different dynamics
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
D_H_NM2_PER_S = 50000.0
PMHC_DENSITY_PER_UM2 = 300.0
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
    """Run the binary with given params, return (parsed_json, stderr)."""
    rd = tmp_path / label
    cmd = [str(_BINARY), "--run-dir", str(rd), "--no-gpu"]
    defaults = {
        "time_sec": 10.0,
        "rigidity_kT": 20.0,
        "seed": 42,
        "n_steps": 50,
        "grid_size": 32,
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        cmd.extend([f"--{k}", str(v)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Binary failed: {result.stderr}"
    return json.loads(result.stdout.strip()), result.stderr


# ---- Default step_mode = brownian ----------------------------------------


class TestDefaultBrownianMode:
    """Verify default step_mode is brownian, not paper."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_default_step_h_not_one(self, tmp_path):
        """Default mode should give step_size_h != 1.0 nm (paper mode gives 1.0)."""
        data, _ = _run(tmp_path, label="def_step")
        step_h = data["diagnostics"]["step_size_h_nm"]
        assert step_h != pytest.approx(1.0, abs=0.01), (
            f"Default step_size_h = {step_h:.4f}, expected != 1.0 (brownian, not paper)"
        )

    def test_default_step_h_matches_brownian(self, tmp_path):
        """Default mode and explicit brownian should give identical step_size_h."""
        d_default, _ = _run(tmp_path, label="def_mode")
        d_brownian, _ = _run(tmp_path, label="expl_brown", step_mode="brownian")
        assert d_default["diagnostics"]["step_size_h_nm"] == pytest.approx(
            d_brownian["diagnostics"]["step_size_h_nm"]
        )

    def test_step_h_scales_with_sqrt_dt(self, tmp_path):
        """In brownian mode, step_size_h = sqrt(2 * D_h * dt)."""
        data, _ = _run(tmp_path, label="sqrt_check")
        diag = data["diagnostics"]
        dt = diag["dt_seconds"]
        step_h = diag["step_size_h_nm"]
        expected = math.sqrt(2.0 * D_H_NM2_PER_S * dt)
        assert step_h == pytest.approx(expected, rel=1e-3), (
            f"step_h={step_h:.4f} != sqrt(2*D_h*dt)={expected:.4f}"
        )

    def test_step_h_varies_with_kappa(self, tmp_path):
        """Brownian step_h should decrease as kappa increases (smaller dt)."""
        d_low, _ = _run(tmp_path, label="kappa_low", rigidity_kT=5.0, n_steps=10)
        d_high, _ = _run(tmp_path, label="kappa_high", rigidity_kT=80.0, n_steps=10)
        step_low = d_low["diagnostics"]["step_size_h_nm"]
        step_high = d_high["diagnostics"]["step_size_h_nm"]
        assert step_low > step_high, (
            f"Expected step_h(kappa=5)={step_low:.4f} > step_h(kappa=80)={step_high:.4f}"
        )
        # Ratio should be sqrt(80/5) = 4.0
        ratio = step_low / step_high
        assert ratio == pytest.approx(math.sqrt(80.0 / 5.0), rel=0.01), (
            f"step_h ratio {ratio:.2f} != expected {math.sqrt(80.0/5.0):.2f}"
        )


# ---- Default binding_mode = gaussian -------------------------------------


class TestDefaultGaussianBinding:
    """Verify default binding_mode is gaussian."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_default_accept_rate_matches_gaussian(self, tmp_path):
        """Default mode and explicit gaussian should give the same accept rate."""
        d_default, _ = _run(
            tmp_path, label="def_bind",
            n_pmhc=30, pmhc_mode="inner_circle", pmhc_seed=1, n_steps=100,
        )
        d_gaussian, _ = _run(
            tmp_path, label="gauss_bind",
            binding_mode="gaussian",
            n_pmhc=30, pmhc_mode="inner_circle", pmhc_seed=1, n_steps=100,
        )
        assert d_default["diagnostics"]["accept_rate"] == pytest.approx(
            d_gaussian["diagnostics"]["accept_rate"]
        )

    def test_gaussian_bound_tcrs_can_move(self, tmp_path):
        """In gaussian binding, the accept rate should be > 0 (molecules can still move)."""
        data, _ = _run(
            tmp_path, label="gauss_move",
            binding_mode="gaussian",
            n_pmhc=50, pmhc_mode="inner_circle", pmhc_seed=1,
            n_steps=200,
        )
        rate = data["diagnostics"]["accept_rate"]
        assert rate > 0.0, "Accept rate should be positive with gaussian binding"

    def test_gaussian_produces_segregation(self, tmp_path):
        """Gaussian binding should still produce TCR/CD45 segregation."""
        data, _ = _run(
            tmp_path, label="gauss_seg",
            binding_mode="gaussian",
            n_pmhc=50, pmhc_mode="inner_circle", pmhc_seed=1,
            n_steps=200, rigidity_kT=30.0,
        )
        diag = data["diagnostics"]
        assert diag["final_tcr_mean_r_nm"] < diag["final_cd45_mean_r_nm"], (
            f"TCR ({diag['final_tcr_mean_r_nm']:.1f}) should be closer to center "
            f"than CD45 ({diag['final_cd45_mean_r_nm']:.1f})"
        )


# ---- Auto-dt with MAX_BENDING_DE_PER_STEP = 0.05 -------------------------


class TestAutoDt:
    """Verify auto-dt calibration: dt_auto inversely proportional to kappa."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_dt_auto_inversely_proportional_to_kappa(self, tmp_path):
        """kappa * dt_auto should be constant (within floating point)."""
        products = []
        for kappa in [5.0, 10.0, 20.0, 40.0, 80.0]:
            data, _ = _run(
                tmp_path, label=f"dtk_{int(kappa)}",
                rigidity_kT=kappa, n_steps=5,
            )
            dt_auto = data["diagnostics"]["dt_auto_seconds"]
            products.append(kappa * dt_auto)
        # All products should be equal
        for p in products[1:]:
            assert p == pytest.approx(products[0], rel=1e-6), (
                f"kappa*dt_auto not constant: {products}"
            )

    def test_dt_auto_positive(self, tmp_path):
        """Auto-calibrated dt should always be positive."""
        for kappa in [1.0, 20.0, 100.0]:
            data, _ = _run(
                tmp_path, label=f"dtp_{int(kappa)}",
                rigidity_kT=kappa, n_steps=5,
            )
            assert data["diagnostics"]["dt_auto_seconds"] > 0

    def test_higher_kappa_smaller_dt(self, tmp_path):
        """Higher rigidity should give smaller auto-dt."""
        d_low, _ = _run(tmp_path, label="dt_low_k", rigidity_kT=5.0, n_steps=5)
        d_high, _ = _run(tmp_path, label="dt_high_k", rigidity_kT=80.0, n_steps=5)
        assert d_low["diagnostics"]["dt_auto_seconds"] > d_high["diagnostics"]["dt_auto_seconds"]


# ---- pMHC density auto-computation ---------------------------------------


class TestPmhcDensity:
    """Verify auto-pMHC density and different pmhc_mode options."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_auto_pmhc_uniform_default_patch(self, tmp_path):
        """Default patch (2000nm), uniform mode: n_pmhc = 300 * 4.0 = 1200."""
        _, stderr = _run(
            tmp_path, label="pmhc_uni",
            pmhc_mode="uniform", n_steps=10,
        )
        expected = int(PMHC_DENSITY_PER_UM2 * (PATCH_SIZE_DEFAULT / 1000.0) ** 2 + 0.5)
        assert "AUTO-PMHC" in stderr, "Expected auto-pMHC log message"
        assert f"n_pmhc={expected}" in stderr

    def test_auto_pmhc_inner_circle(self, tmp_path):
        """Inner-circle mode uses pi*r^2 area where r = patch_size/3."""
        _, stderr = _run(
            tmp_path, label="pmhc_ic",
            pmhc_mode="inner_circle", n_steps=10,
        )
        r = PATCH_SIZE_DEFAULT / 3.0
        area = math.pi * r * r
        expected = int(PMHC_DENSITY_PER_UM2 * area / 1e6 + 0.5)
        assert "AUTO-PMHC" in stderr
        assert f"n_pmhc={expected}" in stderr

    def test_explicit_n_pmhc_overrides_auto(self, tmp_path):
        """Explicit --n_pmhc should suppress auto-computation."""
        _, stderr = _run(
            tmp_path, label="pmhc_explicit",
            n_pmhc=10, pmhc_mode="uniform", pmhc_seed=1,
            n_steps=10,
        )
        assert "AUTO-PMHC" not in stderr

    def test_uniform_and_inner_circle_both_run(self, tmp_path):
        """Both pmhc_mode options should complete successfully."""
        for mode in ["uniform", "inner_circle"]:
            data, _ = _run(
                tmp_path, label=f"pmhc_{mode}",
                pmhc_mode=mode, n_steps=50,
            )
            assert data["depletion_width_nm"] >= 0


# ---- Gaussian vs forced binding dynamics ----------------------------------


class TestGaussianVsForcedDynamics:
    """Gaussian and forced binding should produce different dynamics."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_different_accept_rates(self, tmp_path):
        """Gaussian and forced binding should have noticeably different accept rates."""
        common = dict(
            n_pmhc=30, pmhc_mode="inner_circle", pmhc_seed=1,
            n_steps=200, rigidity_kT=20.0, seed=42,
        )
        d_gauss, _ = _run(
            tmp_path, label="dyn_gauss",
            binding_mode="gaussian", **common,
        )
        d_forced, _ = _run(
            tmp_path, label="dyn_forced",
            binding_mode="forced", **common,
        )
        ar_gauss = d_gauss["diagnostics"]["accept_rate"]
        ar_forced = d_forced["diagnostics"]["accept_rate"]
        # They should not be identical — different binding physics
        assert ar_gauss != pytest.approx(ar_forced, abs=0.01), (
            f"Gaussian ({ar_gauss:.4f}) and forced ({ar_forced:.4f}) accept rates "
            f"should differ by more than 1%"
        )

    def test_both_produce_segregation(self, tmp_path):
        """Both modes should produce meaningful segregation at moderate rigidity."""
        for mode in ["gaussian", "forced"]:
            data, _ = _run(
                tmp_path, label=f"seg_{mode}",
                binding_mode=mode,
                n_pmhc=50, pmhc_mode="inner_circle", pmhc_seed=1,
                n_steps=200, rigidity_kT=30.0,
            )
            diag = data["diagnostics"]
            assert diag["final_tcr_mean_r_nm"] < diag["final_cd45_mean_r_nm"], (
                f"{mode}: TCR ({diag['final_tcr_mean_r_nm']:.1f}) should be inside "
                f"CD45 ({diag['final_cd45_mean_r_nm']:.1f})"
            )

    def test_gaussian_higher_accept_rate_with_pmhc(self, tmp_path):
        """Gaussian binding should generally have higher accept rate than forced
        because forced binding locks TCRs rigidly to pMHC sites."""
        rates_gauss = []
        rates_forced = []
        for seed in range(5):
            common = dict(
                n_pmhc=30, pmhc_mode="inner_circle", pmhc_seed=seed,
                n_steps=100, rigidity_kT=20.0, seed=seed,
            )
            dg, _ = _run(tmp_path, label=f"cmp_g_{seed}", binding_mode="gaussian", **common)
            df, _ = _run(tmp_path, label=f"cmp_f_{seed}", binding_mode="forced", **common)
            rates_gauss.append(dg["diagnostics"]["accept_rate"])
            rates_forced.append(df["diagnostics"]["accept_rate"])

        # On average, gaussian should have higher accept rate
        mean_gauss = np.mean(rates_gauss)
        mean_forced = np.mean(rates_forced)
        assert mean_gauss > mean_forced, (
            f"Expected gaussian accept rate ({mean_gauss:.4f}) > "
            f"forced ({mean_forced:.4f})"
        )
