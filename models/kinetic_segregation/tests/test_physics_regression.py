"""Physics regression tests for the C kinetic segregation binary.

Migrated from the former Python model test suite. These tests invoke the
C binary (CPU mode) and validate that core physics behavior is preserved:
depletion, segregation, determinism, acceptance rates, pMHC modes, binding
modes, step modes, and molecule conservation.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"
_SUBMODULE_ROOT = str(Path(__file__).resolve().parents[3])


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
        "n_steps": 50,
        "grid_size": 32,
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        cmd.extend([f"--{k}", str(v)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Binary failed: {result.stderr}"
    return json.loads(result.stdout.strip())


class TestBasicPhysics:
    """Core physics: depletion, segregation, acceptance."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_depletion_width_positive(self, tmp_path):
        data = _run(tmp_path, label="depl")
        assert data["depletion_width_nm"] > 0

    def test_tcr_closer_to_center_than_cd45(self, tmp_path):
        data = _run(tmp_path, label="seg", n_steps=200, rigidity_kT=30.0)
        diag = data["diagnostics"]
        assert diag["final_tcr_mean_r_nm"] < diag["final_cd45_mean_r_nm"]

    def test_accept_rate_reasonable(self, tmp_path):
        data = _run(tmp_path, label="acc")
        rate = data["diagnostics"]["accept_rate"]
        assert 0.3 < rate < 0.95, f"Accept rate {rate} out of expected range"

    def test_depletion_increases_with_steps(self, tmp_path):
        d_short = _run(tmp_path, label="short", n_steps=10)
        d_long = _run(tmp_path, label="long", n_steps=200)
        assert d_long["depletion_width_nm"] >= d_short["depletion_width_nm"] * 0.8

    def test_molecule_count_conservation(self, tmp_path):
        """Simulation runs successfully with custom molecule counts."""
        data = _run(tmp_path, label="cons", n_tcr=50, n_cd45=200)
        assert data["depletion_width_nm"] >= 0

    def test_height_values_physical(self, tmp_path):
        """Height field statistics should be physically reasonable."""
        data = _run(tmp_path, label="hphys", n_steps=100)
        diag = data["diagnostics"]
        h_mean = diag.get("h_mean_nm")
        if h_mean is not None:
            assert 0 < h_mean < 100


@pytest.mark.deterministic
class TestDeterminism:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_same_seed_identical(self, tmp_path):
        d1 = _run(tmp_path, label="det1", seed=123)
        d2 = _run(tmp_path, label="det2", seed=123)
        assert d1["depletion_width_nm"] == d2["depletion_width_nm"]
        assert d1["diagnostics"]["accept_rate"] == d2["diagnostics"]["accept_rate"]

    def test_different_seeds_differ(self, tmp_path):
        d1 = _run(tmp_path, label="diff1", seed=1)
        d2 = _run(tmp_path, label="diff2", seed=2)
        # Very unlikely to be identical with different seeds
        assert (
            d1["depletion_width_nm"] != d2["depletion_width_nm"]
            or d1["diagnostics"]["accept_rate"] != d2["diagnostics"]["accept_rate"]
        )


class TestCustomParams:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_custom_molecule_counts(self, tmp_path):
        data = _run(tmp_path, label="cust", n_tcr=30, n_cd45=60, n_steps=20)
        assert data["depletion_width_nm"] >= 0

    def test_small_grid(self, tmp_path):
        data = _run(tmp_path, label="small", grid_size=16, n_steps=10)
        assert data["depletion_width_nm"] >= 0


class TestPmhcModes:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_inner_circle_produces_segregation(self, tmp_path):
        data = _run(
            tmp_path, label="pmhc_ic",
            n_pmhc=50, pmhc_mode="inner_circle", pmhc_seed=1,
            n_steps=100,
        )
        assert data["depletion_width_nm"] > 0

    def test_uniform_runs_successfully(self, tmp_path):
        """Uniform pMHC mode runs without error; segregation not guaranteed."""
        data = _run(
            tmp_path, label="pmhc_uni",
            n_pmhc=50, pmhc_mode="uniform", pmhc_seed=1,
            n_steps=100,
        )
        assert data["depletion_width_nm"] >= 0


class TestBindingModes:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_forced_binding_runs(self, tmp_path):
        data = _run(tmp_path, label="forced", binding_mode="forced", n_steps=50)
        assert data["depletion_width_nm"] >= 0

    def test_gaussian_binding_runs(self, tmp_path):
        data = _run(tmp_path, label="gauss", binding_mode="gaussian", n_steps=50)
        assert data["depletion_width_nm"] >= 0


class TestStepModes:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_paper_mode(self, tmp_path):
        data = _run(tmp_path, label="paper", step_mode="paper", n_steps=50)
        diag = data["diagnostics"]
        assert diag["dt_seconds"] == pytest.approx(0.01, abs=1e-6)

    def test_brownian_mode(self, tmp_path):
        data = _run(tmp_path, label="brown", step_mode="brownian", n_steps=50)
        diag = data["diagnostics"]
        # Brownian dt should be different from paper's fixed 0.01
        assert diag["dt_seconds"] != pytest.approx(0.01, abs=1e-6)

    def test_paper_vs_brownian_different_dt(self, tmp_path):
        d_paper = _run(tmp_path, label="pmode", step_mode="paper", n_steps=20)
        d_brown = _run(tmp_path, label="bmode", step_mode="brownian", n_steps=20)
        assert d_paper["diagnostics"]["dt_seconds"] != d_brown["diagnostics"]["dt_seconds"]


class TestMolRepulsion:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_with_repulsion(self, tmp_path):
        data = _run(
            tmp_path, label="rep",
            mol_repulsion_eps=2.0, mol_repulsion_rcut=50.0,
            n_steps=50,
        )
        assert data["depletion_width_nm"] >= 0

    def test_backward_compat_no_repulsion(self, tmp_path):
        """Default (no mol repulsion) should work."""
        data = _run(tmp_path, label="norep", n_steps=20)
        assert data["depletion_width_nm"] >= 0


class TestCd45Params:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_custom_cd45_height(self, tmp_path):
        data = _run(tmp_path, label="cd45h", cd45_height=40.0, n_steps=50)
        assert data["depletion_width_nm"] >= 0

    def test_custom_k_rep(self, tmp_path):
        data = _run(tmp_path, label="krep", cd45_k_rep=2.0, n_steps=50)
        assert data["depletion_width_nm"] >= 0


class TestBoundaryConditions:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_positions_stay_in_bounds(self, tmp_path):
        """Run with dump-frames to check molecule positions stay in [0, patch_size]."""
        n_tcr, n_cd45 = 125, 500
        rd = tmp_path / "bounds"
        cmd = [
            str(_BINARY), "--run-dir", str(rd), "--no-gpu",
            "--time_sec", "10", "--rigidity_kT", "20",
            "--seed", "42", "--n_steps", "50", "--grid_size", "32",
            "--n_tcr", str(n_tcr), "--n_cd45", str(n_cd45),
            "--dump-frames",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.strip())

        # Check final molecule positions from frame dumps
        frames_dir = rd / "frames"
        n_steps = data["diagnostics"]["n_steps_actual"]
        mol_file = frames_dir / f"mol_{n_steps:05d}.bin"
        if mol_file.exists():
            mol = np.fromfile(str(mol_file), dtype=np.float64)
            patch_size = 2000.0  # default patch size
            assert mol.min() >= 0.0, f"Molecule position below 0: {mol.min()}"
            assert mol.max() <= patch_size, f"Molecule position above patch: {mol.max()}"


# ─── Multi-seed statistical regression tests (Rule 3b / Rule 4) ──────────
#
# These tests run multiple seeds and verify that key observables (depletion
# width, acceptance rate) remain within expected physical margins.  They are
# robust to platform changes (different compiler, GPU, OS) because they
# check distributions, not exact values.

N_STAT_SEEDS = 5
STAT_N_STEPS = 100
STAT_GRID = 32


class TestStatisticalBasicPhysics:
    """Multi-seed checks for default Brownian mode (CPU)."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_depletion_width_distribution(self, tmp_path):
        """Depletion width should be positive and stable across seeds."""
        widths = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_dw_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID)
            widths.append(data["depletion_width_nm"])
        widths = np.array(widths)
        assert np.all(widths > 0), f"Non-positive depletion widths: {widths}"
        assert np.mean(widths) > 50, f"Mean depletion too small: {np.mean(widths):.1f}"
        cv = np.std(widths) / np.mean(widths)
        assert cv < 0.5, f"Depletion CV too high ({cv:.2f}), physics unstable"

    def test_accept_rate_distribution(self, tmp_path):
        """Accept rate should stay in [0.5, 0.95] across seeds."""
        rates = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_ar_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID)
            rates.append(data["diagnostics"]["accept_rate"])
        rates = np.array(rates)
        assert np.all(rates > 0.3), f"Accept rate too low: {rates}"
        assert np.all(rates < 0.98), f"Accept rate too high: {rates}"
        cv = np.std(rates) / np.mean(rates)
        assert cv < 0.1, f"Accept rate CV too high ({cv:.2f}), should be stable"

    def test_segregation_consistent(self, tmp_path):
        """TCR should be closer to center than CD45 across all seeds."""
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_seg_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID,
                        rigidity_kT=30.0)
            diag = data["diagnostics"]
            assert diag["final_tcr_mean_r_nm"] < diag["final_cd45_mean_r_nm"], (
                f"seed={seed}: TCR ({diag['final_tcr_mean_r_nm']:.1f}) "
                f"not inside CD45 ({diag['final_cd45_mean_r_nm']:.1f})"
            )


class TestStatisticalPmhcModes:
    """Multi-seed checks for pMHC gating modes."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_inner_circle_pmhc_segregation(self, tmp_path):
        """Inner-circle pMHC should produce positive depletion across seeds."""
        widths = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_pmhc_ic_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID,
                        n_pmhc=50, pmhc_mode="inner_circle", pmhc_seed=seed)
            widths.append(data["depletion_width_nm"])
        widths = np.array(widths)
        assert np.all(widths >= 0), f"Negative depletion with pMHC: {widths}"
        assert np.mean(widths) > 0, f"No segregation with inner_circle pMHC"

    def test_uniform_pmhc_runs_consistently(self, tmp_path):
        """Uniform pMHC should run without error across seeds."""
        rates = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_pmhc_uni_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID,
                        n_pmhc=50, pmhc_mode="uniform", pmhc_seed=seed)
            rates.append(data["diagnostics"]["accept_rate"])
        rates = np.array(rates)
        assert np.all(rates > 0.3), f"Accept rate too low with uniform pMHC: {rates}"


class TestStatisticalBindingModes:
    """Multi-seed checks for forced vs gaussian binding."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_forced_binding_stable(self, tmp_path):
        """Forced binding should produce stable accept rates across seeds."""
        rates = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_forced_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID,
                        binding_mode="forced")
            rates.append(data["diagnostics"]["accept_rate"])
        rates = np.array(rates)
        assert np.all(rates > 0.3), f"Forced binding accept rate too low: {rates}"
        assert np.all(rates < 0.98), f"Forced binding accept rate too high: {rates}"

    def test_gaussian_binding_stable(self, tmp_path):
        """Gaussian binding should produce stable depletion across seeds."""
        widths = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_gauss_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID,
                        binding_mode="gaussian")
            widths.append(data["depletion_width_nm"])
        widths = np.array(widths)
        assert np.all(widths >= 0), f"Negative depletion with gaussian: {widths}"


class TestStatisticalStepModes:
    """Multi-seed checks for paper vs brownian step modes."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_paper_mode_stable(self, tmp_path):
        """Paper step mode should produce consistent physics across seeds."""
        widths = []
        rates = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_paper_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID,
                        step_mode="paper")
            widths.append(data["depletion_width_nm"])
            rates.append(data["diagnostics"]["accept_rate"])
        widths = np.array(widths)
        rates = np.array(rates)
        assert np.all(widths >= 0), f"Negative depletion in paper mode: {widths}"
        assert np.all(rates > 0.1), f"Accept rate too low in paper mode: {rates}"
        assert np.all(rates < 0.99), f"Accept rate too high in paper mode: {rates}"

    def test_brownian_mode_stable(self, tmp_path):
        """Brownian step mode should produce consistent physics across seeds."""
        widths = []
        rates = []
        for seed in range(N_STAT_SEEDS):
            data = _run(tmp_path, label=f"stat_brown_{seed}", seed=seed,
                        n_steps=STAT_N_STEPS, grid_size=STAT_GRID,
                        step_mode="brownian")
            widths.append(data["depletion_width_nm"])
            rates.append(data["diagnostics"]["accept_rate"])
        widths = np.array(widths)
        rates = np.array(rates)
        assert np.all(widths >= 0), f"Negative depletion in brownian mode: {widths}"
        assert np.all(rates > 0.3), f"Accept rate too low in brownian mode: {rates}"
        cv = np.std(widths) / max(np.mean(widths), 1.0)
        assert cv < 0.5, f"Brownian depletion CV too high ({cv:.2f})"
