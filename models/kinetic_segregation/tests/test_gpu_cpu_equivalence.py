"""Regression & equivalence tests for GPU optimization work.

Tests verify that:
A) GPU results match pre-optimization baselines (GPU regression)
B) CPU results match pre-optimization baselines (CPU regression)
C) GPU vs CPU cross-mode equivalence
D) Determinism within each mode (same seed → same output)
E) Conservation / sanity checks
F) Substep equivalence (Tier 2 only, when --grid-substeps is available)

Reference values in reference_values.json were recorded BEFORE any optimization.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"
_REF_FILE = Path(__file__).resolve().parent / "reference_values.json"
PATCH_SIZE_NM = 2000.0


def _ensure_binary():
    if _BINARY.exists():
        return
    result = subprocess.run(
        ["make"], cwd=str(_PKG_DIR),
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to build binary: {result.stderr}")


def _load_reference():
    with open(_REF_FILE) as f:
        return json.load(f)


def _run(tmp_path, *, use_gpu=True, label="run", seed=42, time_sec=5.0,
         rigidity=20.0, grid_size=64, n_steps=500, n_tcr=125, n_cd45=500,
         extra_args=None):
    """Run the C binary and return parsed JSON output."""
    rd = tmp_path / label
    cmd = [
        str(_BINARY),
        "--time_sec", str(time_sec),
        "--rigidity_kT", str(rigidity),
        "--seed", str(seed),
        "--n_steps", str(n_steps),
        "--grid_size", str(grid_size),
        "--n_tcr", str(n_tcr),
        "--n_cd45", str(n_cd45),
        "--run-dir", str(rd),
    ]
    if not use_gpu:
        cmd.append("--no-gpu")
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"Binary failed: {result.stderr}"
    return json.loads(result.stdout.strip())


def _run_with_frames(tmp_path, *, use_gpu=True, label="run", seed=42,
                     time_sec=5.0, rigidity=20.0, grid_size=64, n_steps=500,
                     n_tcr=125, n_cd45=500, extra_args=None):
    """Run with --dump-frames and return (json_output, final_h, tcr_pos, cd45_pos)."""
    rd = tmp_path / label
    cmd = [
        str(_BINARY),
        "--time_sec", str(time_sec),
        "--rigidity_kT", str(rigidity),
        "--seed", str(seed),
        "--n_steps", str(n_steps),
        "--grid_size", str(grid_size),
        "--n_tcr", str(n_tcr),
        "--n_cd45", str(n_cd45),
        "--run-dir", str(rd),
        "--dump-frames",
    ]
    if not use_gpu:
        cmd.append("--no-gpu")
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"Binary failed: {result.stderr}"
    output = json.loads(result.stdout.strip())

    frames_dir = rd / "frames"
    h = np.fromfile(frames_dir / f"h_{n_steps:05d}.bin", dtype=np.float32)
    h = h.reshape(grid_size, grid_size)
    mol = np.fromfile(frames_dir / f"mol_{n_steps:05d}.bin", dtype=np.float64)
    tcr = mol[:n_tcr * 2].reshape(n_tcr, 2)
    cd45 = mol[n_tcr * 2:].reshape(n_cd45, 2)
    return output, h, tcr, cd45


def _mean_radial(pos):
    """Mean radial distance from patch center."""
    center = PATCH_SIZE_NM / 2.0
    r = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    return float(np.mean(r))


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="module")
def _build():
    _ensure_binary()
    if not _BINARY.exists():
        pytest.skip("Binary not available")


# ─── A) GPU Regression ─────────────────────────────────────────────────────

class TestGPURegression:
    """GPU after optimization must match GPU baseline from reference_values.json."""

    def test_gpu_depletion_width(self, tmp_path):
        ref = _load_reference()
        out = _run(tmp_path, use_gpu=True, label="gpu_reg")
        expected = ref["gpu"]["depletion_width_nm"]
        actual = out["depletion_width_nm"]
        assert abs(actual - expected) / expected < 0.02, (
            f"GPU depletion width changed: {expected:.3f} → {actual:.3f}"
        )

    def test_gpu_accept_rate(self, tmp_path):
        ref = _load_reference()
        out = _run(tmp_path, use_gpu=True, label="gpu_ar")
        expected = ref["gpu"]["accept_rate"]
        actual = out["diagnostics"]["accept_rate"]
        assert abs(actual - expected) / expected < 0.02, (
            f"GPU accept rate changed: {expected:.4f} → {actual:.4f}"
        )

    def test_gpu_radial_positions(self, tmp_path):
        ref = _load_reference()
        _, _, tcr, cd45 = _run_with_frames(tmp_path, use_gpu=True, label="gpu_rad")
        tcr_r = _mean_radial(tcr)
        cd45_r = _mean_radial(cd45)
        assert abs(tcr_r - ref["gpu"]["final_tcr_mean_r_nm"]) / ref["gpu"]["final_tcr_mean_r_nm"] < 0.05, (
            f"GPU TCR radial: {ref['gpu']['final_tcr_mean_r_nm']:.1f} → {tcr_r:.1f}"
        )
        assert abs(cd45_r - ref["gpu"]["final_cd45_mean_r_nm"]) / ref["gpu"]["final_cd45_mean_r_nm"] < 0.05, (
            f"GPU CD45 radial: {ref['gpu']['final_cd45_mean_r_nm']:.1f} → {cd45_r:.1f}"
        )


# ─── B) CPU Regression ─────────────────────────────────────────────────────

class TestCPURegression:
    """CPU after optimization must match CPU baseline from reference_values.json."""

    def test_cpu_depletion_width(self, tmp_path):
        ref = _load_reference()
        out = _run(tmp_path, use_gpu=False, label="cpu_reg")
        expected = ref["cpu"]["depletion_width_nm"]
        actual = out["depletion_width_nm"]
        assert abs(actual - expected) / expected < 0.02, (
            f"CPU depletion width changed: {expected:.3f} → {actual:.3f}"
        )

    def test_cpu_accept_rate(self, tmp_path):
        ref = _load_reference()
        out = _run(tmp_path, use_gpu=False, label="cpu_ar")
        expected = ref["cpu"]["accept_rate"]
        actual = out["diagnostics"]["accept_rate"]
        assert abs(actual - expected) / expected < 0.02, (
            f"CPU accept rate changed: {expected:.4f} → {actual:.4f}"
        )

    def test_cpu_radial_positions(self, tmp_path):
        ref = _load_reference()
        _, _, tcr, cd45 = _run_with_frames(tmp_path, use_gpu=False, label="cpu_rad")
        tcr_r = _mean_radial(tcr)
        cd45_r = _mean_radial(cd45)
        assert abs(tcr_r - ref["cpu"]["final_tcr_mean_r_nm"]) / ref["cpu"]["final_tcr_mean_r_nm"] < 0.05, (
            f"CPU TCR radial: {ref['cpu']['final_tcr_mean_r_nm']:.1f} → {tcr_r:.1f}"
        )
        assert abs(cd45_r - ref["cpu"]["final_cd45_mean_r_nm"]) / ref["cpu"]["final_cd45_mean_r_nm"] < 0.05, (
            f"CPU CD45 radial: {ref['cpu']['final_cd45_mean_r_nm']:.1f} → {cd45_r:.1f}"
        )


# ─── C) GPU vs CPU Cross-Mode Equivalence ──────────────────────────────────

class TestCrossMode:
    """GPU and CPU produce results from the same physics (relaxed thresholds)."""

    def test_depletion_width_agreement(self, tmp_path):
        gpu = _run(tmp_path, use_gpu=True, label="xm_gpu")
        cpu = _run(tmp_path, use_gpu=False, label="xm_cpu")
        gpu_w = gpu["depletion_width_nm"]
        cpu_w = cpu["depletion_width_nm"]
        mean_w = (gpu_w + cpu_w) / 2.0
        assert abs(gpu_w - cpu_w) / mean_w < 0.50, (
            f"Depletion widths too different: GPU={gpu_w:.1f}, CPU={cpu_w:.1f}"
        )

    def test_accept_rate_agreement(self, tmp_path):
        gpu = _run(tmp_path, use_gpu=True, label="ar_gpu")
        cpu = _run(tmp_path, use_gpu=False, label="ar_cpu")
        assert abs(gpu["diagnostics"]["accept_rate"] - cpu["diagnostics"]["accept_rate"]) < 0.05, (
            f"Accept rates differ: GPU={gpu['diagnostics']['accept_rate']:.4f}, "
            f"CPU={cpu['diagnostics']['accept_rate']:.4f}"
        )

    def test_height_field_stats(self, tmp_path):
        _, gpu_h, _, _ = _run_with_frames(tmp_path, use_gpu=True, label="hf_gpu")
        _, cpu_h, _, _ = _run_with_frames(tmp_path, use_gpu=False, label="hf_cpu")
        # Mean and std should be within 15%
        gpu_mean, cpu_mean = gpu_h.mean(), cpu_h.mean()
        gpu_std, cpu_std = gpu_h.std(), cpu_h.std()
        assert abs(gpu_mean - cpu_mean) / max(gpu_mean, cpu_mean) < 0.15, (
            f"Height mean: GPU={gpu_mean:.2f}, CPU={cpu_mean:.2f}"
        )
        assert abs(gpu_std - cpu_std) / max(gpu_std, cpu_std) < 0.15, (
            f"Height std: GPU={gpu_std:.2f}, CPU={cpu_std:.2f}"
        )


# ─── D) Determinism ────────────────────────────────────────────────────────

@pytest.mark.deterministic
class TestDeterminism:
    """Same seed must produce reproducible output.

    CPU path is bit-exact deterministic.
    GPU path may have minor accept-counter jitter due to Metal atomic scheduling,
    so we check statistical agreement (depletion width within 5%, accept rate within 1%).
    """

    def test_gpu_deterministic(self, tmp_path):
        out1 = _run(tmp_path, use_gpu=True, label="det_gpu1", seed=77)
        out2 = _run(tmp_path, use_gpu=True, label="det_gpu2", seed=77)
        # GPU atomics may cause minor accept-rate jitter; check statistical agreement
        dw1 = out1["depletion_width_nm"]
        dw2 = out2["depletion_width_nm"]
        mean_dw = (dw1 + dw2) / 2.0
        assert abs(dw1 - dw2) / mean_dw < 0.05, (
            f"GPU depletion widths too different: {dw1:.2f} vs {dw2:.2f}"
        )
        ar1 = out1["diagnostics"]["accept_rate"]
        ar2 = out2["diagnostics"]["accept_rate"]
        assert abs(ar1 - ar2) < 0.01, (
            f"GPU accept rates too different: {ar1:.6f} vs {ar2:.6f}"
        )

    def test_cpu_deterministic(self, tmp_path):
        out1 = _run(tmp_path, use_gpu=False, label="det_cpu1", seed=77)
        out2 = _run(tmp_path, use_gpu=False, label="det_cpu2", seed=77)
        assert out1 == out2, "CPU not deterministic with same seed"

    def test_gpu_frames_deterministic(self, tmp_path):
        _, h1, tcr1, cd451 = _run_with_frames(
            tmp_path, use_gpu=True, label="detf_gpu1", seed=77
        )
        _, h2, tcr2, cd452 = _run_with_frames(
            tmp_path, use_gpu=True, label="detf_gpu2", seed=77
        )
        # GPU height fields should be statistically close (not necessarily bit-identical).
        # Metal GPU atomic scheduling can cause ~0.3% of cells to diverge by up to ~30nm.
        # Brownian mode's larger step_size_h widens the divergence envelope.
        np.testing.assert_allclose(
            h1, h2, rtol=0.25, atol=35.0,
            err_msg="GPU height fields too different between runs"
        )
        # Molecule positions depend on h[] via MC energy, so can diverge more.
        # Check mean radial distance (a bulk statistic) rather than per-molecule.
        r_tcr1 = np.sqrt(np.sum((tcr1 - 1000.0) ** 2, axis=1)).mean()
        r_tcr2 = np.sqrt(np.sum((tcr2 - 1000.0) ** 2, axis=1)).mean()
        assert abs(r_tcr1 - r_tcr2) / ((r_tcr1 + r_tcr2) / 2) < 0.10, (
            f"GPU TCR mean radial distance too different: {r_tcr1:.1f} vs {r_tcr2:.1f}"
        )
        r_cd1 = np.sqrt(np.sum((cd451 - 1000.0) ** 2, axis=1)).mean()
        r_cd2 = np.sqrt(np.sum((cd452 - 1000.0) ** 2, axis=1)).mean()
        assert abs(r_cd1 - r_cd2) / ((r_cd1 + r_cd2) / 2) < 0.10, (
            f"GPU CD45 mean radial distance too different: {r_cd1:.1f} vs {r_cd2:.1f}"
        )

    def test_cpu_frames_deterministic(self, tmp_path):
        _, h1, tcr1, cd451 = _run_with_frames(
            tmp_path, use_gpu=False, label="detf_cpu1", seed=77
        )
        _, h2, tcr2, cd452 = _run_with_frames(
            tmp_path, use_gpu=False, label="detf_cpu2", seed=77
        )
        np.testing.assert_array_equal(h1, h2, err_msg="CPU height not bit-identical")
        np.testing.assert_array_equal(tcr1, tcr2, err_msg="CPU TCR pos not bit-identical")
        np.testing.assert_array_equal(cd451, cd452, err_msg="CPU CD45 pos not bit-identical")


# ─── E) Conservation / Sanity Checks ───────────────────────────────────────

class TestConservation:
    """Basic physical constraints that must always hold."""

    @pytest.mark.parametrize("use_gpu", [True, False], ids=["gpu", "cpu"])
    def test_molecule_count_preserved(self, tmp_path, use_gpu):
        label = "cons_gpu" if use_gpu else "cons_cpu"
        _, _, tcr, cd45 = _run_with_frames(tmp_path, use_gpu=use_gpu, label=label)
        assert tcr.shape == (125, 2), f"TCR count changed: {tcr.shape}"
        assert cd45.shape == (500, 2), f"CD45 count changed: {cd45.shape}"

    @pytest.mark.parametrize("use_gpu", [True, False], ids=["gpu", "cpu"])
    def test_positions_in_bounds(self, tmp_path, use_gpu):
        label = "bnd_gpu" if use_gpu else "bnd_cpu"
        _, _, tcr, cd45 = _run_with_frames(tmp_path, use_gpu=use_gpu, label=label)
        for name, pos in [("TCR", tcr), ("CD45", cd45)]:
            assert pos.min() >= 0, f"{name} position < 0: {pos.min()}"
            assert pos.max() <= PATCH_SIZE_NM, f"{name} position > {PATCH_SIZE_NM}: {pos.max()}"

    @pytest.mark.parametrize("use_gpu", [True, False], ids=["gpu", "cpu"])
    def test_height_physical(self, tmp_path, use_gpu):
        label = "hph_gpu" if use_gpu else "hph_cpu"
        _, h, _, _ = _run_with_frames(tmp_path, use_gpu=use_gpu, label=label)
        assert h.shape == (64, 64), f"Grid shape changed: {h.shape}"
        assert h.min() >= 0, f"Negative height: {h.min()}"
        assert h.max() < 200, f"Unreasonable height: {h.max()}"
        assert 10 < h.mean() < 80, f"Mean height out of range: {h.mean():.1f}"


# ─── F) Substep Equivalence (Tier 2) ───────────────────────────────────────

def _binary_supports_substeps():
    """Check if binary supports --grid-substeps flag."""
    result = subprocess.run(
        [str(_BINARY), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    return "--grid-substeps" in result.stdout or "--grid-substeps" in result.stderr


class TestSubstepEquivalence:
    """Substep batching works and produces reasonable physics.

    Note: with substeps=K, each sim_step does 1 molecular move + K grid updates.
    So 100 steps × substeps=5 has 100 molecular moves (not 500), while
    500 steps × substeps=1 has 500. They are physically different simulations.
    These tests verify substeps don't crash and produce reasonable physics,
    not that they match single-step exactly.
    """

    @pytest.fixture(autouse=True)
    def _check_substeps(self):
        if not _binary_supports_substeps():
            pytest.skip("Binary does not support --grid-substeps yet")

    @pytest.mark.parametrize("use_gpu", [True, False], ids=["gpu", "cpu"])
    def test_substep_runs_and_produces_depletion(self, tmp_path, use_gpu):
        """Substep=5 produces positive depletion width (physics working)."""
        mode = "gpu" if use_gpu else "cpu"
        out = _run(tmp_path, use_gpu=use_gpu, label=f"sub5_{mode}",
                   n_steps=100, extra_args=["--grid-substeps", "5"])
        w = out["depletion_width_nm"]
        assert w > 0, f"{mode.upper()} substep=5 depletion width is 0"
        assert w < 1500, f"{mode.upper()} substep=5 depletion unreasonably large: {w}"

    @pytest.mark.parametrize("use_gpu", [True, False], ids=["gpu", "cpu"])
    def test_substep_accept_rate_reasonable(self, tmp_path, use_gpu):
        """Substep=5 accept rate is in a reasonable range."""
        mode = "gpu" if use_gpu else "cpu"
        out = _run(tmp_path, use_gpu=use_gpu, label=f"subar5_{mode}",
                   n_steps=100, extra_args=["--grid-substeps", "5"])
        r = out["diagnostics"]["accept_rate"]
        assert 0.1 < r < 0.99, (
            f"{mode.upper()} substep=5 accept rate out of range: {r:.4f}"
        )
