"""Physics consistency tests: compare C CPU and C GPU height fields."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"

GRID_SIZE = 100
RIGIDITY = 50.0
TIME_SEC = 20.0
N_STEPS = 100
N_TCR = 125
N_CD45 = 500


def _ensure_binary():
    if _BINARY.exists():
        return
    result = subprocess.run(
        ["make"], cwd=str(_PKG_DIR),
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to build binary: {result.stderr}")


def _run_c(tmp_path, seed, use_gpu=False, label="c"):
    """Run C model with frame dumps, return final h and molecule positions."""
    rd = tmp_path / f"{label}_seed{seed}"
    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT", str(RIGIDITY),
        "--seed", str(seed),
        "--n_steps", str(N_STEPS),
        "--grid_size", str(GRID_SIZE),
        "--run-dir", str(rd),
        "--binding_mode", "forced",
        "--n_pmhc", "-1",
        "--dump-frames",
    ]
    if not use_gpu:
        cmd.append("--no-gpu")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr

    frames_dir = rd / "frames"
    h = np.fromfile(frames_dir / f"h_{N_STEPS:05d}.bin", dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE)
    mol = np.fromfile(frames_dir / f"mol_{N_STEPS:05d}.bin", dtype=np.float64)
    tcr = mol[: N_TCR * 2].reshape(N_TCR, 2)
    cd45 = mol[N_TCR * 2:].reshape(N_CD45, 2)

    output = json.loads(result.stdout.strip())
    return h, tcr, cd45, output


def _center_edge_means(h):
    """Return (center_mean, edge_mean) for the height field."""
    n = h.shape[0]
    c = n // 2
    r = n // 8
    Y, X = np.ogrid[:n, :n]
    mask = ((X - c) ** 2 + (Y - c) ** 2) <= r ** 2
    return float(h[mask].mean()), float(h[~mask].mean())


class TestHeightFieldStatistics:
    """Verify that CPU, GPU, and Python produce physically consistent height fields."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_center_depressed(self, tmp_path):
        """Center of the membrane should be depressed (tight contact) in both modes."""
        cpu_h, _, _, _ = _run_c(tmp_path, seed=42, use_gpu=False, label="cpu")
        gpu_h, _, _, _ = _run_c(tmp_path, seed=42, use_gpu=True, label="gpu")

        for name, h in [("C CPU", cpu_h), ("C GPU", gpu_h)]:
            center_mean, edge_mean = _center_edge_means(h)
            assert center_mean < 30.0, f"{name}: center too high ({center_mean:.1f}nm)"
            assert edge_mean > 30.0, f"{name}: edge too low ({edge_mean:.1f}nm)"
            assert edge_mean - center_mean > 10.0, (
                f"{name}: insufficient contrast (center={center_mean:.1f}, edge={edge_mean:.1f})"
            )

    def test_height_range_physical(self, tmp_path):
        """Height field should stay within physical bounds (0 to ~50nm)."""
        cpu_h, _, _, _ = _run_c(tmp_path, seed=7, use_gpu=False, label="cpu")
        gpu_h, _, _, _ = _run_c(tmp_path, seed=7, use_gpu=True, label="gpu")

        for name, h in [("C CPU", cpu_h), ("C GPU", gpu_h)]:
            assert h.min() >= 0.0, f"{name}: negative height {h.min()}"
            assert h.max() < 100.0, f"{name}: unreasonably high {h.max()}"
            assert 20.0 < h.mean() < 80.0, f"{name}: mean out of range {h.mean():.1f}"

    def test_thermal_fluctuations_present(self, tmp_path):
        """Edge region should show thermal fluctuations (std > 0 for non-depressed cells)."""
        cpu_h, _, _, _ = _run_c(tmp_path, seed=99, use_gpu=False, label="cpu")
        gpu_h, _, _, _ = _run_c(tmp_path, seed=99, use_gpu=True, label="gpu")

        n = GRID_SIZE
        c = n // 2
        r = n // 4  # exclude wider center region
        Y, X = np.ogrid[:n, :n]
        edge_mask = ((X - c) ** 2 + (Y - c) ** 2) > r ** 2

        for name, h in [("C CPU", cpu_h), ("C GPU", gpu_h)]:
            edge_std = h[edge_mask].std()
            assert edge_std > 0.5, f"{name}: edge std too low ({edge_std:.2f}) — membrane frozen?"


class TestCpuGpuConsistency:
    """Statistical tests that CPU and GPU produce distributions from the same physics."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_height_distribution_consistent(self, tmp_path):
        """CPU vs GPU height distributions should be in the same physical range.

        CPU and GPU use different RNGs (PCG64 vs Philox) and different float32
        rounding, so exact distributional match is not expected. We check that
        the mean and std are within 15% of each other (same physics, different
        numerical paths).
        """
        n_seeds = 10
        cpu_means = []
        gpu_means = []
        cpu_stds = []
        gpu_stds = []

        for seed in range(n_seeds):
            cpu_h, _, _, _ = _run_c(tmp_path, seed=seed, use_gpu=False, label=f"cpu{seed}")
            gpu_h, _, _, _ = _run_c(tmp_path, seed=seed, use_gpu=True, label=f"gpu{seed}")
            cpu_means.append(cpu_h.mean())
            gpu_means.append(gpu_h.mean())
            cpu_stds.append(cpu_h.std())
            gpu_stds.append(gpu_h.std())

        cpu_mean_avg = np.mean(cpu_means)
        gpu_mean_avg = np.mean(gpu_means)
        assert abs(cpu_mean_avg - gpu_mean_avg) / max(cpu_mean_avg, gpu_mean_avg) < 0.15, (
            f"Mean height differs >15%: CPU={cpu_mean_avg:.2f}, GPU={gpu_mean_avg:.2f}"
        )

        cpu_std_avg = np.mean(cpu_stds)
        gpu_std_avg = np.mean(gpu_stds)
        assert abs(cpu_std_avg - gpu_std_avg) / max(cpu_std_avg, gpu_std_avg) < 0.15, (
            f"Height std differs >15%: CPU={cpu_std_avg:.2f}, GPU={gpu_std_avg:.2f}"
        )

    def test_accept_rate_consistent(self, tmp_path):
        """Accept rates should be in the same range for CPU and GPU."""
        n_seeds = 10
        cpu_rates = []
        gpu_rates = []

        for seed in range(n_seeds):
            _, _, _, cpu_out = _run_c(tmp_path, seed=seed, use_gpu=False, label=f"cpur{seed}")
            _, _, _, gpu_out = _run_c(tmp_path, seed=seed, use_gpu=True, label=f"gpur{seed}")
            cpu_rates.append(cpu_out["diagnostics"]["accept_rate"])
            gpu_rates.append(gpu_out["diagnostics"]["accept_rate"])

        # Accept rates should be similar (within 10% absolute)
        cpu_mean = np.mean(cpu_rates)
        gpu_mean = np.mean(gpu_rates)
        assert abs(cpu_mean - gpu_mean) < 0.10, (
            f"Accept rates differ: CPU={cpu_mean:.4f}, GPU={gpu_mean:.4f}"
        )


class TestAcceptRateGap:
    """Verify that CPU vs GPU acceptance rate gap stays bounded and does not amplify."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    @staticmethod
    def _get_accept_rate(binary, tmp_path, seed, n_steps, grid_size, use_gpu):
        label = "gpu" if use_gpu else "cpu"
        rd = tmp_path / f"gap_{label}_{n_steps}_{seed}"
        cmd = [
            str(binary),
            "--time_sec", "10.0",
            "--rigidity_kT", "20.0",
            "--seed", str(seed),
            "--n_steps", str(n_steps),
            "--grid_size", str(grid_size),
            "--run-dir", str(rd),
            "--binding_mode", "forced",
            "--n_pmhc", "-1",
        ]
        if not use_gpu:
            cmd.append("--no-gpu")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip())["diagnostics"]["accept_rate"]

    def test_gap_bounded_short(self, tmp_path):
        """CPU vs GPU accept rate gap < 2% at 50 steps across 5 seeds."""
        gaps = []
        for seed in range(5):
            cpu = self._get_accept_rate(_BINARY, tmp_path, seed, 50, 50, False)
            gpu = self._get_accept_rate(_BINARY, tmp_path, seed, 50, 50, True)
            gaps.append(abs(gpu - cpu))
        mean_gap = np.mean(gaps)
        assert mean_gap < 0.02, f"Mean gap {mean_gap:.4f} exceeds 2% at 50 steps"

    def test_gap_bounded_medium(self, tmp_path):
        """CPU vs GPU accept rate gap < 2% at 500 steps across 5 seeds."""
        gaps = []
        for seed in range(5):
            cpu = self._get_accept_rate(_BINARY, tmp_path, seed, 500, 50, False)
            gpu = self._get_accept_rate(_BINARY, tmp_path, seed, 500, 50, True)
            gaps.append(abs(gpu - cpu))
        mean_gap = np.mean(gaps)
        assert mean_gap < 0.02, f"Mean gap {mean_gap:.4f} exceeds 2% at 500 steps"

    @pytest.mark.slow
    def test_gap_no_amplification(self, tmp_path):
        """Gap does not amplify: < 2% at 5000 steps across 3 seeds."""
        gaps = []
        for seed in range(3):
            cpu = self._get_accept_rate(_BINARY, tmp_path, seed, 5000, 50, False)
            gpu = self._get_accept_rate(_BINARY, tmp_path, seed, 5000, 50, True)
            gaps.append(abs(gpu - cpu))
        mean_gap = np.mean(gaps)
        assert mean_gap < 0.02, f"Mean gap {mean_gap:.4f} exceeds 2% at 5000 steps"


@pytest.mark.slow
class TestGridConvergence:
    """Verify that different grid sizes with same physical time produce similar physics."""

    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_grid_convergence_depletion(self, tmp_path):
        """Finer grid with same time_sec converges to similar depletion physics."""
        n_seeds = 10

        widths_32 = []
        widths_64 = []

        for seed in range(n_seeds):
            for grid, widths, label in [(32, widths_32, "g32"), (64, widths_64, "g64")]:
                rd = tmp_path / f"{label}_seed{seed}"
                cmd = [
                    str(_BINARY),
                    "--time_sec", "10.0",
                    "--rigidity_kT", "30.0",
                    "--seed", str(seed),
                    "--grid_size", str(grid),
                    "--run-dir", str(rd),
                    "--binding_mode", "forced",
                    "--n_pmhc", "-1",
                    "--no-gpu",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                assert result.returncode == 0, result.stderr
                output = json.loads(result.stdout.strip())
                widths.append(output["depletion_width_nm"])

        # Both grids should produce depletion widths from the same distribution
        stat, p = stats.ks_2samp(widths_32, widths_64)
        assert p > 0.01, (
            f"Grid convergence failed: KS p={p:.4f}\n"
            f"Grid 32: {[f'{x:.1f}' for x in widths_32]}\n"
            f"Grid 64: {[f'{x:.1f}' for x in widths_64]}"
        )


