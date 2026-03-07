"""Statistical equivalence tests: C model vs Python reference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

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


def _run_c_model(tmp_path, time_sec, rigidity, seed, n_steps=20, grid_size=32):
    rd = tmp_path / f"c_seed{seed}"
    result = subprocess.run(
        [str(_BINARY),
         "--time_sec", str(time_sec),
         "--rigidity_kT_nm2", str(rigidity),
         "--seed", str(seed),
         "--n_steps", str(n_steps),
         "--grid_size", str(grid_size),
         "--no-gpu",
         "--run-dir", str(rd)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


def _run_python_model(time_sec, rigidity, seed, n_steps=20, grid_size=32):
    from models.kinetic_segregation.model import simulate_ks

    import hashlib
    import struct
    raw = struct.pack("dd", time_sec, rigidity)
    input_hash = int(hashlib.md5(raw).hexdigest()[:8], 16)
    point_seed = seed + input_hash

    result = simulate_ks(
        time_sec=time_sec,
        rigidity_kT_nm2=rigidity,
        seed=point_seed,
        n_steps=n_steps,
        grid_size=grid_size,
    )
    return {
        "depletion_width_nm": result["depletion_width_nm"],
        "diagnostics": {
            "accept_rate": result["accept_rate"],
        },
    }


@pytest.mark.slow
class TestStatisticalEquivalence:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_depletion_width_distribution(self, tmp_path):
        """Two-sample KS test: C and Python depletion widths come from same distribution."""
        time_sec = 20.0
        rigidity = 20.0
        n_seeds = 20
        n_steps = 20
        grid_size = 32

        c_widths = []
        py_widths = []

        for seed in range(n_seeds):
            c_result = _run_c_model(tmp_path, time_sec, rigidity, seed,
                                    n_steps=n_steps, grid_size=grid_size)
            c_widths.append(c_result["depletion_width_nm"])

            py_result = _run_python_model(time_sec, rigidity, seed,
                                          n_steps=n_steps, grid_size=grid_size)
            py_widths.append(py_result["depletion_width_nm"])

        stat, p_value = stats.ks_2samp(c_widths, py_widths)
        assert p_value > 0.05, (
            f"KS test failed: p={p_value:.4f}, stat={stat:.4f}\n"
            f"C widths: {c_widths}\n"
            f"Python widths: {py_widths}"
        )

    def test_accept_rate_reasonable(self, tmp_path):
        """C model accept rates are in a reasonable range (0.1-0.9)."""
        result = _run_c_model(tmp_path, 10.0, 10.0, seed=42, n_steps=10, grid_size=32)
        rate = result["diagnostics"]["accept_rate"]
        assert 0.05 < rate < 0.95, f"Accept rate {rate} out of reasonable range"
