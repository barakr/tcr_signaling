"""Tests for GPU KS model CLI (binary and Python wrapper)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"
_SUBMODULE_ROOT = str(Path(__file__).resolve().parents[3])


def _ensure_binary():
    """Build binary if not present."""
    if _BINARY.exists():
        return
    result = subprocess.run(
        ["make"], cwd=str(_PKG_DIR),
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to build binary: {result.stderr}")


class TestBinaryCli:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_produces_output(self, tmp_path):
        result = subprocess.run(
            [str(_BINARY),
             "--time_sec", "10",
             "--rigidity_kT_nm2", "10",
             "--seed", "42",
             "--n_steps", "5",
             "--grid_size", "16",
             "--no-gpu",
             "--run-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr

        out_file = tmp_path / "out" / "segregation.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "depletion_width_nm" in data
        assert "inputs" in data
        assert data["inputs"]["time_sec"] == 10.0
        assert data["inputs"]["rigidity_kT_nm2"] == 10.0

    def test_stdout_is_valid_json(self, tmp_path):
        result = subprocess.run(
            [str(_BINARY),
             "--time_sec", "5",
             "--rigidity_kT_nm2", "5",
             "--seed", "1",
             "--n_steps", "3",
             "--grid_size", "16",
             "--no-gpu",
             "--run-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.strip())
        assert isinstance(data["depletion_width_nm"], (int, float))

    def test_missing_required_args(self, tmp_path):
        result = subprocess.run(
            [str(_BINARY), "--run-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0

    def test_deterministic(self, tmp_path):
        """Same seed produces identical output."""
        runs = []
        for i in range(2):
            rd = tmp_path / f"run{i}"
            result = subprocess.run(
                [str(_BINARY),
                 "--time_sec", "10",
                 "--rigidity_kT_nm2", "20",
                 "--seed", "123",
                 "--n_steps", "5",
                 "--grid_size", "16",
                 "--no-gpu",
                 "--run-dir", str(rd)],
                capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, result.stderr
            runs.append(json.loads(result.stdout.strip()))
        assert runs[0] == runs[1]

    def test_gpu_deterministic(self, tmp_path):
        """Same seed produces identical GPU output across runs."""
        runs = []
        for i in range(2):
            rd = tmp_path / f"gpu_run{i}"
            result = subprocess.run(
                [str(_BINARY),
                 "--time_sec", "10",
                 "--rigidity_kT_nm2", "20",
                 "--seed", "123",
                 "--n_steps", "5",
                 "--grid_size", "16",
                 "--run-dir", str(rd)],
                capture_output=True, text=True, timeout=60,
            )
            if "CPU fallback" in result.stderr:
                pytest.skip("Metal GPU not available")
            assert result.returncode == 0, result.stderr
            runs.append(json.loads(result.stdout.strip()))
        assert runs[0] == runs[1]

    def test_diagnostics_fields(self, tmp_path):
        result = subprocess.run(
            [str(_BINARY),
             "--time_sec", "10",
             "--rigidity_kT_nm2", "10",
             "--n_steps", "3",
             "--grid_size", "16",
             "--no-gpu",
             "--run-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.strip())
        diag = data["diagnostics"]
        assert "accept_rate" in diag
        assert "final_tcr_mean_r_nm" in diag
        assert "final_cd45_mean_r_nm" in diag
        assert "n_steps_actual" in diag
        assert 0.0 <= diag["accept_rate"] <= 1.0


class TestPythonWrapper:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()
        if not _BINARY.exists():
            pytest.skip("Binary not available")

    def test_wrapper_produces_output(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "models.kinetic_segregation",
             "--time_sec", "5",
             "--rigidity_kT_nm2", "10",
             "--n_steps", "3",
             "--grid_size", "16",
             "--no-gpu",
             "--run-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=60,
            cwd=_SUBMODULE_ROOT,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.strip())
        assert "depletion_width_nm" in data
