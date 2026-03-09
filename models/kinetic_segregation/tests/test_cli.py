"""Tests for kinetic segregation CLI entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Submodule root — needed as cwd so ``-m models.kinetic_segregation`` works.
_SUBMODULE_ROOT = str(Path(__file__).resolve().parents[3])


class TestKsCli:
    def test_cli_produces_output(self, tmp_path):
        """CLI writes segregation.json with expected fields."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.kinetic_segregation",
                "--time_sec",
                "10",
                "--rigidity_kT_nm2",
                "10",
                "--seed",
                "42",
                "--n_steps",
                "50",
                "--n_tcr",
                "20",
                "--n_cd45",
                "40",
                "--grid_size",
                "16",
                "--run-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_SUBMODULE_ROOT,
        )
        assert result.returncode == 0, result.stderr

        out_file = tmp_path / "out" / "segregation.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "depletion_width_nm" in data
        assert "inputs" in data
        assert data["inputs"]["time_sec"] == 10.0
        assert data["inputs"]["rigidity_kT_nm2"] == 10.0

    def test_cli_stdout_is_valid_json(self, tmp_path):
        """CLI stdout should be parseable JSON."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.kinetic_segregation",
                "--time_sec",
                "5",
                "--rigidity_kT_nm2",
                "5",
                "--seed",
                "1",
                "--n_steps",
                "50",
                "--n_tcr",
                "20",
                "--n_cd45",
                "40",
                "--grid_size",
                "16",
                "--run-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_SUBMODULE_ROOT,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.strip())
        assert isinstance(data["depletion_width_nm"], float)

    def test_cli_missing_required_args(self, tmp_path):
        """CLI exits with non-zero status when required args are missing."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.kinetic_segregation",
                "--run-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_SUBMODULE_ROOT,
        )
        assert result.returncode != 0

    def test_cli_param_file(self, tmp_path):
        """CLI loads params from JSON file."""
        params = {
            "time_sec": 5.0,
            "rigidity_kT_nm2": 8.0,
            "n_steps": 20,
            "seed": 7,
            "n_tcr": 10,
            "n_cd45": 20,
            "grid_size": 16,
        }
        param_file = tmp_path / "params.json"
        param_file.write_text(json.dumps(params))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.kinetic_segregation",
                "--params",
                str(param_file),
                "--run-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_SUBMODULE_ROOT,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.strip())
        assert data["inputs"]["time_sec"] == 5.0
        assert data["inputs"]["rigidity_kT_nm2"] == 8.0

    def test_cli_override_param_file(self, tmp_path):
        """CLI args override param file values."""
        params = {
            "time_sec": 5.0,
            "rigidity_kT_nm2": 8.0,
            "n_steps": 20,
            "n_pmhc": 50,
            "n_tcr": 10,
            "n_cd45": 20,
            "grid_size": 16,
        }
        param_file = tmp_path / "params.json"
        param_file.write_text(json.dumps(params))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.kinetic_segregation",
                "--params",
                str(param_file),
                "--n_pmhc",
                "10",
                "--run-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_SUBMODULE_ROOT,
        )
        assert result.returncode == 0, result.stderr
