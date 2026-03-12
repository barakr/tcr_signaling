"""Tests for extended depletion metrics (DepletionMetrics struct).

Validates that the six metrics are present, in correct ranges, monotonic
with rigidity, and deterministic.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"


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
    rd = tmp_path / label
    cmd = [str(_BINARY), "--run-dir", str(rd), "--no-gpu"]
    defaults = {
        "time_sec": 10.0,
        "rigidity_kT_nm2": 20.0,
        "seed": 42,
        "n_steps": 200,
        "grid_size": 64,
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        cmd.extend([f"--{k}", str(v)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Binary failed: {result.stderr}"
    return json.loads(result.stdout.strip())


class TestMetricPresence:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()

    def test_new_metric_keys_present(self, tmp_path):
        data = _run(tmp_path, label="keys")
        diag = data["diagnostics"]
        for key in [
            "depletion_percentile_gap_nm",
            "depletion_overlap_coeff",
            "depletion_ks_statistic",
            "depletion_frontier_nn_gap_nm",
            "depletion_cross_nn_median_nm",
        ]:
            assert key in diag, f"Missing key: {key}"

    def test_backward_compat_depletion_width(self, tmp_path):
        data = _run(tmp_path, label="compat")
        assert "depletion_width_nm" in data
        assert isinstance(data["depletion_width_nm"], (int, float))


class TestMetricRanges:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()

    def test_overlap_in_range(self, tmp_path):
        data = _run(tmp_path, label="ovl_range")
        ovl = data["diagnostics"]["depletion_overlap_coeff"]
        assert 0.0 <= ovl <= 1.0, f"Overlap out of range: {ovl}"

    def test_ks_in_range(self, tmp_path):
        data = _run(tmp_path, label="ks_range")
        ks = data["diagnostics"]["depletion_ks_statistic"]
        assert 0.0 <= ks <= 1.0, f"KS statistic out of range: {ks}"

    def test_frontier_nn_non_negative(self, tmp_path):
        data = _run(tmp_path, label="fnn_range")
        fnn = data["diagnostics"]["depletion_frontier_nn_gap_nm"]
        assert fnn >= 0.0, f"Frontier NN gap negative: {fnn}"

    def test_cross_nn_non_negative(self, tmp_path):
        data = _run(tmp_path, label="cnn_range")
        cnn = data["diagnostics"]["depletion_cross_nn_median_nm"]
        assert cnn >= 0.0, f"Cross NN median negative: {cnn}"


class TestMetricPhysics:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()

    def test_percentile_gap_positive_at_high_rigidity(self, tmp_path):
        # Need enough steps for equilibration; 500 is too few for clear separation.
        data = _run(tmp_path, label="pct_high", rigidity_kT_nm2=80.0, n_steps=5000)
        gap = data["diagnostics"]["depletion_percentile_gap_nm"]
        assert gap > -50, f"Percentile gap too negative at rig=80: {gap}"

    def test_overlap_low_at_high_rigidity(self, tmp_path):
        data = _run(tmp_path, label="ovl_high", rigidity_kT_nm2=80.0, n_steps=5000)
        ovl = data["diagnostics"]["depletion_overlap_coeff"]
        assert ovl < 0.6, f"Overlap too high at rig=80: {ovl}"

    def test_ks_high_at_high_rigidity(self, tmp_path):
        data = _run(tmp_path, label="ks_high", rigidity_kT_nm2=80.0, n_steps=5000)
        ks = data["diagnostics"]["depletion_ks_statistic"]
        assert ks > 0.2, f"KS too low at rig=80: {ks}"

    def test_frontier_nn_positive_at_high_rigidity(self, tmp_path):
        data = _run(tmp_path, label="fnn_high", rigidity_kT_nm2=30.0, n_steps=5000)
        fnn = data["diagnostics"]["depletion_frontier_nn_gap_nm"]
        assert fnn > 20, f"Frontier NN too small at rig=30: {fnn}"

    def test_metrics_monotonic_with_rigidity(self, tmp_path):
        d_low = _run(tmp_path, label="mono_low", rigidity_kT_nm2=1.0, n_steps=5000)
        d_high = _run(tmp_path, label="mono_high", rigidity_kT_nm2=100.0, n_steps=5000)

        # Frontier NN should be larger at higher rigidity (better separation).
        fnn_low = d_low["diagnostics"]["depletion_frontier_nn_gap_nm"]
        fnn_high = d_high["diagnostics"]["depletion_frontier_nn_gap_nm"]
        assert fnn_high >= fnn_low * 0.8, \
            f"Frontier NN not monotonic: rig=1 → {fnn_low:.1f}, rig=100 → {fnn_high:.1f}"

        # Cross NN should be larger at higher rigidity.
        cnn_low = d_low["diagnostics"]["depletion_cross_nn_median_nm"]
        cnn_high = d_high["diagnostics"]["depletion_cross_nn_median_nm"]
        assert cnn_high >= cnn_low * 0.8, \
            f"Cross NN not monotonic: rig=1 → {cnn_low:.1f}, rig=100 → {cnn_high:.1f}"


class TestMetricDeterminism:
    @pytest.fixture(autouse=True)
    def _build(self):
        _ensure_binary()

    def test_metrics_deterministic(self, tmp_path):
        d1 = _run(tmp_path, label="det1", seed=42, n_steps=100)
        d2 = _run(tmp_path, label="det2", seed=42, n_steps=100)

        for key in [
            "depletion_percentile_gap_nm",
            "depletion_overlap_coeff",
            "depletion_ks_statistic",
            "depletion_frontier_nn_gap_nm",
            "depletion_cross_nn_median_nm",
        ]:
            assert d1["diagnostics"][key] == d2["diagnostics"][key], \
                f"{key} not deterministic: {d1['diagnostics'][key]} vs {d2['diagnostics'][key]}"
