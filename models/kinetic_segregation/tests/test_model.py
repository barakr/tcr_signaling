"""Tests for kinetic segregation model helpers and full simulation."""

from __future__ import annotations

import numpy as np

from models.kinetic_segregation.model import (
    _compute_depletion_width,
    _height_at,
    _initial_positions,
    simulate_ks,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestInitialPositions:
    def test_uniform_positions_in_bounds(self):
        rng = np.random.default_rng(42)
        pos = _initial_positions(100, 2000.0, rng, center_bias=False)
        assert pos.shape == (100, 2)
        assert np.all(pos >= 0.0)
        assert np.all(pos <= 2000.0)

    def test_center_bias_near_center(self):
        rng = np.random.default_rng(42)
        pos = _initial_positions(200, 2000.0, rng, center_bias=True)
        center = 1000.0
        mean_r = np.sqrt(np.sum((pos - center) ** 2, axis=1)).mean()
        assert mean_r < 600.0

    def test_uniform_spread(self):
        rng = np.random.default_rng(42)
        pos = _initial_positions(500, 2000.0, rng, center_bias=False)
        center = 1000.0
        mean_r = np.sqrt(np.sum((pos - center) ** 2, axis=1)).mean()
        assert mean_r > 500.0


class TestHeightAt:
    def test_correct_lookup(self):
        h = np.arange(16).reshape(4, 4).astype(np.float64)
        dx = 500.0  # patch=2000, grid=4
        pos = np.array([[250.0, 750.0]])  # grid cell (0, 1)
        heights = _height_at(pos, h, dx)
        assert heights[0] == h[0, 1]

    def test_clipping_at_boundary(self):
        h = np.ones((4, 4)) * 35.0
        dx = 500.0
        pos = np.array([[2500.0, -100.0]])  # out of bounds
        heights = _height_at(pos, h, dx)
        assert heights[0] == 35.0  # clipped to valid grid


class TestComputeDepletionWidth:
    def test_separated_clusters(self):
        """Well-separated TCR and CD45 give positive depletion width."""
        patch = 2000.0
        center = 1000.0
        rng = np.random.default_rng(0)
        tcr = rng.normal(center, 50.0, size=(50, 2))
        cd45 = rng.normal(center, 50.0, size=(100, 2)) + 500.0
        width = _compute_depletion_width(tcr, cd45, patch)
        assert width > 0.0

    def test_overlapping_returns_nonnegative(self):
        """Overlapping distributions return non-negative width."""
        patch = 2000.0
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, patch, size=(100, 2))
        width = _compute_depletion_width(pos[:50], pos[50:], patch)
        assert width >= 0.0


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------
class TestSimulateKs:
    def test_returns_required_keys(self):
        result = simulate_ks(time_sec=10, rigidity_kT_nm2=10, n_steps=100, seed=42)
        assert "depletion_width_nm" in result
        assert "final_tcr_mean_r_nm" in result
        assert "final_cd45_mean_r_nm" in result
        assert "accept_rate" in result
        assert "n_steps_actual" in result

    def test_depletion_width_positive(self):
        result = simulate_ks(time_sec=50, rigidity_kT_nm2=20, n_steps=2000, seed=42)
        assert result["depletion_width_nm"] > 0.0

    def test_tcr_closer_to_center_than_cd45(self):
        """TCR molecules should be closer to center than CD45 after simulation."""
        result = simulate_ks(time_sec=50, rigidity_kT_nm2=20, n_steps=2000, seed=42)
        assert result["final_tcr_mean_r_nm"] < result["final_cd45_mean_r_nm"]

    def test_accept_rate_reasonable(self):
        """Accept rate should be between 0 and 1."""
        result = simulate_ks(time_sec=10, rigidity_kT_nm2=10, n_steps=500, seed=42)
        assert 0.0 < result["accept_rate"] <= 1.0

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        r1 = simulate_ks(time_sec=10, rigidity_kT_nm2=10, n_steps=200, seed=123)
        r2 = simulate_ks(time_sec=10, rigidity_kT_nm2=10, n_steps=200, seed=123)
        assert r1["depletion_width_nm"] == r2["depletion_width_nm"]
        assert r1["accept_rate"] == r2["accept_rate"]

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        r1 = simulate_ks(time_sec=10, rigidity_kT_nm2=10, n_steps=500, seed=1)
        r2 = simulate_ks(time_sec=10, rigidity_kT_nm2=10, n_steps=500, seed=2)
        assert r1["depletion_width_nm"] != r2["depletion_width_nm"]

    def test_n_steps_auto_scaling(self):
        """When n_steps is None, it scales with time_sec."""
        r_short = simulate_ks(time_sec=5, rigidity_kT_nm2=10, seed=42)
        r_long = simulate_ks(time_sec=50, rigidity_kT_nm2=10, seed=42)
        assert r_short["n_steps_actual"] < r_long["n_steps_actual"]

    def test_custom_molecule_counts(self):
        """Simulation works with custom molecule counts."""
        result = simulate_ks(
            time_sec=10, rigidity_kT_nm2=10, n_tcr=10, n_cd45=20, n_steps=100, seed=42
        )
        assert result["depletion_width_nm"] >= 0.0

    def test_small_grid(self):
        """Simulation works with small grid."""
        result = simulate_ks(time_sec=10, rigidity_kT_nm2=10, grid_size=8, n_steps=100, seed=42)
        assert result["depletion_width_nm"] >= 0.0

    def test_snapshot_recording(self):
        """Snapshot interval records intermediate states."""
        result = simulate_ks(
            time_sec=10, rigidity_kT_nm2=10, n_steps=100, seed=42, snapshot_interval=25
        )
        assert "snapshots" in result
        # Initial + 4 snapshots at steps 25, 50, 75, 100
        assert len(result["snapshots"]) == 5
        assert result["snapshots"][0]["step"] == 0
        assert result["snapshots"][-1]["step"] == 100

    def test_no_snapshots_by_default(self):
        """No snapshots key when snapshot_interval is 0."""
        result = simulate_ks(time_sec=10, rigidity_kT_nm2=10, n_steps=100, seed=42)
        assert "snapshots" not in result
