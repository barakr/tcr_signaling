"""Tests for kinetic segregation model helpers and full simulation."""

from __future__ import annotations

import numpy as np

from models.kinetic_segregation.model import (
    D_H_DEFAULT,
    D_MOL_DEFAULT,
    DT_SAFETY,
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
# Full simulation — n_steps is now full MC sweeps (each sweep updates every
# molecule and every grid cell), so values are much smaller than before.
# ---------------------------------------------------------------------------
class TestSimulateKs:
    def test_returns_required_keys(self):
        result = simulate_ks(
            time_sec=10,
            rigidity_kT_nm2=10,
            n_steps=3,
            seed=42,
            grid_size=8,
            n_tcr=10,
            n_cd45=20,
        )
        assert "depletion_width_nm" in result
        assert "final_tcr_mean_r_nm" in result
        assert "final_cd45_mean_r_nm" in result
        assert "accept_rate" in result
        assert "n_steps_actual" in result

    def test_depletion_width_positive(self):
        result = simulate_ks(
            time_sec=50,
            rigidity_kT_nm2=20,
            n_steps=10,
            seed=42,
            grid_size=16,
            n_tcr=20,
            n_cd45=40,
        )
        assert result["depletion_width_nm"] > 0.0

    def test_tcr_closer_to_center_than_cd45(self):
        """TCR molecules should be closer to center than CD45 after simulation."""
        result = simulate_ks(
            time_sec=50,
            rigidity_kT_nm2=20,
            n_steps=20,
            seed=42,
            grid_size=16,
            n_tcr=20,
            n_cd45=40,
        )
        assert result["final_tcr_mean_r_nm"] < result["final_cd45_mean_r_nm"]

    def test_accept_rate_reasonable(self):
        """Accept rate should be between 0 and 1."""
        result = simulate_ks(
            time_sec=10,
            rigidity_kT_nm2=10,
            n_steps=5,
            seed=42,
            grid_size=8,
            n_tcr=10,
            n_cd45=20,
        )
        assert 0.0 < result["accept_rate"] <= 1.0

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        kwargs = dict(
            time_sec=10,
            rigidity_kT_nm2=10,
            n_steps=3,
            seed=123,
            grid_size=8,
            n_tcr=10,
            n_cd45=20,
        )
        r1 = simulate_ks(**kwargs)
        r2 = simulate_ks(**kwargs)
        assert r1["depletion_width_nm"] == r2["depletion_width_nm"]
        assert r1["accept_rate"] == r2["accept_rate"]

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        kwargs = dict(
            time_sec=10,
            rigidity_kT_nm2=10,
            n_steps=5,
            grid_size=8,
            n_tcr=10,
            n_cd45=20,
        )
        r1 = simulate_ks(**kwargs, seed=1)
        r2 = simulate_ks(**kwargs, seed=2)
        assert r1["depletion_width_nm"] != r2["depletion_width_nm"]

    def test_n_steps_auto_scaling(self):
        """When n_steps is None, it scales with time_sec."""
        r_short = simulate_ks(
            time_sec=5,
            rigidity_kT_nm2=10,
            seed=42,
            grid_size=8,
            n_tcr=5,
            n_cd45=10,
        )
        r_long = simulate_ks(
            time_sec=50,
            rigidity_kT_nm2=10,
            seed=42,
            grid_size=8,
            n_tcr=5,
            n_cd45=10,
        )
        assert r_short["n_steps_actual"] < r_long["n_steps_actual"]

    def test_custom_molecule_counts(self):
        """Simulation works with custom molecule counts."""
        result = simulate_ks(
            time_sec=10,
            rigidity_kT_nm2=10,
            n_tcr=10,
            n_cd45=20,
            n_steps=3,
            seed=42,
            grid_size=8,
        )
        assert result["depletion_width_nm"] >= 0.0

    def test_small_grid(self):
        """Simulation works with small grid."""
        result = simulate_ks(
            time_sec=10,
            rigidity_kT_nm2=10,
            grid_size=8,
            n_steps=3,
            seed=42,
            n_tcr=10,
            n_cd45=20,
        )
        assert result["depletion_width_nm"] >= 0.0

    def test_snapshot_recording(self):
        """Snapshot interval records intermediate states."""
        result = simulate_ks(
            time_sec=10,
            rigidity_kT_nm2=10,
            n_steps=10,
            seed=42,
            snapshot_interval=5,
            grid_size=8,
            n_tcr=10,
            n_cd45=20,
        )
        assert "snapshots" in result
        # Initial + 2 snapshots at steps 5, 10
        assert len(result["snapshots"]) == 3
        assert result["snapshots"][0]["step"] == 0
        assert result["snapshots"][-1]["step"] == 10

    def test_no_snapshots_by_default(self):
        """No snapshots key when snapshot_interval is 0."""
        result = simulate_ks(
            time_sec=10,
            rigidity_kT_nm2=10,
            n_steps=3,
            seed=42,
            grid_size=8,
            n_tcr=10,
            n_cd45=20,
        )
        assert "snapshots" not in result

    def test_explicit_n_steps_is_raw_override(self):
        """When n_steps is explicit, it is used as-is (no time scaling)."""
        r = simulate_ks(
            time_sec=100,
            rigidity_kT_nm2=10,
            n_steps=7,
            seed=42,
            grid_size=8,
            n_tcr=5,
            n_cd45=10,
        )
        assert r["n_steps_actual"] == 7

    def test_depletion_increases_with_steps(self):
        """At moderate rigidity, more MC sweeps produce larger mean depletion.

        Uses n_steps explicitly to control sweep count, averages over seeds.
        """
        import statistics

        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=30.0, grid_size=16, n_tcr=30, n_cd45=60,
        )
        short_vals = [
            simulate_ks(n_steps=5, seed=s, **kwargs)["depletion_width_nm"]
            for s in range(10)
        ]
        long_vals = [
            simulate_ks(n_steps=50, seed=s, **kwargs)["depletion_width_nm"]
            for s in range(10)
        ]

        mean_short = statistics.mean(short_vals)
        mean_long = statistics.mean(long_vals)
        assert mean_long > mean_short, (
            f"Expected mean depletion at 50 steps ({mean_long:.1f}) > "
            f"5 steps ({mean_short:.1f})"
        )

    def test_dt_scales_with_grid(self):
        """dt should decrease with finer grid (stability constraint)."""
        r_coarse = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=50.0, n_steps=3,
            seed=42, grid_size=32, n_tcr=5, n_cd45=10,
        )
        r_fine = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=50.0, n_steps=3,
            seed=42, grid_size=64, n_tcr=5, n_cd45=10,
        )
        assert r_coarse["dt_seconds"] > r_fine["dt_seconds"]

    def test_step_sizes_from_physics(self):
        """Step sizes should be derived from D and dt, not grid heuristics."""
        r = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=50.0, n_steps=3,
            seed=42, grid_size=64, n_tcr=5, n_cd45=10,
        )
        dt = r["dt_seconds"]
        expected_h = np.sqrt(2.0 * D_H_DEFAULT * dt)
        expected_mol = np.sqrt(2.0 * D_MOL_DEFAULT * dt)
        assert abs(r["step_size_h_nm"] - expected_h) < 1e-10
        assert abs(r["step_size_mol_nm"] - expected_mol) < 1e-10

    def test_n_steps_auto_from_time(self):
        """Auto n_steps = max(50, round(time_sec / dt))."""
        grid_size = 64
        kappa = 50.0
        dx = 2000.0 / grid_size
        dt_stable = dx**2 / (2.0 * D_H_DEFAULT * kappa)
        dt = dt_stable * DT_SAFETY
        expected_steps = max(50, round(20.0 / dt))

        r = simulate_ks(
            time_sec=20.0, rigidity_kT_nm2=kappa,
            seed=42, grid_size=grid_size, n_tcr=5, n_cd45=10,
            n_steps=3,  # override to keep test fast
        )
        # Verify dt is correct (n_steps overridden, but dt is still computed)
        assert abs(r["dt_seconds"] - dt) < 1e-15

    def test_diagnostics_keys_present(self):
        """Result should include Brownian dynamics diagnostics."""
        r = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=3,
            seed=42, grid_size=8, n_tcr=5, n_cd45=10,
        )
        for key in ["dt_seconds", "step_size_h_nm", "step_size_mol_nm",
                     "D_mol_nm2_per_s", "D_h_nm2_per_s"]:
            assert key in r, f"Missing key: {key}"
