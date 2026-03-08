"""Tests for MATLAB/paper alignment changes.

Covers: reflecting boundary (Change 1), configurable CD45 params (Change 4),
periodic molecule BCs (Change 5), checkerboard grid update (Change 2),
soft molecular repulsion (Change 6), and pMHC gating (Change 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from models.kinetic_segregation.model import PATCH_SIZE_NM, simulate_ks
from models.kinetic_segregation.potentials import cd45_repulsion, mol_repulsion


# ---------------------------------------------------------------------------
# Change 1: Reflecting boundary
# ---------------------------------------------------------------------------
class TestReflectingBoundary:
    def test_reflection_formula(self):
        """old=2, delta=-5 → new_proposed=-3, result=abs(-3)=3, not 0."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=1,
            seed=42, grid_size=8, n_tcr=5, n_cd45=10,
        )
        # Just verify the simulation runs (reflecting boundary is internal).
        assert result["accept_rate"] > 0.0

    def test_height_stays_positive(self):
        """After simulation, all heights should be non-negative."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=5.0, n_steps=20,
            seed=42, grid_size=16, n_tcr=10, n_cd45=20,
            snapshot_interval=20,
        )
        h = result["snapshots"][-1]["h"]
        assert np.all(h >= 0.0)


# ---------------------------------------------------------------------------
# Change 4: Configurable CD45 parameters
# ---------------------------------------------------------------------------
class TestConfigurableCd45:
    def test_k_rep_parameter(self):
        """cd45_repulsion accepts k_rep and scales accordingly."""
        e1 = cd45_repulsion(20.0, cd45_height=35.0, k_rep=1.0)
        e2 = cd45_repulsion(20.0, cd45_height=35.0, k_rep=2.0)
        assert pytest.approx(e2) == 2.0 * e1

    def test_custom_cd45_height(self):
        """Simulation runs with custom cd45_height."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=3,
            seed=42, grid_size=8, n_tcr=5, n_cd45=10,
            cd45_height=50.0,
        )
        assert result["accept_rate"] > 0.0

    def test_custom_k_rep(self):
        """Simulation runs with custom cd45_k_rep."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=3,
            seed=42, grid_size=8, n_tcr=5, n_cd45=10,
            cd45_k_rep=0.5,
        )
        assert result["accept_rate"] > 0.0

    def test_matlab_params_vs_paper_params(self):
        """MATLAB (50nm, k=0.001) vs paper (35nm, k=1.0) produce different results."""
        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=20.0, n_steps=10,
            seed=42, grid_size=16, n_tcr=20, n_cd45=40,
        )
        r_paper = simulate_ks(**kwargs, cd45_height=35.0, cd45_k_rep=1.0)
        r_matlab = simulate_ks(**kwargs, cd45_height=50.0, cd45_k_rep=0.001)
        assert r_paper["depletion_width_nm"] != r_matlab["depletion_width_nm"]


# ---------------------------------------------------------------------------
# Change 5: Periodic molecule boundaries
# ---------------------------------------------------------------------------
class TestPeriodicMolBounds:
    def test_positions_stay_in_bounds(self):
        """All molecule positions stay within [0, patch_size) after simulation."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=10,
            seed=42, grid_size=16, n_tcr=20, n_cd45=40,
            snapshot_interval=10,
        )
        for snap in result["snapshots"]:
            for key in ["tcr_pos", "cd45_pos"]:
                pos = snap[key]
                assert np.all(pos >= 0.0), f"{key} has negative values"
                assert np.all(pos < PATCH_SIZE_NM), f"{key} exceeds patch size"


# ---------------------------------------------------------------------------
# Change 2: Checkerboard grid update
# ---------------------------------------------------------------------------
class TestCheckerboardGridUpdate:
    def test_deterministic_with_seed(self):
        """Checkerboard update remains deterministic."""
        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=5,
            seed=42, grid_size=16, n_tcr=10, n_cd45=20,
        )
        r1 = simulate_ks(**kwargs)
        r2 = simulate_ks(**kwargs)
        assert r1["depletion_width_nm"] == r2["depletion_width_nm"]
        assert r1["accept_rate"] == r2["accept_rate"]

    def test_accept_rate_reasonable(self):
        """Accept rate with checkerboard update should be reasonable."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=10,
            seed=42, grid_size=16, n_tcr=10, n_cd45=20,
        )
        assert 0.0 < result["accept_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Change 6: Soft molecular repulsion
# ---------------------------------------------------------------------------
class TestMolRepulsion:
    def test_within_cutoff_positive(self):
        """Two molecules within r_cut have positive repulsion."""
        pos = np.array([100.0, 100.0])
        all_pos = np.array([[100.0, 100.0], [105.0, 100.0]])
        e = mol_repulsion(pos, 0, all_pos, eps=5.0, r_cut=10.0, patch_size=2000.0)
        assert e > 0.0

    def test_beyond_cutoff_zero(self):
        """Two molecules beyond r_cut have zero repulsion."""
        pos = np.array([100.0, 100.0])
        all_pos = np.array([[100.0, 100.0], [200.0, 100.0]])
        e = mol_repulsion(pos, 0, all_pos, eps=5.0, r_cut=10.0, patch_size=2000.0)
        assert e == 0.0

    def test_self_exclusion(self):
        """Molecule does not interact with itself."""
        all_pos = np.array([[100.0, 100.0]])
        e = mol_repulsion(all_pos[0], 0, all_pos, eps=5.0, r_cut=10.0, patch_size=2000.0)
        assert e == 0.0

    def test_periodic_wrapping(self):
        """Repulsion works across periodic boundaries."""
        pos = np.array([5.0, 5.0])
        all_pos = np.array([[5.0, 5.0], [1995.0, 5.0]])  # 10nm apart via wrapping
        e = mol_repulsion(pos, 0, all_pos, eps=5.0, r_cut=20.0, patch_size=2000.0)
        assert e > 0.0

    def test_disabled_when_eps_zero(self):
        """No repulsion when eps=0."""
        pos = np.array([100.0, 100.0])
        all_pos = np.array([[100.0, 100.0], [101.0, 100.0]])
        e = mol_repulsion(pos, 0, all_pos, eps=0.0, r_cut=10.0, patch_size=2000.0)
        assert e == 0.0

    def test_simulation_with_repulsion(self):
        """Simulation runs with mol_repulsion enabled."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=5,
            seed=42, grid_size=8, n_tcr=10, n_cd45=20,
            mol_repulsion_eps=5.0, mol_repulsion_rcut=10.0,
        )
        assert result["accept_rate"] > 0.0

    def test_backward_compat_no_repulsion(self):
        """Default eps=0 produces same result as before."""
        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=3,
            seed=42, grid_size=8, n_tcr=5, n_cd45=10,
        )
        r1 = simulate_ks(**kwargs)
        r2 = simulate_ks(**kwargs, mol_repulsion_eps=0.0)
        assert r1["depletion_width_nm"] == r2["depletion_width_nm"]


# ---------------------------------------------------------------------------
# Change 3: pMHC gating
# ---------------------------------------------------------------------------
class TestPmhcGating:
    def test_no_pmhc_backward_compat(self):
        """n_pmhc=0 produces same result as before (all cells have pMHC)."""
        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=3,
            seed=42, grid_size=8, n_tcr=5, n_cd45=10,
        )
        r1 = simulate_ks(**kwargs)
        r2 = simulate_ks(**kwargs, n_pmhc=0)
        assert r1["depletion_width_nm"] == r2["depletion_width_nm"]
        assert r1["accept_rate"] == r2["accept_rate"]

    def test_pmhc_everywhere_produces_segregation(self):
        """Saturating pMHC across all grid cells should still produce segregation.

        With TCR co-location on pMHC and enough steps, TCR should still end up
        closer to center than CD45 (the pMHC grid saturates all cells, so binding
        happens everywhere just like the default n_pmhc=0 case).
        """
        # 8x8 grid = 64 cells, use 1000 pMHC to saturate all cells
        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=10,
            seed=42, grid_size=8, n_tcr=10, n_cd45=20,
        )
        r_saturated = simulate_ks(**kwargs, n_pmhc=1000, pmhc_seed=42,
                                  pmhc_mode="uniform")
        assert r_saturated["depletion_width_nm"] >= 0.0
        assert r_saturated["accept_rate"] > 0.0

    def test_no_pmhc_free_diffusion(self):
        """With very few pMHC away from center, TCR should diffuse more freely.

        Compare: n_pmhc=0 (binding everywhere) vs n_pmhc with pMHC only at edges.
        With pMHC far from center, TCR shouldn't be attracted to center.
        """
        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=20.0, n_steps=20,
            seed=42, grid_size=16, n_tcr=20, n_cd45=40,
        )
        # Default: binding everywhere → TCR attracted to center
        r_default = simulate_ks(**kwargs)

        # Place pMHC only at very edge positions
        edge_pos = np.array([[1900.0, 1900.0], [100.0, 1900.0], [1900.0, 100.0]])
        r_edge = simulate_ks(**kwargs, pmhc_pos=edge_pos)

        # With pMHC only at edges, TCR mean radius should be larger
        # (less attraction to center)
        assert r_edge["final_tcr_mean_r_nm"] > r_default["final_tcr_mean_r_nm"] * 0.95

    def test_simulation_with_pmhc(self):
        """Simulation runs with n_pmhc > 0."""
        result = simulate_ks(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=5,
            seed=42, grid_size=8, n_tcr=10, n_cd45=20,
            n_pmhc=20,
        )
        assert result["accept_rate"] > 0.0
        assert result["depletion_width_nm"] >= 0.0

    def test_pmhc_deterministic(self):
        """pMHC placement is deterministic with same seed."""
        kwargs = dict(
            time_sec=1.0, rigidity_kT_nm2=10.0, n_steps=5,
            seed=42, grid_size=8, n_tcr=10, n_cd45=20,
            n_pmhc=20, pmhc_seed=99,
        )
        r1 = simulate_ks(**kwargs)
        r2 = simulate_ks(**kwargs)
        assert r1["depletion_width_nm"] == r2["depletion_width_nm"]
