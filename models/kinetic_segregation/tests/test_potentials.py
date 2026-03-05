"""Tests for kinetic segregation potential functions.

Covers bending energy, TCR-pMHC attractive potential, and CD45 repulsion.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.kinetic_segregation.potentials import (
    bending_energy,
    cd45_repulsion,
    tcr_pmhc_potential,
)


class TestBendingEnergy:
    def test_flat_membrane_zero_energy(self):
        """A perfectly flat membrane has zero bending energy."""
        h = np.ones((16, 16)) * 35.0
        assert bending_energy(h, kappa=10.0, dx=100.0) == 0.0

    def test_curved_membrane_positive_energy(self):
        """A membrane with curvature has positive bending energy."""
        h = np.zeros((16, 16))
        h[8, 8] = 10.0  # single bump
        energy = bending_energy(h, kappa=10.0, dx=100.0)
        assert energy > 0.0

    def test_energy_scales_with_kappa(self):
        """Bending energy scales linearly with kappa."""
        h = np.zeros((16, 16))
        h[8, 8] = 10.0
        e1 = bending_energy(h, kappa=5.0, dx=100.0)
        e2 = bending_energy(h, kappa=10.0, dx=100.0)
        assert pytest.approx(e2, rel=1e-10) == 2.0 * e1

    def test_symmetry(self):
        """Bending energy is invariant to height field sign."""
        h = np.random.default_rng(0).standard_normal((16, 16))
        e_pos = bending_energy(h, kappa=5.0, dx=50.0)
        e_neg = bending_energy(-h, kappa=5.0, dx=50.0)
        assert pytest.approx(e_pos) == e_neg


class TestTcrPmhcPotential:
    def test_minimum_at_zero_height(self):
        """TCR potential is most negative at h=0 (tight contact)."""
        e_zero = tcr_pmhc_potential(0.0, u_assoc=20.0)
        e_far = tcr_pmhc_potential(50.0, u_assoc=20.0)
        assert e_zero < e_far
        assert e_zero == pytest.approx(-20.0)

    def test_approaches_zero_far_from_surface(self):
        """TCR potential decays to ~0 far from the membrane."""
        e = tcr_pmhc_potential(100.0, u_assoc=20.0, sigma_bind=3.0)
        assert abs(e) < 1e-10

    def test_scales_with_u_assoc(self):
        """Potential depth scales with u_assoc."""
        e1 = tcr_pmhc_potential(0.0, u_assoc=10.0)
        e2 = tcr_pmhc_potential(0.0, u_assoc=20.0)
        assert pytest.approx(e2) == 2.0 * e1

    def test_gaussian_shape(self):
        """Potential has correct Gaussian shape at sigma distance."""
        sigma = 3.0
        e_sigma = tcr_pmhc_potential(sigma, u_assoc=20.0, sigma_bind=sigma)
        expected = -20.0 * np.exp(-0.5)
        assert pytest.approx(e_sigma) == expected


class TestCd45Repulsion:
    def test_no_repulsion_above_height(self):
        """No repulsion when membrane height >= CD45 ectodomain height."""
        assert cd45_repulsion(35.0, cd45_height=35.0) == 0.0
        assert cd45_repulsion(50.0, cd45_height=35.0) == 0.0

    def test_repulsion_below_height(self):
        """Repulsive energy when membrane height < CD45 height."""
        e = cd45_repulsion(30.0, cd45_height=35.0)
        assert e > 0.0
        expected = 0.5 * 1.0 * (35.0 - 30.0) ** 2
        assert pytest.approx(e) == expected

    def test_increases_with_compression(self):
        """Repulsion increases as membrane gets closer to surface."""
        e1 = cd45_repulsion(30.0)
        e2 = cd45_repulsion(20.0)
        e3 = cd45_repulsion(5.0)
        assert e1 < e2 < e3

    def test_zero_height_maximum(self):
        """Maximum repulsion at h=0."""
        e = cd45_repulsion(0.0, cd45_height=35.0)
        expected = 0.5 * 35.0**2
        assert pytest.approx(e) == expected
