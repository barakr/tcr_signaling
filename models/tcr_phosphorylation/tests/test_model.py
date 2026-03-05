"""Tests for TCR phosphorylation model — pTCR fraction and density."""

from __future__ import annotations

import pytest

from models.tcr_phosphorylation.model import ptcr_density, ptcr_fraction


class TestPtcrFraction:
    def test_high_lck_approaches_one(self):
        """Very high Lck activity drives fraction → 1."""
        frac = ptcr_fraction(
            mean_lck_activity=1e6,
            phosphorylation_rate=1.0,
            dephosphorylation_rate=1.0,
        )
        assert pytest.approx(frac, abs=1e-4) == 1.0

    def test_zero_lck_returns_zero(self):
        """Zero Lck activity → no phosphorylation → fraction = 0."""
        frac = ptcr_fraction(
            mean_lck_activity=0.0,
            phosphorylation_rate=1.0,
            dephosphorylation_rate=1.0,
        )
        assert frac == 0.0

    def test_equal_rates_gives_half(self):
        """When k_phos * Lck = k_dephos, fraction = 0.5."""
        frac = ptcr_fraction(
            mean_lck_activity=1.0,
            phosphorylation_rate=1.0,
            dephosphorylation_rate=1.0,
        )
        assert pytest.approx(frac) == 0.5

    def test_fraction_in_unit_interval(self):
        """Fraction is always in [0, 1]."""
        frac = ptcr_fraction(
            mean_lck_activity=5.0,
            phosphorylation_rate=2.0,
            dephosphorylation_rate=3.0,
        )
        assert 0.0 <= frac <= 1.0

    def test_zero_rates_returns_zero(self):
        """Both rates zero → total=0 → return 0 gracefully."""
        frac = ptcr_fraction(
            mean_lck_activity=1.0,
            phosphorylation_rate=0.0,
            dephosphorylation_rate=0.0,
        )
        assert frac == 0.0


class TestPtcrDensity:
    def test_density_equals_fraction_times_total(self):
        """Density = fraction * tcr_density."""
        density = ptcr_density(ptcr_frac=0.5, tcr_density=100.0)
        assert pytest.approx(density) == 50.0

    def test_zero_fraction_zero_density(self):
        """Zero fraction → zero density."""
        assert ptcr_density(0.0, 200.0) == 0.0

    def test_full_fraction_equals_total(self):
        """Fraction = 1 → density = tcr_density."""
        assert ptcr_density(1.0, 150.0) == pytest.approx(150.0)
