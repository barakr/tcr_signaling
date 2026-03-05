"""Tests for Lck activity model — mean active Lck concentration."""

from __future__ import annotations

import pytest

from models.lck_activity.model import mean_lck_activity


class TestMeanLckActivity:
    def test_zero_radius_returns_zero(self):
        """Zero contact radius → no area → zero activity."""
        result = mean_lck_activity(
            cd45_boundary_density=1.0,
            lck_decay_length=1.0,
            lck_activation_rate=1.0,
            contact_radius=0.0,
        )
        assert result == 0.0

    def test_zero_decay_returns_zero(self):
        """Zero decay length → no Lck propagation → zero activity."""
        result = mean_lck_activity(
            cd45_boundary_density=1.0,
            lck_decay_length=0.0,
            lck_activation_rate=1.0,
            contact_radius=5.0,
        )
        assert result == 0.0

    def test_positive_result(self):
        """Positive inputs give positive mean activity."""
        result = mean_lck_activity(
            cd45_boundary_density=10.0,
            lck_decay_length=1.0,
            lck_activation_rate=0.5,
            contact_radius=5.0,
        )
        assert result > 0.0

    def test_scales_with_activation_rate(self):
        """Mean activity scales linearly with activation rate."""
        base = dict(cd45_boundary_density=10.0, lck_decay_length=1.0, contact_radius=5.0)
        a1 = mean_lck_activity(lck_activation_rate=1.0, **base)
        a2 = mean_lck_activity(lck_activation_rate=2.0, **base)
        assert pytest.approx(a2) == 2.0 * a1

    def test_scales_with_boundary_density(self):
        """Mean activity scales linearly with CD45 boundary density."""
        base = dict(lck_decay_length=1.0, lck_activation_rate=0.5, contact_radius=5.0)
        a1 = mean_lck_activity(cd45_boundary_density=5.0, **base)
        a2 = mean_lck_activity(cd45_boundary_density=10.0, **base)
        assert pytest.approx(a2) == 2.0 * a1

    def test_large_decay_approaches_peak(self):
        """With decay >> radius, mean ≈ peak (uniform Lck across disk)."""
        peak = 0.5 * 10.0  # activation_rate * boundary_density
        result = mean_lck_activity(
            cd45_boundary_density=10.0,
            lck_decay_length=1000.0,
            lck_activation_rate=0.5,
            contact_radius=1.0,
        )
        assert pytest.approx(result, rel=0.01) == peak
