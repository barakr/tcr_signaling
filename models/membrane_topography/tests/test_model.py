"""Tests for membrane topography model — contact geometry functions."""

from __future__ import annotations

import numpy as np
import pytest

from models.membrane_topography.model import contact_fraction, contact_perimeter


class TestContactFraction:
    def test_fraction_clamped_to_one(self):
        """Fraction never exceeds 1.0 even if disk > patch."""
        frac = contact_fraction(contact_radius=10.0, patch_size=5.0)
        assert frac == 1.0

    def test_fraction_in_range(self):
        """Fraction is between 0 and 1 for radius < patch/sqrt(pi)."""
        frac = contact_fraction(contact_radius=1.0, patch_size=10.0)
        assert 0.0 < frac < 1.0

    def test_known_value(self):
        """Check against manual calculation: pi*r^2 / s^2."""
        r, s = 5.0, 20.0
        expected = np.pi * r**2 / s**2
        assert contact_fraction(r, s) == pytest.approx(expected)

    def test_zero_radius(self):
        """Zero radius gives zero fraction."""
        assert contact_fraction(0.0, 10.0) == 0.0


class TestContactPerimeter:
    def test_known_perimeter(self):
        """Perimeter = 2*pi*r."""
        r = 3.0
        assert contact_perimeter(r) == pytest.approx(2.0 * np.pi * r)

    def test_zero_radius(self):
        """Zero radius gives zero perimeter."""
        assert contact_perimeter(0.0) == 0.0

    def test_scales_linearly(self):
        """Perimeter scales linearly with radius."""
        p1 = contact_perimeter(2.0)
        p2 = contact_perimeter(4.0)
        assert pytest.approx(p2) == 2.0 * p1
