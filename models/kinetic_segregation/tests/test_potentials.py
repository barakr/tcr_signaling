"""Tests for C potential functions via ctypes, validated against analytical formulas."""
from __future__ import annotations

import ctypes
import math
import subprocess
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_LIB_PATH = _PKG_DIR / "build" / "libks_potentials.dylib"


def _build_testlib():
    """Build the shared library if not present."""
    if _LIB_PATH.exists():
        return
    result = subprocess.run(
        ["make", "testlib"], cwd=str(_PKG_DIR),
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to build test library: {result.stderr}")


@pytest.fixture(scope="module")
def lib():
    _build_testlib()
    if not _LIB_PATH.exists():
        pytest.skip("Test library not available")
    lib = ctypes.CDLL(str(_LIB_PATH))

    # tcr_pmhc_potential(double h, double u_assoc, double sigma_bind) -> double
    lib.tcr_pmhc_potential.restype = ctypes.c_double
    lib.tcr_pmhc_potential.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]

    # cd45_repulsion(double h, double cd45_height, double k_rep) -> double
    lib.cd45_repulsion.restype = ctypes.c_double
    lib.cd45_repulsion.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]

    # bending_energy_delta(double *h, int n, double kappa, double dx,
    #                      int gi, int gj, double old_val, double new_val) -> double
    lib.bending_energy_delta.restype = ctypes.c_double
    lib.bending_energy_delta.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
    ]

    return lib


class TestTcrPotentialC:
    def test_zero_height(self, lib):
        """E(h=0) = -u_assoc."""
        e = lib.tcr_pmhc_potential(0.0, 20.0, 3.0)
        assert e == pytest.approx(-20.0)

    def test_far_away(self, lib):
        """Potential decays to ~0 far from surface."""
        e = lib.tcr_pmhc_potential(100.0, 20.0, 3.0)
        assert abs(e) < 1e-10

    def test_at_sigma(self, lib):
        """Correct Gaussian value at h=sigma."""
        e = lib.tcr_pmhc_potential(3.0, 20.0, 3.0)
        expected = -20.0 * math.exp(-0.5)
        assert e == pytest.approx(expected)

    def test_scales_with_u_assoc(self, lib):
        e1 = lib.tcr_pmhc_potential(0.0, 10.0, 3.0)
        e2 = lib.tcr_pmhc_potential(0.0, 20.0, 3.0)
        assert e2 == pytest.approx(2.0 * e1)

    def test_matches_analytical(self, lib):
        """Compare against analytical formula: -u_assoc * exp(-h^2 / (2*sigma^2))."""
        u_assoc, sigma = 20.0, 3.0
        for h in [0.0, 1.5, 3.0, 10.0, 35.0, 50.0]:
            c_val = lib.tcr_pmhc_potential(h, u_assoc, sigma)
            expected = -u_assoc * math.exp(-h ** 2 / (2.0 * sigma ** 2))
            assert c_val == pytest.approx(expected, abs=1e-12), f"Mismatch at h={h}"


class TestCd45RepulsionC:
    def test_above_height(self, lib):
        assert lib.cd45_repulsion(35.0, 35.0, 1.0) == 0.0
        assert lib.cd45_repulsion(50.0, 35.0, 1.0) == 0.0

    def test_below_height(self, lib):
        e = lib.cd45_repulsion(30.0, 35.0, 1.0)
        expected = 0.5 * (35.0 - 30.0) ** 2
        assert e == pytest.approx(expected)

    def test_zero_height(self, lib):
        e = lib.cd45_repulsion(0.0, 35.0, 1.0)
        expected = 0.5 * 35.0 ** 2
        assert e == pytest.approx(expected)

    def test_matches_analytical(self, lib):
        """Compare against analytical formula: 0.5*k_rep*(cd45_h - h)^2 if h < cd45_h."""
        cd45_h, k_rep = 35.0, 1.0
        for h in [0.0, 5.0, 20.0, 34.9, 35.0, 50.0]:
            c_val = lib.cd45_repulsion(h, cd45_h, k_rep)
            expected = 0.5 * k_rep * (cd45_h - h) ** 2 if h < cd45_h else 0.0
            assert c_val == pytest.approx(expected, abs=1e-12), f"Mismatch at h={h}"


class TestBendingEnergyDeltaC:
    def test_perturbation_nonzero(self, lib):
        """Perturbing a non-flat membrane should produce nonzero energy delta."""
        import numpy as np

        rng = np.random.default_rng(42)
        n = 16
        h_np = rng.uniform(0, 50, (n, n))
        kappa = 10.0
        dx = 50.0

        nonzero_count = 0
        for _ in range(20):
            gi, gj = rng.integers(0, n, size=2)
            old_val = h_np[gi, gj]
            new_val = old_val + rng.normal(0, 2.0)

            h_tmp = h_np.copy()
            h_tmp[gi, gj] = new_val
            h_flat = h_tmp.flatten().astype(np.float64)
            h_c = (ctypes.c_double * len(h_flat))(*h_flat)
            c_val = lib.bending_energy_delta(h_c, n, kappa, dx, int(gi), int(gj), old_val, new_val)

            if abs(new_val - old_val) > 0.01:
                nonzero_count += 1
                assert abs(c_val) > 0.0, f"Zero delta for nontrivial perturbation at ({gi},{gj})"

        assert nonzero_count >= 15, "Too few nontrivial perturbations tested"

    def test_flat_zero(self, lib):
        """Flat membrane, no change -> zero delta."""
        import numpy as np

        n = 16
        h_flat = np.full(n * n, 35.0)
        h_c = (ctypes.c_double * len(h_flat))(*h_flat)
        delta = lib.bending_energy_delta(h_c, n, 10.0, 50.0, 8, 8, 35.0, 35.0)
        assert delta == pytest.approx(0.0, abs=1e-15)
