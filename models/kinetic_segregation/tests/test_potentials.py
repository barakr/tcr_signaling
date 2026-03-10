"""Tests for C potential functions via ctypes, validated against analytical formulas."""
from __future__ import annotations

import ctypes
import math
import subprocess
import sys
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_LIB_EXT = ".dylib" if sys.platform == "darwin" else ".so"
_LIB_PATH = _PKG_DIR / "build" / f"libks_potentials{_LIB_EXT}"


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


# ── Cell-list accelerated repulsion tests ──────────────────────────────

class _CellListCtypes:
    """Wrapper for CellList ctypes bindings."""

    def __init__(self, lib):
        import numpy as np
        self.lib = lib
        self.np = np

        # CellList struct layout (matches cell_list.h).
        class CellListStruct(ctypes.Structure):
            _fields_ = [
                ("head", ctypes.POINTER(ctypes.c_int)),
                ("next", ctypes.POINTER(ctypes.c_int)),
                ("nc", ctypes.c_int),
                ("capacity", ctypes.c_int),
                ("cell_size", ctypes.c_double),
                ("inv_cell_size", ctypes.c_double),
                ("patch_size", ctypes.c_double),
            ]

        self.CellListStruct = CellListStruct

        lib.cell_list_init.restype = None
        lib.cell_list_init.argtypes = [
            ctypes.POINTER(CellListStruct), ctypes.c_int,
            ctypes.c_double, ctypes.c_double,
        ]
        lib.cell_list_build.restype = None
        lib.cell_list_build.argtypes = [
            ctypes.POINTER(CellListStruct),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ]
        lib.cell_list_free.restype = None
        lib.cell_list_free.argtypes = [ctypes.POINTER(CellListStruct)]

        lib.mol_repulsion.restype = ctypes.c_double
        lib.mol_repulsion.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        lib.mol_repulsion_cell.restype = ctypes.c_double
        lib.mol_repulsion_cell.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(CellListStruct),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        lib.mol_repulsion_delta.restype = ctypes.c_double
        lib.mol_repulsion_delta.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(CellListStruct),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]

    def make_positions(self, n_mol, patch_size, rng):
        pos = rng.uniform(0, patch_size, (n_mol, 2))
        flat = pos.flatten().astype(self.np.float64)
        c_pos = (ctypes.c_double * len(flat))(*flat)
        return pos, c_pos

    def build_cell_list(self, c_pos, n_mol, r_cut, patch_size):
        cl = self.CellListStruct()
        self.lib.cell_list_init(ctypes.byref(cl), n_mol, r_cut, patch_size)
        self.lib.cell_list_build(ctypes.byref(cl), c_pos, n_mol)
        return cl


class TestMolRepulsionCell:
    """Verify cell-list mol_repulsion matches brute-force."""

    def test_matches_brute_force(self, lib):
        import numpy as np
        rng = np.random.default_rng(123)
        w = _CellListCtypes(lib)
        patch = 2000.0
        r_cut = 50.0
        eps = 2.0
        n_mol = 150

        pos, c_pos = w.make_positions(n_mol, patch, rng)
        cl = w.build_cell_list(c_pos, n_mol, r_cut, patch)

        for idx in range(0, n_mol, 10):
            p = (ctypes.c_double * 2)(pos[idx, 0], pos[idx, 1])
            brute = lib.mol_repulsion(p, idx, c_pos, n_mol, eps, r_cut, patch)
            cell = lib.mol_repulsion_cell(p, idx, ctypes.byref(cl), c_pos,
                                          eps, r_cut, patch)
            assert cell == pytest.approx(brute, abs=1e-12), (
                f"Mismatch at idx={idx}: brute={brute}, cell={cell}"
            )

        lib.cell_list_free(ctypes.byref(cl))

    def test_periodic_boundary(self, lib):
        """Molecules near patch edges should interact across boundary."""
        import numpy as np
        w = _CellListCtypes(lib)
        patch = 2000.0
        r_cut = 50.0
        eps = 2.0

        # Place two molecules near opposite edges (30nm apart across boundary).
        pos = np.array([[10.0, 1000.0], [1980.0, 1000.0]], dtype=np.float64)
        c_pos = (ctypes.c_double * 4)(*pos.flatten())
        cl = w.build_cell_list(c_pos, 2, r_cut, patch)

        p0 = (ctypes.c_double * 2)(pos[0, 0], pos[0, 1])
        brute = lib.mol_repulsion(p0, 0, c_pos, 2, eps, r_cut, patch)
        cell = lib.mol_repulsion_cell(p0, 0, ctypes.byref(cl), c_pos,
                                      eps, r_cut, patch)
        assert brute > 0.0, "Brute force should detect periodic interaction"
        assert cell == pytest.approx(brute, abs=1e-12)

        lib.cell_list_free(ctypes.byref(cl))

    def test_delta_matches_two_calls(self, lib):
        """mol_repulsion_delta should equal new_e - old_e."""
        import numpy as np
        rng = np.random.default_rng(456)
        w = _CellListCtypes(lib)
        patch = 2000.0
        r_cut = 50.0
        eps = 2.0
        n_mol = 100

        pos, c_pos = w.make_positions(n_mol, patch, rng)
        cl = w.build_cell_list(c_pos, n_mol, r_cut, patch)

        for _ in range(20):
            idx = rng.integers(0, n_mol)
            old_p = (ctypes.c_double * 2)(pos[idx, 0], pos[idx, 1])
            new_xy = rng.uniform(0, patch, 2)
            new_p = (ctypes.c_double * 2)(new_xy[0], new_xy[1])

            old_e = lib.mol_repulsion_cell(old_p, idx, ctypes.byref(cl),
                                           c_pos, eps, r_cut, patch)
            new_e = lib.mol_repulsion_cell(new_p, idx, ctypes.byref(cl),
                                           c_pos, eps, r_cut, patch)
            delta = lib.mol_repulsion_delta(old_p, new_p, idx,
                                            ctypes.byref(cl), c_pos,
                                            eps, r_cut, patch)
            expected = new_e - old_e
            assert delta == pytest.approx(expected, abs=1e-10), (
                f"Delta mismatch at idx={idx}: delta={delta}, expected={expected}"
            )

        lib.cell_list_free(ctypes.byref(cl))
