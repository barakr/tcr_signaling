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

    # mol_repulsion(pos, idx, all_pos, n_mol, eps, r_cut, patch_size) -> double
    lib.mol_repulsion.restype = ctypes.c_double
    lib.mol_repulsion.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
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


# ── Additional mol_repulsion brute-force tests ──────────────────────────

class TestMolRepulsionBruteForce:
    """Unit tests for the brute-force mol_repulsion function."""

    def test_single_molecule(self, lib):
        """Single molecule has zero repulsion (no neighbors)."""
        pos = (ctypes.c_double * 2)(500.0, 500.0)
        e = lib.mol_repulsion(pos, 0, pos, 1, 2.0, 50.0, 2000.0)
        assert e == pytest.approx(0.0, abs=1e-15)

    def test_two_distant_molecules(self, lib):
        """Two molecules far apart have zero repulsion."""
        c_pos = (ctypes.c_double * 4)(100.0, 100.0, 900.0, 900.0)
        p0 = (ctypes.c_double * 2)(100.0, 100.0)
        e = lib.mol_repulsion(p0, 0, c_pos, 2, 2.0, 50.0, 2000.0)
        assert e == pytest.approx(0.0, abs=1e-15)

    def test_two_close_molecules(self, lib):
        """Two molecules within r_cut have expected repulsion."""
        r_cut, eps = 50.0, 2.0
        c_pos = (ctypes.c_double * 4)(500.0, 500.0, 520.0, 500.0)
        p0 = (ctypes.c_double * 2)(500.0, 500.0)
        e = lib.mol_repulsion(p0, 0, c_pos, 2, eps, r_cut, 2000.0)
        expected = eps * (1.0 - 20.0 / r_cut) ** 2
        assert e == pytest.approx(expected, abs=1e-12)

    def test_self_exclusion(self, lib):
        """mol_repulsion should not include self-interaction."""
        c_pos = (ctypes.c_double * 6)(500.0, 500.0, 500.0, 500.0, 500.0, 500.0)
        p0 = (ctypes.c_double * 2)(500.0, 500.0)
        e = lib.mol_repulsion(p0, 0, c_pos, 3, 2.0, 50.0, 2000.0)
        # 2 neighbors at r=0: each contributes eps * (1-0)^2 = 2.0
        assert e == pytest.approx(4.0, abs=1e-12)

    def test_analytical_formula(self, lib):
        """Verify E = eps * (1 - r/r_cut)^2 for multiple distances."""
        r_cut, eps, patch = 50.0, 3.0, 2000.0
        for r in [5.0, 10.0, 25.0, 40.0, 49.9]:
            c_pos = (ctypes.c_double * 4)(500.0, 500.0, 500.0 + r, 500.0)
            p0 = (ctypes.c_double * 2)(500.0, 500.0)
            e = lib.mol_repulsion(p0, 0, c_pos, 2, eps, r_cut, patch)
            expected = eps * (1.0 - r / r_cut) ** 2
            assert e == pytest.approx(expected, abs=1e-10), f"Mismatch at r={r}"

    def test_exactly_at_cutoff(self, lib):
        """At exactly r_cut, repulsion should be zero."""
        c_pos = (ctypes.c_double * 4)(500.0, 500.0, 550.0, 500.0)
        p0 = (ctypes.c_double * 2)(500.0, 500.0)
        e = lib.mol_repulsion(p0, 0, c_pos, 2, 2.0, 50.0, 2000.0)
        assert e == pytest.approx(0.0, abs=1e-12)


# ── Cell-list incremental move tests ────────────────────────────────────

class TestCellListMove:
    """Unit tests for cell_list_move (incremental linked-list update)."""

    def test_move_matches_rebuild(self, lib):
        """After cell_list_move, repulsion should match a full rebuild."""
        import numpy as np
        w = _CellListCtypes(lib)
        rng = np.random.default_rng(789)
        patch, r_cut = 2000.0, 50.0
        n_mol = 50

        pos, c_pos = w.make_positions(n_mol, patch, rng)
        cl = w.build_cell_list(c_pos, n_mol, r_cut, patch)

        lib.cell_list_move.restype = None
        lib.cell_list_move.argtypes = [
            ctypes.POINTER(w.CellListStruct), ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]

        idx = 10
        nc = cl.nc
        old_ci = max(0, min(int(pos[idx, 0] * cl.inv_cell_size), nc - 1))
        old_cj = max(0, min(int(pos[idx, 1] * cl.inv_cell_size), nc - 1))
        old_cell = old_ci * nc + old_cj

        new_x, new_y = 100.0, 100.0
        new_ci = max(0, min(int(new_x * cl.inv_cell_size), nc - 1))
        new_cj = max(0, min(int(new_y * cl.inv_cell_size), nc - 1))
        new_cell = new_ci * nc + new_cj

        lib.cell_list_move(ctypes.byref(cl), idx, old_cell, new_cell)
        c_pos[2 * idx] = new_x
        c_pos[2 * idx + 1] = new_y

        cl2 = w.build_cell_list(c_pos, n_mol, r_cut, patch)

        for test_idx in [0, idx, n_mol - 1]:
            p = (ctypes.c_double * 2)(c_pos[2 * test_idx], c_pos[2 * test_idx + 1])
            e_inc = lib.mol_repulsion_cell(p, test_idx, ctypes.byref(cl),
                                            c_pos, 2.0, r_cut, patch)
            e_rebuild = lib.mol_repulsion_cell(p, test_idx, ctypes.byref(cl2),
                                                c_pos, 2.0, r_cut, patch)
            assert e_inc == pytest.approx(e_rebuild, abs=1e-12), (
                f"Move vs rebuild mismatch at idx={test_idx}"
            )

        lib.cell_list_free(ctypes.byref(cl))
        lib.cell_list_free(ctypes.byref(cl2))

    def test_move_same_cell_noop(self, lib):
        """Moving within the same cell should be a no-op."""
        w = _CellListCtypes(lib)
        c_pos = (ctypes.c_double * 4)(25.0, 25.0, 26.0, 26.0)
        cl = w.build_cell_list(c_pos, 2, 50.0, 2000.0)

        lib.cell_list_move.restype = None
        lib.cell_list_move.argtypes = [
            ctypes.POINTER(w.CellListStruct), ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        nc = cl.nc
        ci = max(0, min(int(25.0 * cl.inv_cell_size), nc - 1))
        cj = max(0, min(int(25.0 * cl.inv_cell_size), nc - 1))
        cell = ci * nc + cj

        lib.cell_list_move(ctypes.byref(cl), 0, cell, cell)

        p0 = (ctypes.c_double * 2)(25.0, 25.0)
        e = lib.mol_repulsion_cell(p0, 0, ctypes.byref(cl), c_pos, 2.0, 50.0, 2000.0)
        brute = lib.mol_repulsion(p0, 0, c_pos, 2, 2.0, 50.0, 2000.0)
        assert e == pytest.approx(brute, abs=1e-12)

        lib.cell_list_free(ctypes.byref(cl))


# ── Bending energy delta edge cases ─────────────────────────────────────

class TestBendingEnergyDeltaEdgeCases:
    """Test bending energy delta at boundaries and special cases."""

    def test_corner_cell(self, lib):
        """Delta computation at grid corners uses periodic wrapping."""
        import numpy as np
        rng = np.random.default_rng(111)
        n = 8
        kappa, dx = 10.0, 50.0
        h_np = rng.uniform(30, 70, (n, n))

        for gi, gj in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
            old_val = h_np[gi, gj]
            new_val = old_val + 5.0
            h_tmp = h_np.copy()
            h_tmp[gi, gj] = new_val
            h_flat = h_tmp.flatten().astype(np.float64)
            h_c = (ctypes.c_double * len(h_flat))(*h_flat)
            delta = lib.bending_energy_delta(h_c, n, kappa, dx, gi, gj, old_val, new_val)
            assert abs(delta) > 0.0, f"Zero delta at corner ({gi},{gj})"

    def test_symmetric_perturbation(self, lib):
        """+dh and -dh on a flat membrane should give equal energy deltas."""
        import numpy as np
        n, kappa, dx, h0, dh = 16, 10.0, 50.0, 50.0, 3.0
        gi, gj = 8, 8

        h_pos = np.full(n * n, h0)
        h_pos[gi * n + gj] = h0 + dh
        h_neg = np.full(n * n, h0)
        h_neg[gi * n + gj] = h0 - dh

        h_c_pos = (ctypes.c_double * len(h_pos))(*h_pos)
        h_c_neg = (ctypes.c_double * len(h_neg))(*h_neg)

        d_pos = lib.bending_energy_delta(h_c_pos, n, kappa, dx, gi, gj, h0, h0 + dh)
        d_neg = lib.bending_energy_delta(h_c_neg, n, kappa, dx, gi, gj, h0, h0 - dh)
        assert d_pos == pytest.approx(d_neg, rel=1e-10)

    def test_scales_with_kappa(self, lib):
        """Bending energy should scale linearly with kappa."""
        import numpy as np
        rng = np.random.default_rng(222)
        n, dx = 16, 50.0
        h_np = rng.uniform(30, 70, (n, n))
        gi, gj = 5, 5
        old_val = h_np[gi, gj]
        new_val = old_val + 3.0
        h_np[gi, gj] = new_val
        h_flat = h_np.flatten().astype(np.float64)

        h_c1 = (ctypes.c_double * len(h_flat))(*h_flat)
        h_c2 = (ctypes.c_double * len(h_flat))(*h_flat)
        d1 = lib.bending_energy_delta(h_c1, n, 10.0, dx, gi, gj, old_val, new_val)
        d2 = lib.bending_energy_delta(h_c2, n, 20.0, dx, gi, gj, old_val, new_val)
        assert d2 == pytest.approx(2.0 * d1, rel=1e-10)

    def test_matches_numerical_finite_difference(self, lib):
        """Compare C delta against full-grid bending energy finite difference."""
        import numpy as np
        rng = np.random.default_rng(333)
        n, kappa, dx = 8, 10.0, 50.0
        h_np = rng.uniform(30, 70, (n, n))
        gi, gj = 3, 5
        old_val = h_np[gi, gj]
        new_val = old_val + 2.0

        def _total_bending_energy(h, n, kappa, dx):
            dx2 = dx * dx
            E = 0.0
            for i in range(n):
                for j in range(n):
                    lap = (h[(i-1)%n, j] + h[(i+1)%n, j] +
                           h[i, (j-1)%n] + h[i, (j+1)%n] - 4*h[i, j]) / dx2
                    E += lap ** 2
            return 0.5 * kappa * E * dx2

        E_old = _total_bending_energy(h_np, n, kappa, dx)
        h_new = h_np.copy()
        h_new[gi, gj] = new_val
        E_new = _total_bending_energy(h_new, n, kappa, dx)

        h_flat = h_new.flatten().astype(np.float64)
        h_c = (ctypes.c_double * len(h_flat))(*h_flat)
        c_delta = lib.bending_energy_delta(h_c, n, kappa, dx, gi, gj, old_val, new_val)

        assert c_delta == pytest.approx(E_new - E_old, rel=1e-6)
