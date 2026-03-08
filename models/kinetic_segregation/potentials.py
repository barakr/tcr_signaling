"""Interaction potential functions for the kinetic segregation model.

Implements the energy terms governing membrane-molecule interactions in the
Monte Carlo simulation of TCR-CD45 spatial segregation on a 2D membrane patch.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bending_energy(h: NDArray[np.float64], kappa: float, dx: float) -> float:
    """Discrete membrane bending energy: E = (kappa/2) * sum(laplacian(h)^2) * dx^2.

    Uses periodic boundary conditions to avoid artificial edge effects.

    Parameters
    ----------
    h : 2D array of membrane heights (nm).
    kappa : Bending rigidity in kT (already in energy units).
    dx : Grid spacing in nm.
    """
    dx2 = dx * dx
    lap = (
        np.roll(h, 1, axis=0) + np.roll(h, -1, axis=0)
        + np.roll(h, 1, axis=1) + np.roll(h, -1, axis=1)
        - 4.0 * h
    ) / dx2
    return float(0.5 * kappa * np.sum(lap**2) * dx2)


def _lap_at(h: NDArray[np.float64], i: int, j: int, n: int, dx2: float) -> float:
    """Compute the discrete Laplacian at (i,j) with periodic BCs."""
    return (
        h[(i - 1) % n, j] + h[(i + 1) % n, j]
        + h[i, (j - 1) % n] + h[i, (j + 1) % n]
        - 4.0 * h[i, j]
    ) / dx2


def bending_energy_delta(
    h: NDArray[np.float64],
    kappa: float,
    dx: float,
    gi: int,
    gj: int,
    old_val: float,
    new_val: float,
) -> float:
    """Compute change in bending energy when h[gi,gj] changes from old_val to new_val.

    Instead of recomputing the full bending energy (O(N^2)), this computes the
    local delta in O(1) by only considering the affected Laplacian cells.
    Uses periodic BCs matching bending_energy().

    h must already contain new_val at h[gi,gj].
    """
    n = h.shape[0]
    dx2 = dx * dx
    delta_h = new_val - old_val
    delta_e = 0.0

    # Affected cells: (gi,gj) and its 4 periodic neighbors.
    affected = [
        (gi, gj),
        ((gi - 1) % n, gj),
        ((gi + 1) % n, gj),
        (gi, (gj - 1) % n),
        (gi, (gj + 1) % n),
    ]

    for ai, aj in affected:
        new_lap = _lap_at(h, ai, aj, n, dx2)
        # How much delta_h changed this cell's Laplacian:
        # Self: h[i,j] appears as -4*h in the stencil.
        # Neighbor: h[gi,gj] appears as +h in the stencil.
        if ai == gi and aj == gj:
            shift = -4.0 * delta_h / dx2
        else:
            shift = delta_h / dx2
        old_lap = new_lap - shift
        delta_e += new_lap * new_lap - old_lap * old_lap

    return float(0.5 * kappa * delta_e * dx2)


def tcr_pmhc_potential(h_at_mol: float, u_assoc: float, sigma_bind: float = 3.0) -> float:
    """Attractive Gaussian well for TCR at tight contact (h ~ 0).

    E = -U_assoc * exp(-h^2 / (2 * sigma^2))

    Parameters
    ----------
    h_at_mol : Membrane height at molecule position (nm).
    u_assoc : Binding potential depth in kT.
    sigma_bind : Width of the Gaussian well (nm), default 3 nm.
    """
    return float(-u_assoc * np.exp(-(h_at_mol**2) / (2.0 * sigma_bind**2)))


def mol_repulsion(
    pos: NDArray[np.float64],
    idx: int,
    all_pos: NDArray[np.float64],
    eps: float,
    r_cut: float,
    patch_size: float,
) -> float:
    """Soft pairwise repulsive potential between nearby molecules.

    Truncated harmonic repulsion: E = eps * (1 - r/r_cut)^2 for r < r_cut.

    Uses minimum-image convention for periodic boundaries.

    Parameters
    ----------
    pos : Position of the molecule being evaluated (shape (2,)).
    idx : Index of the molecule (excluded from self-interaction).
    all_pos : All positions of molecules of the same type (shape (N, 2)).
    eps : Repulsion strength (kT).
    r_cut : Cutoff distance (nm).
    patch_size : Simulation domain size for periodic wrapping (nm).
    """
    if eps <= 0.0 or r_cut <= 0.0:
        return 0.0
    total = 0.0
    half_patch = patch_size / 2.0
    for j in range(len(all_pos)):
        if j == idx:
            continue
        dx = pos[0] - all_pos[j, 0]
        dy = pos[1] - all_pos[j, 1]
        # Minimum image convention
        if dx > half_patch:
            dx -= patch_size
        elif dx < -half_patch:
            dx += patch_size
        if dy > half_patch:
            dy -= patch_size
        elif dy < -half_patch:
            dy += patch_size
        r = np.sqrt(dx * dx + dy * dy)
        if r < r_cut:
            ratio = 1.0 - r / r_cut
            total += eps * ratio * ratio
    return float(total)


def cd45_repulsion(
    h_at_mol: float, cd45_height: float = 35.0, k_rep: float = 1.0
) -> float:
    """Soft repulsive barrier for CD45 when membrane height < CD45 ectodomain height.

    CD45 has a large ectodomain (~35 nm) and is excluded from regions where the
    membrane gap is smaller than this height. We use a harmonic repulsive wall:

    E = 0.5 * k_rep * (cd45_height - h)^2   if h < cd45_height
    E = 0                                     otherwise

    Parameters
    ----------
    h_at_mol : Membrane height at CD45 position (nm).
    cd45_height : Ectodomain height (nm), default 35 nm.
    k_rep : Repulsive spring constant (kT/nm²), default 1.0.
    """
    if h_at_mol < cd45_height:
        return float(0.5 * k_rep * (cd45_height - h_at_mol) ** 2)
    return 0.0
