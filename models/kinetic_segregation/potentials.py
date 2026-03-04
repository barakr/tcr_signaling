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

    Parameters
    ----------
    h : 2D array of membrane heights (nm).
    kappa : Bending rigidity in kT (already in energy units).
    dx : Grid spacing in nm.
    """
    # Discrete Laplacian via finite differences with zero-padding at boundaries
    lap = np.zeros_like(h)
    lap[1:-1, :] += h[:-2, :] + h[2:, :] - 2.0 * h[1:-1, :]
    lap[:, 1:-1] += h[:, :-2] + h[:, 2:] - 2.0 * h[:, 1:-1]
    lap /= dx**2
    return float(0.5 * kappa * np.sum(lap**2) * dx**2)


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


def cd45_repulsion(h_at_mol: float, cd45_height: float = 35.0) -> float:
    """Soft repulsive barrier for CD45 when membrane height < CD45 ectodomain height.

    CD45 has a large ectodomain (~35 nm) and is excluded from regions where the
    membrane gap is smaller than this height. We use a harmonic repulsive wall:

    E = 0.5 * k_rep * (cd45_height - h)^2   if h < cd45_height
    E = 0                                     otherwise

    Parameters
    ----------
    h_at_mol : Membrane height at CD45 position (nm).
    cd45_height : Ectodomain height (nm), default 35 nm.
    """
    k_rep = 1.0  # repulsive spring constant in kT/nm^2
    if h_at_mol < cd45_height:
        return float(0.5 * k_rep * (cd45_height - h_at_mol) ** 2)
    return 0.0
