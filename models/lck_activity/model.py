"""Lck activity model — active Lck spatial distribution.

Computes the mean active-Lck (Lck*) concentration within the tight-contact
zone, given the CD45 boundary density and Lck decay parameters.

Active Lck decays exponentially from the contact boundary inward with a
characteristic decay length. The model integrates this radial profile over
the contact disk to yield the mean Lck* level.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import numpy as np


def mean_lck_activity(
    cd45_boundary_density: float,
    lck_decay_length: float,
    lck_activation_rate: float,
    contact_radius: float,
) -> float:
    """Mean Lck* concentration inside the contact disk.

    Lck* = activation_rate * cd45_boundary * exp(-r / decay_length),
    integrated over the disk and normalized by disk area.
    Uses analytical integration in polar coordinates.
    """
    if contact_radius <= 0 or lck_decay_length <= 0:
        return 0.0
    lam = lck_decay_length
    R = contact_radius
    peak = lck_activation_rate * cd45_boundary_density
    # Integrate r * exp(-(R - r) / lam) from 0 to R (Lck decays inward from edge)
    # = exp(-R/lam) * integral_0^R r * exp(r/lam) dr
    # = exp(-R/lam) * lam * [R*exp(R/lam) - lam*(exp(R/lam) - 1)]
    # = lam * [R - lam*(1 - exp(-R/lam))]
    integral = lam * (R - lam * (1.0 - np.exp(-R / lam)))
    disk_area = 0.5 * R**2  # half R^2 from polar integration (2pi cancels)
    if disk_area <= 0:
        return 0.0
    return float(peak * integral / disk_area)
