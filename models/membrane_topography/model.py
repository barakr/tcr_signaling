"""Membrane topography model — IRM-derived tight-contact geometry.

Generates a 2D binary contact map on a square patch. Tight-contact regions
are modeled as a circular disk of given radius centered in the patch.
The output is the fraction of the patch area that is in tight contact.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import numpy as np


def contact_fraction(contact_radius: float, patch_size: float) -> float:
    """Fraction of patch area within the tight-contact disk."""
    disk_area = np.pi * contact_radius**2
    patch_area = patch_size**2
    return float(min(disk_area / patch_area, 1.0))


def contact_perimeter(contact_radius: float) -> float:
    """Perimeter of the tight-contact disk in um."""
    return float(2.0 * np.pi * contact_radius)
