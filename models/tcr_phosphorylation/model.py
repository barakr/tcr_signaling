"""TCR phosphorylation model — pTCR generation from Lck* and TCR density.

Computes the steady-state fraction of phosphorylated TCR ITAMs given
the mean active Lck concentration, TCR surface density, and the
phosphorylation / dephosphorylation rates.

At steady state: pTCR_fraction = (k_phos * Lck*) / (k_phos * Lck* + k_dephos)

This is validated against ZAP-70 super-resolution recruitment data.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations


def ptcr_fraction(
    mean_lck_activity: float,
    phosphorylation_rate: float,
    dephosphorylation_rate: float,
) -> float:
    """Steady-state fraction of phosphorylated TCR ITAMs."""
    forward = phosphorylation_rate * mean_lck_activity
    total = forward + dephosphorylation_rate
    if total <= 0:
        return 0.0
    return forward / total


def ptcr_density(
    ptcr_frac: float,
    tcr_density: float,
) -> float:
    """Absolute density of pTCR molecules (molecules/um^2)."""
    return ptcr_frac * tcr_density
