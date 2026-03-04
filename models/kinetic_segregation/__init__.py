"""Kinetic segregation model — TCR-CD45 spatial segregation on a 2D membrane.

2D Monte Carlo simulation of molecular segregation driven by membrane bending
rigidity and TCR-pMHC binding. Outputs the depletion zone width (nm).

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

from .model import simulate_ks

__all__ = ["simulate_ks"]
