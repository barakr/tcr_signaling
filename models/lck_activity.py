"""Lck activity model — active Lck spatial distribution.

Computes the mean active-Lck (Lck*) concentration within the tight-contact
zone, given the CD45 boundary density and Lck decay parameters.

Active Lck decays exponentially from the contact boundary inward with a
characteristic decay length. The model integrates this radial profile over
the contact disk to yield the mean Lck* level.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mean_lck_activity(
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
    mean_lck = peak * integral / disk_area
    return float(mean_lck)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lck activity model")
    parser.add_argument("--cd45_boundary_density", type=float, required=True)
    parser.add_argument("--lck_decay_length", type=float, required=True, help="um")
    parser.add_argument("--lck_activation_rate", type=float, required=True, help="1/s")
    parser.add_argument("--contact_radius", type=float, required=True, help="um")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_lck = _mean_lck_activity(
        args.cd45_boundary_density,
        args.lck_decay_length,
        args.lck_activation_rate,
        args.contact_radius,
    )

    payload = {
        "mean_lck_activity": mean_lck,
        "inputs": {
            "cd45_boundary_density": args.cd45_boundary_density,
            "lck_decay_length": args.lck_decay_length,
            "lck_activation_rate": args.lck_activation_rate,
            "contact_radius": args.contact_radius,
        },
    }
    (out_dir / "lck_activity.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
