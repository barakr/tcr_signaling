"""Membrane topography model — IRM-derived tight-contact geometry.

Generates a 2D binary contact map on a square patch. Tight-contact regions
are modeled as a circular disk of given radius centered in the patch.
The output is the fraction of the patch area that is in tight contact.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _contact_fraction(contact_radius: float, patch_size: float) -> float:
    """Fraction of patch area within the tight-contact disk."""
    disk_area = np.pi * contact_radius**2
    patch_area = patch_size**2
    return float(min(disk_area / patch_area, 1.0))


def _contact_perimeter(contact_radius: float) -> float:
    """Perimeter of the tight-contact disk in um."""
    return float(2.0 * np.pi * contact_radius)


def main() -> int:
    parser = argparse.ArgumentParser(description="Membrane topography model")
    parser.add_argument("--contact_radius", type=float, required=True, help="um")
    parser.add_argument("--patch_size", type=float, required=True, help="um")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    fraction = _contact_fraction(args.contact_radius, args.patch_size)
    perimeter = _contact_perimeter(args.contact_radius)

    payload = {
        "contact_fraction": fraction,
        "contact_perimeter_um": perimeter,
        "inputs": {
            "contact_radius": args.contact_radius,
            "patch_size": args.patch_size,
        },
    }
    (out_dir / "topography.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
