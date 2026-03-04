"""Kinetic segregation model — CD45 exclusion from tight contacts.

Given a tight-contact geometry (contact fraction and perimeter), computes
the effective CD45 surface density outside the contact zone and the
boundary concentration of CD45 at the contact edge.

The model assumes CD45 is uniformly distributed outside the tight-contact
zone and fully excluded from within it (kinetic segregation hypothesis).

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cd45_boundary_density(
    cd45_bulk_density: float,
    contact_fraction: float,
) -> float:
    """CD45 density at the contact boundary (molecules/um^2).

    CD45 excluded from contact zone redistributes to the remaining area,
    increasing the effective density outside the contact.
    """
    free_fraction = 1.0 - contact_fraction
    if free_fraction <= 0:
        return 0.0
    return cd45_bulk_density / free_fraction


def main() -> int:
    parser = argparse.ArgumentParser(description="Kinetic segregation model")
    parser.add_argument("--contact_fraction", type=float, required=True)
    parser.add_argument("--cd45_bulk_density", type=float, required=True, help="molecules/um^2")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_density = _cd45_boundary_density(args.cd45_bulk_density, args.contact_fraction)

    payload = {
        "cd45_boundary_density": boundary_density,
        "inputs": {
            "contact_fraction": args.contact_fraction,
            "cd45_bulk_density": args.cd45_bulk_density,
        },
    }
    (out_dir / "segregation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
