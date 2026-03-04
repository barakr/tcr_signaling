"""TCR phosphorylation model — pTCR generation from Lck* and TCR density.

Computes the steady-state fraction of phosphorylated TCR ITAMs given
the mean active Lck concentration, TCR surface density, and the
phosphorylation / dephosphorylation rates.

At steady state: pTCR_fraction = (k_phos * Lck*) / (k_phos * Lck* + k_dephos)

This is validated against ZAP-70 super-resolution recruitment data.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ptcr_fraction(
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


def _ptcr_density(
    ptcr_fraction: float,
    tcr_density: float,
) -> float:
    """Absolute density of pTCR molecules (molecules/um^2)."""
    return ptcr_fraction * tcr_density


def main() -> int:
    parser = argparse.ArgumentParser(description="TCR phosphorylation model")
    parser.add_argument("--mean_lck_activity", type=float, required=True)
    parser.add_argument("--tcr_density", type=float, required=True, help="molecules/um^2")
    parser.add_argument("--phosphorylation_rate", type=float, required=True, help="1/s")
    parser.add_argument("--dephosphorylation_rate", type=float, required=True, help="1/s")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    fraction = _ptcr_fraction(
        args.mean_lck_activity,
        args.phosphorylation_rate,
        args.dephosphorylation_rate,
    )
    density = _ptcr_density(fraction, args.tcr_density)

    payload = {
        "ptcr_fraction": fraction,
        "ptcr_density": density,
        "inputs": {
            "mean_lck_activity": args.mean_lck_activity,
            "tcr_density": args.tcr_density,
            "phosphorylation_rate": args.phosphorylation_rate,
            "dephosphorylation_rate": args.dephosphorylation_rate,
        },
    }
    (out_dir / "phosphorylation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
