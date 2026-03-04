"""CLI entrypoint for the TCR phosphorylation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import ptcr_density, ptcr_fraction


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

    fraction = ptcr_fraction(
        args.mean_lck_activity,
        args.phosphorylation_rate,
        args.dephosphorylation_rate,
    )
    density = ptcr_density(fraction, args.tcr_density)

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
