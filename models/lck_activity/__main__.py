"""CLI entrypoint for the Lck activity model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import mean_lck_activity


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

    mean_lck = mean_lck_activity(
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
