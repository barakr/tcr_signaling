"""CLI entrypoint for the membrane topography model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import contact_fraction, contact_perimeter


def main() -> int:
    parser = argparse.ArgumentParser(description="Membrane topography model")
    parser.add_argument("--contact_radius", type=float, required=True, help="um")
    parser.add_argument("--patch_size", type=float, required=True, help="um")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    fraction = contact_fraction(args.contact_radius, args.patch_size)
    perimeter = contact_perimeter(args.contact_radius)

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
