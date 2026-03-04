"""CLI entrypoint for the kinetic segregation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import simulate_ks


def main() -> int:
    parser = argparse.ArgumentParser(description="Kinetic segregation Monte Carlo model")
    parser.add_argument("--time_sec", type=float, required=True, help="Simulation time (sec)")
    parser.add_argument(
        "--rigidity_kT_nm2", type=float, required=True, help="Membrane bending rigidity (kT)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n_tcr", type=int, default=50, help="Number of TCR molecules")
    parser.add_argument("--n_cd45", type=int, default=100, help="Number of CD45 molecules")
    parser.add_argument("--n_steps", type=int, default=None, help="MC steps (default: auto)")
    parser.add_argument("--grid_size", type=int, default=32, help="Membrane grid resolution")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = simulate_ks(
        time_sec=args.time_sec,
        rigidity_kT_nm2=args.rigidity_kT_nm2,
        seed=args.seed,
        n_tcr=args.n_tcr,
        n_cd45=args.n_cd45,
        n_steps=args.n_steps,
        grid_size=args.grid_size,
    )

    payload = {
        "depletion_width_nm": result["depletion_width_nm"],
        "diagnostics": {
            "final_tcr_mean_r_nm": result["final_tcr_mean_r_nm"],
            "final_cd45_mean_r_nm": result["final_cd45_mean_r_nm"],
            "accept_rate": result["accept_rate"],
            "n_steps_actual": result["n_steps_actual"],
        },
        "inputs": {
            "time_sec": args.time_sec,
            "rigidity_kT_nm2": args.rigidity_kT_nm2,
        },
    }
    (out_dir / "segregation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
