"""CLI entrypoint for the kinetic segregation model."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
    parser.add_argument("--n_tcr", type=lambda x: int(float(x)), default=50, help="Number of TCR molecules")
    parser.add_argument("--n_cd45", type=lambda x: int(float(x)), default=100, help="Number of CD45 molecules")
    parser.add_argument("--n_steps", type=lambda x: int(float(x)), default=None, help="MC steps (default: auto)")
    parser.add_argument("--grid_size", type=lambda x: int(float(x)), default=64, help="Membrane grid resolution")
    parser.add_argument("--D_mol", type=float, default=None, help="Molecular diffusion coeff (nm²/s, default 1e5)")
    parser.add_argument("--D_h", type=float, default=None, help="Membrane height diffusion coeff (nm²/s, default 5e4)")
    parser.add_argument("--dt", type=float, default=None, help="Override time step (seconds, auto if omitted)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive a unique reproducible seed per DOE point so each (time, rigidity)
    # combination gets independent MC noise while remaining deterministic.
    raw = struct.pack("dd", args.time_sec, args.rigidity_kT_nm2)
    input_hash = int(hashlib.md5(raw).hexdigest()[:8], 16)
    point_seed = args.seed + input_hash

    extra_kwargs = {}
    if args.D_mol is not None:
        extra_kwargs["D_mol"] = args.D_mol
    if args.D_h is not None:
        extra_kwargs["D_h"] = args.D_h
    if args.dt is not None:
        extra_kwargs["dt_override"] = args.dt

    result = simulate_ks(
        time_sec=args.time_sec,
        rigidity_kT_nm2=args.rigidity_kT_nm2,
        seed=point_seed,
        n_tcr=args.n_tcr,
        n_cd45=args.n_cd45,
        n_steps=args.n_steps,
        grid_size=args.grid_size,
        **extra_kwargs,
    )

    payload = {
        "depletion_width_nm": result["depletion_width_nm"],
        "diagnostics": {
            "final_tcr_mean_r_nm": result["final_tcr_mean_r_nm"],
            "final_cd45_mean_r_nm": result["final_cd45_mean_r_nm"],
            "accept_rate": result["accept_rate"],
            "n_steps_actual": result["n_steps_actual"],
            "dt_seconds": result["dt_seconds"],
            "step_size_h_nm": result["step_size_h_nm"],
            "step_size_mol_nm": result["step_size_mol_nm"],
            "D_mol_nm2_per_s": result["D_mol_nm2_per_s"],
            "D_h_nm2_per_s": result["D_h_nm2_per_s"],
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
