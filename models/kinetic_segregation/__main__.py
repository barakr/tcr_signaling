"""CLI entrypoint for the kinetic segregation model."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from .model import simulate_ks


def _merge_params(args: argparse.Namespace, params_dict: dict) -> None:
    """Apply param file values where CLI left defaults (None)."""
    for key, val in params_dict.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, val)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kinetic segregation Monte Carlo model")
    parser.add_argument("--params", type=str, default=None, help="JSON parameter file")
    parser.add_argument("--time_sec", type=float, default=None, help="Simulation time (sec)")
    parser.add_argument(
        "--rigidity_kT_nm2", type=float, default=None, help="Membrane bending rigidity (kT)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n_tcr", type=lambda x: int(float(x)), default=None, help="Number of TCR molecules")
    parser.add_argument("--n_cd45", type=lambda x: int(float(x)), default=None, help="Number of CD45 molecules")
    parser.add_argument("--n_steps", type=lambda x: int(float(x)), default=None, help="MC steps (default: auto)")
    parser.add_argument("--grid_size", type=lambda x: int(float(x)), default=None, help="Membrane grid resolution")
    parser.add_argument("--D_mol", type=float, default=None, help="Molecular diffusion coeff (nm²/s, default 1e5)")
    parser.add_argument("--D_h", type=float, default=None, help="Membrane height diffusion coeff (nm²/s, default 5e4)")
    parser.add_argument("--dt", type=float, default=None, help="Override time step (seconds, auto if omitted)")
    parser.add_argument("--cd45_height", type=float, default=None, help="CD45 ectodomain height (nm, default 35)")
    parser.add_argument("--cd45_k_rep", type=float, default=None, help="CD45 repulsive spring constant (kT/nm², default 1.0)")
    parser.add_argument("--mol_repulsion_eps", type=float, default=None, help="Soft molecular repulsion strength (kT, default 0=off)")
    parser.add_argument("--mol_repulsion_rcut", type=float, default=None, help="Soft molecular repulsion cutoff (nm, default 10)")
    parser.add_argument("--n_pmhc", type=lambda x: int(float(x)), default=None, help="Number of static pMHC molecules (0=all cells have pMHC)")
    parser.add_argument("--pmhc_seed", type=int, default=None, help="Seed for pMHC random positions")
    parser.add_argument("--pmhc_mode", type=str, default=None, help="pMHC placement: 'uniform' or 'inner_circle'")
    parser.add_argument("--pmhc_radius", type=float, default=None, help="pMHC placement radius (nm)")
    parser.add_argument("--h0_tcr", type=float, default=None, help="TCR-pMHC bond length (nm, default 13)")
    parser.add_argument("--init_height", type=float, default=None, help="Initial membrane height (nm, default 70)")
    parser.add_argument("--binding_mode", type=str, default=None, help="'forced' or 'gaussian'")
    parser.add_argument("--step_mode", type=str, default=None, help="'paper' or 'brownian'")
    args = parser.parse_args()

    # Load param file and merge (CLI > param file > built-in defaults)
    if args.params is not None:
        with open(args.params) as f:
            _merge_params(args, json.load(f))

    # Apply built-in defaults for required/defaulted params
    if args.seed is None:
        args.seed = 42
    if args.n_tcr is None:
        args.n_tcr = 125
    if args.n_cd45 is None:
        args.n_cd45 = 500
    if args.grid_size is None:
        args.grid_size = 64

    # Validate required params
    if args.time_sec is None:
        parser.error("--time_sec is required (via CLI or param file)")
    if args.rigidity_kT_nm2 is None:
        parser.error("--rigidity_kT_nm2 is required (via CLI or param file)")

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
    if args.cd45_height is not None:
        extra_kwargs["cd45_height"] = args.cd45_height
    if args.cd45_k_rep is not None:
        extra_kwargs["cd45_k_rep"] = args.cd45_k_rep
    if args.mol_repulsion_eps is not None:
        extra_kwargs["mol_repulsion_eps"] = args.mol_repulsion_eps
    if args.mol_repulsion_rcut is not None:
        extra_kwargs["mol_repulsion_rcut"] = args.mol_repulsion_rcut
    if args.n_pmhc is not None:
        extra_kwargs["n_pmhc"] = args.n_pmhc
    if args.pmhc_seed is not None:
        extra_kwargs["pmhc_seed"] = args.pmhc_seed
    if args.pmhc_mode is not None:
        extra_kwargs["pmhc_mode"] = args.pmhc_mode
    if args.pmhc_radius is not None:
        extra_kwargs["pmhc_radius"] = args.pmhc_radius
    if args.h0_tcr is not None:
        extra_kwargs["h0_tcr"] = args.h0_tcr
    if args.init_height is not None:
        extra_kwargs["init_height"] = args.init_height
    if args.binding_mode is not None:
        extra_kwargs["binding_mode"] = args.binding_mode
    if args.step_mode is not None:
        extra_kwargs["step_mode"] = args.step_mode

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
