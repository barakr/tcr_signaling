"""CLI wrapper: delegates to the compiled ks_gpu binary via subprocess."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _find_binary() -> Path:
    """Locate the ks_gpu binary relative to this package."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "ks_gpu",
        pkg_dir / "build" / "ks_gpu",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(
        f"ks_gpu binary not found. Build it first: cd {pkg_dir} && make"
    )


def _merge_params(args: argparse.Namespace, params_dict: dict) -> None:
    """Apply param file values where CLI left defaults (None)."""
    for key, val in params_dict.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, val)


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU-accelerated kinetic segregation model")
    parser.add_argument("--params", type=str, default=None, help="JSON parameter file")
    parser.add_argument("--time_sec", type=float, default=None)
    parser.add_argument("--rigidity_kT_nm2", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n_tcr", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--n_cd45", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--n_steps", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--grid_size", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--no-gpu", action="store_true", help="Disable Metal GPU")
    parser.add_argument("--D_mol", type=float, default=None, help="Molecular diffusion coeff (nm²/s)")
    parser.add_argument("--D_h", type=float, default=None, help="Membrane height diffusion coeff (nm²/s)")
    parser.add_argument("--dt", type=float, default=None, help="Override time step (seconds)")
    parser.add_argument("--cd45_height", type=float, default=None, help="CD45 ectodomain height (nm)")
    parser.add_argument("--cd45_k_rep", type=float, default=None, help="CD45 repulsive spring constant (kT/nm²)")
    parser.add_argument("--mol_repulsion_eps", type=float, default=None, help="Soft molecular repulsion strength (kT)")
    parser.add_argument("--mol_repulsion_rcut", type=float, default=None, help="Soft molecular repulsion cutoff (nm)")
    parser.add_argument("--n_pmhc", type=lambda x: int(float(x)), default=None, help="Number of static pMHC molecules")
    parser.add_argument("--pmhc_seed", type=int, default=None, help="Seed for pMHC positions")
    parser.add_argument("--pmhc_mode", type=str, default=None, help="pMHC placement: 'uniform' or 'inner_circle'")
    parser.add_argument("--pmhc_radius", type=float, default=None, help="pMHC placement radius (nm)")
    parser.add_argument("--binding_mode", type=str, default=None, help="'forced' or 'gaussian'")
    parser.add_argument("--step_mode", type=str, default=None, help="'paper' or 'brownian'")
    parser.add_argument("--h0_tcr", type=float, default=None, help="TCR-pMHC bond length (nm)")
    parser.add_argument("--init_height", type=float, default=None, help="Initial membrane height (nm)")
    parser.add_argument("--dump-frames", action="store_true", help="Dump binary frame files for movie rendering")
    parser.add_argument("--dump-interval", type=int, default=None, help="Dump every N steps (default: 1)")
    parser.add_argument("--grid-substeps", type=int, default=None, help="Grid Phase 2 substeps per molecular move (default: 1)")
    args = parser.parse_args()

    # Load param file and merge (CLI > param file > built-in defaults)
    if args.params is not None:
        with open(args.params) as f:
            _merge_params(args, json.load(f))

    # Apply built-in defaults
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

    binary = _find_binary()
    cmd = [
        str(binary),
        "--time_sec", str(args.time_sec),
        "--rigidity_kT_nm2", str(args.rigidity_kT_nm2),
        "--seed", str(args.seed),
        "--run-dir", args.run_dir,
        "--n_tcr", str(args.n_tcr),
        "--n_cd45", str(args.n_cd45),
        "--grid_size", str(args.grid_size),
    ]
    if args.n_steps is not None:
        cmd.extend(["--n_steps", str(args.n_steps)])
    if args.no_gpu:
        cmd.append("--no-gpu")
    if args.D_mol is not None:
        cmd.extend(["--D_mol", str(args.D_mol)])
    if args.D_h is not None:
        cmd.extend(["--D_h", str(args.D_h)])
    if args.dt is not None:
        cmd.extend(["--dt", str(args.dt)])
    if args.cd45_height is not None:
        cmd.extend(["--cd45_height", str(args.cd45_height)])
    if args.cd45_k_rep is not None:
        cmd.extend(["--cd45_k_rep", str(args.cd45_k_rep)])
    if args.mol_repulsion_eps is not None:
        cmd.extend(["--mol_repulsion_eps", str(args.mol_repulsion_eps)])
    if args.mol_repulsion_rcut is not None:
        cmd.extend(["--mol_repulsion_rcut", str(args.mol_repulsion_rcut)])
    if args.n_pmhc is not None:
        cmd.extend(["--n_pmhc", str(args.n_pmhc)])
    if args.pmhc_seed is not None:
        cmd.extend(["--pmhc_seed", str(args.pmhc_seed)])
    if args.pmhc_mode is not None:
        cmd.extend(["--pmhc_mode", str(args.pmhc_mode)])
    if args.pmhc_radius is not None:
        cmd.extend(["--pmhc_radius", str(args.pmhc_radius)])
    if args.binding_mode is not None:
        cmd.extend(["--binding_mode", str(args.binding_mode)])
    if args.step_mode is not None:
        cmd.extend(["--step_mode", str(args.step_mode)])
    if args.h0_tcr is not None:
        cmd.extend(["--h0_tcr", str(args.h0_tcr)])
    if args.init_height is not None:
        cmd.extend(["--init_height", str(args.init_height)])
    if args.dump_frames:
        cmd.append("--dump-frames")
    if args.dump_interval is not None:
        cmd.extend(["--dump-interval", str(args.dump_interval)])
    if args.grid_substeps is not None:
        cmd.extend(["--grid-substeps", str(args.grid_substeps)])
    if args.params is not None:
        cmd.extend(["--params", args.params])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    # Validate and forward JSON output
    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"Invalid JSON output from binary: {result.stdout[:200]}", file=sys.stderr)
        return 1

    print(json.dumps(data, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
