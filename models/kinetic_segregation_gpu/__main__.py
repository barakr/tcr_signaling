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


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU-accelerated kinetic segregation model")
    parser.add_argument("--time_sec", type=float, required=True)
    parser.add_argument("--rigidity_kT_nm2", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n_tcr", type=lambda x: int(float(x)), default=50)
    parser.add_argument("--n_cd45", type=lambda x: int(float(x)), default=100)
    parser.add_argument("--n_steps", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--grid_size", type=lambda x: int(float(x)), default=64)
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
    args = parser.parse_args()

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
