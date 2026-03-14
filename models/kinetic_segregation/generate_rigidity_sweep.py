"""Generate movies sweeping membrane rigidity with gaussian binding + pMHC influence.

Small membrane patch (250 nm), grid 50×50, gaussian binding + excluded volume.
Fixed molecule counts: 30 TCR/pMHC, 30 CD45.
Rigidity values: log-spaced 1–100 kT/nm².
GPU only.  Movies saved to ~/Downloads/ks_adjust_gaussian/.

Uses --patch_size 250 and --sigma_r 2.0 (runtime params, no rebuild needed).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_BINARY = _PKG_DIR / "ks_gpu"
_RENDER = _PKG_DIR / "render_movie.py"
_OUTPUT_DIR = Path.home() / "Downloads" / "ks_adjust_gaussian"
_RENDER_PYTHON = Path.home() / "miniconda3" / "envs" / "py314_bayesmm" / "bin" / "python"
_RENDER_ENV = {
    **os.environ,
    "PATH": str(_RENDER_PYTHON.parent) + os.pathsep + os.environ.get("PATH", ""),
}

# Fixed parameters
GRID_SIZE = 50
TIME_SEC = 10       # 10s simulation
N_TCR = 30
N_CD45 = 30
SEED = 42

# Step size: dt=1e-5 → step_mol = sqrt(2*10000*1e-5) ≈ 0.45 nm
DT = 1e-5           # fine step for accurate binding dynamics
N_STEPS_TOTAL = int(TIME_SEC / DT)  # 400,000
DUMP_INTERVAL = N_STEPS_TOTAL // 200  # ~200 frames → ~13s movie at 15fps

# Rigidity values to sweep (kT/nm²) — log-spaced from 1 to 100
RIGIDITY_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# Patch size (runtime, no rebuild needed)
PATCH_SIZE = 250.0

# pMHC placement: inner_circle, radius = patch/12 ≈ 21 nm
PMHC_RADIUS = 21

# Gaussian binding parameters
SIGMA_R = 2.0  # lateral pMHC influence range (nm)

# Excluded volume parameters
MOL_REPULSION_EPS = 2.0
MOL_REPULSION_RCUT = 10.0


def run_sim(run_dir: Path, rigidity: float) -> float:
    """Run GPU simulation with gaussian binding + repulsion. Return wall time."""
    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT", str(rigidity),
        "--seed", str(SEED),
        "--grid_size", str(GRID_SIZE),
        "--n_tcr", str(N_TCR),
        "--n_cd45", str(N_CD45),
        "--n_pmhc", str(N_TCR),  # n_pmhc matches n_tcr
        "--pmhc_radius", str(PMHC_RADIUS),
        "--binding_mode", "gaussian",
        "--dt", str(DT),
        "--mol_repulsion_eps", str(MOL_REPULSION_EPS),
        "--mol_repulsion_rcut", str(MOL_REPULSION_RCUT),
        "--patch_size", str(PATCH_SIZE),
        "--sigma_r", str(SIGMA_R),
        "--run-dir", str(run_dir),
        "--dump-frames",
        "--dump-interval", str(DUMP_INTERVAL),
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}):", file=sys.stderr)
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
        print(f"  stdout: {result.stdout[:500]}", file=sys.stderr)
        return -1.0
    # Print last few lines of stdout for summary
    lines = result.stdout.strip().split("\n")
    for line in lines[-5:]:
        print(f"  {line}")
    return elapsed


def render_movie(frames_dir: Path, output: Path, rigidity: float):
    """Render movie from frames with pMHC markers shown."""
    cmd = [
        str(_RENDER_PYTHON), str(_RENDER),
        str(frames_dir),
        "-o", str(output),
        "--fps", "15",
        "--dpi", "120",
        "--rigidity", str(rigidity),
        "--show-pmhc",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            env=_RENDER_ENV)
    if result.returncode != 0:
        print(f"  Render failed: {result.stderr[:500]}", file=sys.stderr)
        return False
    return True


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(RIGIDITY_VALUES)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i, rigidity in enumerate(RIGIDITY_VALUES, 1):
            tag = f"rig{rigidity:.0f}"
            print(f"\n[{i}/{total}] rigidity={rigidity} kT/nm² "
                  f"(grid={GRID_SIZE}, n_tcr={N_TCR}, n_cd45={N_CD45}, {TIME_SEC}s)")

            # Run simulation
            sim_dir = tmp_path / tag
            t_wall = run_sim(sim_dir, rigidity)
            if t_wall < 0:
                continue
            print(f"  Simulation: {t_wall:.1f}s wall time")

            # Render movie
            frames_dir = sim_dir / "frames"
            if frames_dir.exists():
                movie_name = f"gaussian_rig{rigidity:.0f}_30mol_250nm.mp4"
                movie_path = _OUTPUT_DIR / movie_name
                ok = render_movie(frames_dir, movie_path, rigidity)
                if ok:
                    print(f"  Movie: {movie_path}")
                else:
                    print("  Movie: FAILED")
            else:
                print("  No frames directory found")

    print(f"\nDone! Movies saved to {_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
