"""Generate movies for 3×3 screen: TCR/pMHC ∈ {5,10,20} × CD45 ∈ {5,10,20}.

Small membrane patch (250 nm), grid 50×50, gaussian binding + pMHC influence.
GPU only.  Movies saved to ~/Downloads/ks_small_patch_screen/.

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
_OUTPUT_DIR = Path.home() / "Downloads" / "ks_small_patch_screen"
# Use conda env with matplotlib + ffmpeg for rendering.
_RENDER_PYTHON = Path.home() / "miniconda3" / "envs" / "py314_bayesmm" / "bin" / "python"
_RENDER_ENV = {
    **os.environ,
    "PATH": str(_RENDER_PYTHON.parent) + os.pathsep + os.environ.get("PATH", ""),
}

# Screen parameters
GRID_SIZE = 50
TIME_SEC = 200
RIGIDITY = 20.0        # kT/nm² (paper default)
DUMP_INTERVAL = 100    # 20000 steps / 100 = 200 frames → ~13s movie at 15fps
SEED = 42

# Molecule counts to screen
N_TCR_PMHC_VALUES = [5, 10, 20]
N_CD45_VALUES = [5, 10, 20]

# Patch size (runtime, no rebuild needed)
PATCH_SIZE = 250.0

# pMHC placement: inner_circle, radius = patch/3 ≈ 83 nm
PMHC_RADIUS = 83

# Gaussian binding parameters
SIGMA_R = 2.0  # lateral pMHC influence range (nm)

# Excluded volume parameters (standard)
MOL_REPULSION_EPS = 2.0
MOL_REPULSION_RCUT = 10.0


def run_sim(run_dir: Path, n_tcr: int, n_cd45: int) -> float:
    """Run GPU simulation with gaussian binding + repulsion. Return wall time."""
    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT", str(RIGIDITY),
        "--seed", str(SEED),
        "--grid_size", str(GRID_SIZE),
        "--n_tcr", str(n_tcr),
        "--n_cd45", str(n_cd45),
        "--n_pmhc", str(n_tcr),  # n_pmhc matches n_tcr
        "--pmhc_radius", str(PMHC_RADIUS),
        "--binding_mode", "gaussian",
        "--mol_repulsion_eps", str(MOL_REPULSION_EPS),
        "--mol_repulsion_rcut", str(MOL_REPULSION_RCUT),
        "--patch_size", str(PATCH_SIZE),
        "--sigma_r", str(SIGMA_R),
        "--run-dir", str(run_dir),
        "--dump-frames",
        "--dump-interval", str(DUMP_INTERVAL),
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}):", file=sys.stderr)
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
        print(f"  stdout: {result.stdout[:500]}", file=sys.stderr)
        return -1.0
    return elapsed


def render_movie(frames_dir: Path, output: Path):
    """Render movie from frames with pMHC markers shown."""
    cmd = [
        str(_RENDER_PYTHON), str(_RENDER),
        str(frames_dir),
        "-o", str(output),
        "--fps", "15",
        "--dpi", "120",
        "--rigidity", str(RIGIDITY),
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
    conditions = [
        (nt, nc)
        for nt in N_TCR_PMHC_VALUES
        for nc in N_CD45_VALUES
    ]
    total = len(conditions)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i, (n_tcr, n_cd45) in enumerate(conditions, 1):
            tag = f"tcr{n_tcr}_cd45{n_cd45}"
            print(f"\n[{i}/{total}] n_tcr=n_pmhc={n_tcr}, n_cd45={n_cd45} "
                  f"(grid={GRID_SIZE}, {TIME_SEC}s, rig={RIGIDITY})")

            # Run simulation
            sim_dir = tmp_path / tag
            t_wall = run_sim(sim_dir, n_tcr, n_cd45)
            if t_wall < 0:
                continue
            print(f"  Simulation: {t_wall:.1f}s wall time")

            # Render movie
            frames_dir = sim_dir / "frames"
            if frames_dir.exists():
                movie_name = f"{tag}_250nm.mp4"
                movie_path = _OUTPUT_DIR / movie_name
                ok = render_movie(frames_dir, movie_path)
                if ok:
                    print(f"  Movie: {movie_path}")
                else:
                    print("  Movie: FAILED")
            else:
                print("  No frames directory found")

    print(f"\nDone! Movies saved to {_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
