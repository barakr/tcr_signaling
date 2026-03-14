"""Generate movies for gaussian+repulsion with pMHC, rigidity 0-100, 300s physical time.

GPU only. Movies saved to ~/Downloads/ks_screen2/.
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
_OUTPUT_DIR = Path.home() / "Downloads" / "ks_screen2"
# Use conda env with matplotlib + ffmpeg for rendering.
_RENDER_PYTHON = Path.home() / "miniconda3" / "envs" / "py314_bayesmm" / "bin" / "python"
_RENDER_ENV = {**os.environ, "PATH": str(_RENDER_PYTHON.parent) + os.pathsep + os.environ.get("PATH", "")}

# 11 linearly-spaced rigidity values from 0 to 100 kT/nm².
RIGIDITIES = list(range(0, 101, 10))  # [0, 10, 20, ..., 100]

GRID_SIZE = 128
TIME_SEC = 300       # 300 seconds physical time
DUMP_INTERVAL = 150  # 30000 steps / 150 = 200 frames
SEED = 42
N_TCR = 125
N_CD45 = 250
N_PMHC = 125
PMHC_RADIUS = 333


def run_sim(run_dir: Path, rigidity: float) -> float:
    """Run GPU simulation with gaussian binding + repulsion + pMHC. Return wall time."""
    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT", str(rigidity),
        "--seed", str(SEED),
        "--grid_size", str(GRID_SIZE),
        "--n_tcr", str(N_TCR),
        "--n_cd45", str(N_CD45),
        "--n_pmhc", str(N_PMHC),
        "--pmhc_radius", str(PMHC_RADIUS),
        "--binding_mode", "gaussian",
        "--mol_repulsion_eps", "2.0",
        "--mol_repulsion_rcut", "50.0",
        "--run-dir", str(run_dir),
        "--dump-frames",
        "--dump-interval", str(DUMP_INTERVAL),
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:300]}", file=sys.stderr)
        return -1.0
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=_RENDER_ENV)
    if result.returncode != 0:
        print(f"  Render failed: {result.stderr[:300]}", file=sys.stderr)
        return False
    return True


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(RIGIDITIES)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i, rig in enumerate(RIGIDITIES, 1):
            tag = f"gauss_repul_r{rig}"
            print(f"\n[{i}/{total}] rigidity={rig} kT/nm² (GPU, gauss+repul, 300s)")

            # Run simulation
            sim_dir = tmp_path / tag
            t_wall = run_sim(sim_dir, float(rig))
            if t_wall < 0:
                continue
            print(f"  Simulation: {t_wall:.1f}s wall time")

            # Render movie
            frames_dir = sim_dir / "frames"
            if frames_dir.exists():
                movie_name = f"gauss_repul_rig{rig}.mp4"
                movie_path = _OUTPUT_DIR / movie_name
                ok = render_movie(frames_dir, movie_path, float(rig))
                if ok:
                    print(f"  Movie: {movie_path}")
                else:
                    print(f"  Movie: FAILED")
            else:
                print(f"  No frames directory found")

    print(f"\nDone! Movies saved to {_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
