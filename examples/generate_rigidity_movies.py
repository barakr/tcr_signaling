#!/usr/bin/env python3
"""Generate MP4 movies of 5-second KS simulations at different membrane rigidities.

Uses the GPU-accelerated kinetic segregation binary to dump binary frames,
then renders each to an MP4 via render_movie.py.

Usage::

    python projects/tcr_signaling/examples/generate_rigidity_movies.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
RIGIDITIES = [5, 10, 20, 50, 100]  # kT/nm²
TIME_SEC = 100.0
GRID_SIZE = 128
SEED = 42
DUMP_INTERVAL = 50
FPS = 15
OUTPUT_DIR = Path.home() / "Downloads"

# Paths relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent
_SUBMODULE_ROOT = _SCRIPT_DIR.parent
_GPU_DIR = _SUBMODULE_ROOT / "models" / "kinetic_segregation"
_BINARY = _GPU_DIR / "ks_gpu"
_RENDER_SCRIPT = _GPU_DIR / "render_movie.py"
_RENDER_PYTHON = Path.home() / "miniconda3" / "envs" / "py314_bayesmm" / "bin" / "python"


def run_one(rigidity: float, tmp_root: Path) -> Path:
    """Run GPU simulation with frame dumps, then render to MP4."""
    run_dir = tmp_root / f"ks_K{rigidity:g}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"ks_rigidity_{rigidity:g}.mp4"

    # Step 1: run GPU binary directly with frame dumps
    print(f"\n{'='*60}")
    print(f"Running GPU simulation: rigidity={rigidity} kT/nm², time={TIME_SEC}s, grid={GRID_SIZE}")
    print(f"{'='*60}")
    gpu_cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT_nm2", str(rigidity),
        "--grid_size", str(GRID_SIZE),
        "--seed", str(SEED),
        "--run-dir", str(run_dir),
        "--dump-frames",
        "--dump-interval", str(DUMP_INTERVAL),
    ]
    result = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"ERROR: GPU simulation failed for rigidity={rigidity}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"GPU simulation failed (rigidity={rigidity})")
    print(f"Simulation complete. Output: {result.stdout.strip()[:120]}")

    # Step 2: render frames to MP4
    frames_dir = run_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    print(f"Rendering movie to {output_path} ...")
    render_python = str(_RENDER_PYTHON) if _RENDER_PYTHON.exists() else sys.executable
    render_cmd = [
        render_python, str(_RENDER_SCRIPT),
        str(frames_dir),
        "-o", str(output_path),
        "--fps", str(FPS),
    ]
    result = subprocess.run(render_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"ERROR: Rendering failed for rigidity={rigidity}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Rendering failed (rigidity={rigidity})")
    print(result.stdout.strip())

    return output_path


def main() -> int:
    if not _BINARY.exists():
        print(f"GPU binary not found at {_BINARY}", file=sys.stderr)
        print(f"Build it first: cd {_GPU_DIR} && make", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating KS movies for rigidities: {RIGIDITIES} kT/nm²")
    print(f"Output directory: {OUTPUT_DIR}")

    outputs: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="ks_movies_") as tmp_root:
        for rigidity in RIGIDITIES:
            out = run_one(rigidity, Path(tmp_root))
            outputs.append(out)

    print(f"\n{'='*60}")
    print("All movies generated:")
    for p in outputs:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}  ({size_mb:.1f} MB)")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
