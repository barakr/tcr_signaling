"""Generate movies for all 4 config combos × 10 rigidity values, grid=128.

Times GPU vs CPU for each run and produces a summary table.
Movies saved to ~/Downloads/ks_y/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent
_BINARY = _PKG_DIR / "ks_gpu"
_RENDER = _PKG_DIR / "render_movie.py"
_OUTPUT_DIR = Path.home() / "Downloads" / "ks_y"
# Ensure the running Python's bin dir is on PATH so ffmpeg is found.
_RENDER_ENV = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}

# 10 log-spaced rigidity values from 1 to 100 kT/nm².
RIGIDITIES = np.logspace(0, 2, 10)

# 4 configuration combos.
CONFIGS = {
    "gauss": [],
    "forced": [
        "--binding_mode", "forced", "--n_pmhc", "125", "--pmhc_radius", "333",
    ],
    "gauss_repul": [
        "--mol_repulsion_eps", "2.0", "--mol_repulsion_rcut", "50.0",
    ],
    "forced_repul": [
        "--mol_repulsion_eps", "2.0", "--mol_repulsion_rcut", "50.0",
        "--binding_mode", "forced", "--n_pmhc", "125", "--pmhc_radius", "333",
    ],
}

GRID_SIZE = 128
N_STEPS = 10000
DUMP_INTERVAL = 50
TIME_SEC = 200
SEED = 42
N_TCR = 125
N_CD45 = 250


def run_sim(run_dir: Path, rigidity: float, config_name: str, use_gpu: bool) -> float:
    """Run simulation and return wall-clock time in seconds."""
    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT_nm2", str(rigidity),
        "--seed", str(SEED),
        "--n_steps", str(N_STEPS),
        "--grid_size", str(GRID_SIZE),
        "--n_tcr", str(N_TCR),
        "--n_cd45", str(N_CD45),
        "--run-dir", str(run_dir),
        "--dump-frames",
        "--dump-interval", str(DUMP_INTERVAL),
    ] + CONFIGS[config_name]
    if not use_gpu:
        cmd.append("--no-gpu")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}", file=sys.stderr)
        return -1.0
    return elapsed


def render_movie(frames_dir: Path, output: Path, rigidity: float):
    """Render movie from frames."""
    cmd = [
        sys.executable, str(_RENDER),
        str(frames_dir),
        "-o", str(output),
        "--fps", "15",
        "--dpi", "120",
        "--rigidity", str(rigidity),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=_RENDER_ENV)
    if result.returncode != 0:
        print(f"  Render failed: {result.stderr[:200]}", file=sys.stderr)
        return False
    return True


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(CONFIGS) * len(RIGIDITIES)
    done = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for cfg_name in CONFIGS:
            for rig in RIGIDITIES:
                done += 1
                tag = f"{cfg_name}_r{rig:.1f}"
                print(f"\n[{done}/{total}] {tag} (grid={GRID_SIZE}, steps={N_STEPS})")

                # --- GPU run (for movie + timing) ---
                gpu_dir = tmp_path / f"{tag}_gpu"
                t_gpu = run_sim(gpu_dir, rig, cfg_name, use_gpu=True)
                print(f"  GPU: {t_gpu:.2f}s")

                # --- CPU run (timing only) ---
                cpu_dir = tmp_path / f"{tag}_cpu"
                t_cpu = run_sim(cpu_dir, rig, cfg_name, use_gpu=False)
                print(f"  CPU: {t_cpu:.2f}s")

                speedup = t_cpu / t_gpu if t_gpu > 0 else 0
                print(f"  Speedup: {speedup:.1f}x")

                results.append({
                    "config": cfg_name,
                    "rigidity": round(rig, 1),
                    "gpu_sec": round(t_gpu, 2),
                    "cpu_sec": round(t_cpu, 2),
                    "speedup": round(speedup, 1),
                })

                # --- Render movie from GPU run ---
                frames_dir = gpu_dir / "frames"
                if frames_dir.exists():
                    movie_name = f"{cfg_name}_rig{rig:.1f}.mp4"
                    movie_path = _OUTPUT_DIR / movie_name
                    ok = render_movie(frames_dir, movie_path, rig)
                    if ok:
                        print(f"  Movie: {movie_path.name}")
                    else:
                        print(f"  Movie: FAILED")
                else:
                    print(f"  No frames directory found")

    # --- Summary table ---
    print("\n\n" + "=" * 85)
    print("TIMING SUMMARY — grid=128, steps=10000, all 4 configs × 10 rigidities")
    print("=" * 85)
    print(f"{'Config':<14} {'Rigidity':>8} {'CPU (s)':>9} {'GPU (s)':>9} {'Speedup':>8}")
    print("-" * 85)
    for r in results:
        sp = f"{r['speedup']:.1f}x" if r['speedup'] > 0 else "N/A"
        print(f"{r['config']:<14} {r['rigidity']:>8.1f} {r['cpu_sec']:>9.2f} "
              f"{r['gpu_sec']:>9.2f} {sp:>8}")

    # Per-config averages.
    print("-" * 85)
    for cfg in CONFIGS:
        rows = [r for r in results if r["config"] == cfg]
        avg_cpu = np.mean([r["cpu_sec"] for r in rows])
        avg_gpu = np.mean([r["gpu_sec"] for r in rows])
        avg_sp = avg_cpu / avg_gpu if avg_gpu > 0 else 0
        print(f"{cfg + ' (avg)':<14} {'':>8} {avg_cpu:>9.2f} {avg_gpu:>9.2f} "
              f"{avg_sp:>7.1f}x")

    # Save JSON.
    out_json = _OUTPUT_DIR / "timing_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_json}")
    print(f"Movies saved to {_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
