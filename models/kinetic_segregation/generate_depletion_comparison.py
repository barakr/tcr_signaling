"""Generate comparison movies at 4 rigidities to assess depletion metrics.

GPU, gaussian binding + repulsion + pMHC, 300s physical time.
Movies saved to ~/Downloads/ks_depletion_metrics/.
Also writes a summary JSON with all metrics for each rigidity.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_BINARY = _PKG_DIR / "ks_gpu"
_RENDER = _PKG_DIR / "render_movie.py"
_OUTPUT_DIR = Path.home() / "Downloads" / "ks_depletion_metrics"
_RENDER_PYTHON = Path.home() / "miniconda3" / "envs" / "py314_bayesmm" / "bin" / "python"
_RENDER_ENV = {
    **os.environ,
    "PATH": str(_RENDER_PYTHON.parent) + os.pathsep + os.environ.get("PATH", ""),
}

RIGIDITIES = [1, 10, 30, 100]

GRID_SIZE = 128
TIME_SEC = 300
DUMP_INTERVAL = 150
SEED = 42
N_TCR = 125
N_CD45 = 250
N_PMHC = 125
PMHC_RADIUS = 333


def run_sim(run_dir: Path, rigidity: float) -> float:
    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT_nm2", str(rigidity),
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
    cmd = [
        str(_RENDER_PYTHON), str(_RENDER),
        str(frames_dir),
        "-o", str(output),
        "--fps", "15",
        "--dpi", "120",
        "--rigidity", str(rigidity),
        "--show-pmhc",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, env=_RENDER_ENV
    )
    if result.returncode != 0:
        print(f"  Render failed: {result.stderr[:300]}", file=sys.stderr)
        return False
    return True


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(RIGIDITIES)
    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i, rig in enumerate(RIGIDITIES, 1):
            print(f"\n[{i}/{total}] rigidity={rig} kT/nm²")

            sim_dir = tmp_path / f"rig{rig}"
            t_wall = run_sim(sim_dir, float(rig))
            if t_wall < 0:
                continue
            print(f"  Simulation: {t_wall:.1f}s")

            # Read metrics from JSON output.
            seg_json = sim_dir / "out" / "segregation.json"
            if seg_json.exists():
                data = json.loads(seg_json.read_text())
                diag = data["diagnostics"]
                results[f"rigidity_{rig}"] = {
                    "median_diff": data["depletion_width_nm"],
                    "pct_gap": diag.get("depletion_percentile_gap_nm"),
                    "overlap": diag.get("depletion_overlap_coeff"),
                    "ks_stat": diag.get("depletion_ks_statistic"),
                    "frontier_nn": diag.get("depletion_frontier_nn_gap_nm"),
                    "cross_nn": diag.get("depletion_cross_nn_median_nm"),
                }
                print(f"  Metrics: median_diff={data['depletion_width_nm']:.0f} "
                      f"pct_gap={diag.get('depletion_percentile_gap_nm', 0):.0f} "
                      f"overlap={diag.get('depletion_overlap_coeff', 0):.2f} "
                      f"frontier_nn={diag.get('depletion_frontier_nn_gap_nm', 0):.0f} "
                      f"cross_nn={diag.get('depletion_cross_nn_median_nm', 0):.0f}")

            # Render movie.
            frames_dir = sim_dir / "frames"
            if frames_dir.exists():
                movie_path = _OUTPUT_DIR / f"rig{rig}_gauss_repul.mp4"
                ok = render_movie(frames_dir, movie_path, float(rig))
                print(f"  Movie: {'OK' if ok else 'FAILED'} → {movie_path.name}")

    # Summary.
    summary_path = _OUTPUT_DIR / "metrics_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"Movies saved to {_OUTPUT_DIR}/")

    # Print table.
    print(f"\n{'Rig':>6} {'median':>8} {'pct_gap':>8} {'overlap':>8} "
          f"{'KS':>6} {'front_nn':>9} {'cross_nn':>9}")
    print("-" * 60)
    for rig in RIGIDITIES:
        key = f"rigidity_{rig}"
        if key in results:
            r = results[key]
            print(f"{rig:>6} {r['median_diff']:>8.0f} {r['pct_gap']:>8.0f} "
                  f"{r['overlap']:>8.2f} {r['ks_stat']:>6.2f} "
                  f"{r['frontier_nn']:>9.0f} {r['cross_nn']:>9.0f}")


if __name__ == "__main__":
    main()
