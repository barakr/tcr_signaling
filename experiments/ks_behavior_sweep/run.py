#!/usr/bin/env python3
"""KS behavior sweep: learn 8 segregation metrics as f(K, dt, t).

Each run simulates 10 seconds with --snapshot-interval 0.1, producing
100 metric snapshots per run.  The DOE sweeps K (rigidity) and dt
(timestep); time is tracked via the binary's snapshot output.

Uses bayesian_metamodeling DOE planning (plan_points) for reproducible
grid design; execution is direct binary invocation with the new
--snapshot-interval flag.

Usage
-----
    cd projects/tcr_signaling
    python experiments/ks_behavior_sweep/run.py              # full sweep
    python experiments/ks_behavior_sweep/run.py --dry-run     # show plan only
    python experiments/ks_behavior_sweep/run.py --n-seeds 1 --max-runs 1  # quick test

Output goes to ~/Downloads/metamodel_ks/ (results.csv, provenance.json, etc.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMODULE_DIR = SCRIPT_DIR.parent.parent          # projects/tcr_signaling
REPO_ROOT = SUBMODULE_DIR.parent.parent            # metamodeler_codex_scaffold_docs
BINARY = SUBMODULE_DIR / "models" / "kinetic_segregation" / "ks_gpu"
SPEC_PATH = SCRIPT_DIR / "spec.json"
OUT_DIR = Path.home() / "Downloads" / "metamodel_ks"

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
SIM_TIME = 10.0          # fixed simulation time (seconds)
SNAPSHOT_INTERVAL = 0.1  # metric snapshots every 0.1s → 100 snapshots per run

FIXED_ARGS = [
    "--time_sec", str(SIM_TIME),
    "--grid_size", "100",       # doubled from 50 (keeps dx=5nm)
    "--patch_size", "500",      # doubled from 250
    "--n_tcr", "30",
    "--n_cd45", "30",
    "--n_pmhc", "30",
    "--pmhc_radius", "21.0",    # physical radius unchanged
    "--pmhc_mode", "1",         # inner_circle
    "--step_mode", "brownian",
    "--binding_mode", "gaussian",
    "--monitor-binding", "3.0",
    "--monitor-interval", "50",
    "--snapshot-interval", str(SNAPSHOT_INTERVAL),
]

N_SEEDS_DEFAULT = 5
SEED_START = 42
PLOT_INTERVAL_SEC = 180
TIMEOUT_SEC = 3600  # 60 min max per run (10s sim at small dt + large grid)

# The 8 metrics extracted from each snapshot
METRIC_NAMES = [
    ("bound_fraction", "Bound Fraction"),
    ("depletion_width_nm", "Depletion Width (nm)"),
    ("overlap_coeff", "Overlap Coeff"),
    ("ks_statistic", "KS Statistic"),
    ("percentile_gap_nm", "Percentile Gap (nm)"),
    ("frontier_nn_gap_nm", "Frontier NN Gap (nm)"),
    ("bt_cd45_nn_p10_nm", "BT->CD45 NN P10 (nm)"),
    ("cd45_bt_nn_p10_nm", "CD45->BT NN P10 (nm)"),
]
METRIC_KEYS = [m[0] for m in METRIC_NAMES]

# Mapping from snapshot JSON keys to our CSV column names
SNAP_KEY_MAP = {
    "bound_fraction": "bound_fraction",
    "depletion_width_nm": "depletion_width_nm",
    "depletion_overlap_coeff": "overlap_coeff",
    "depletion_ks_statistic": "ks_statistic",
    "depletion_percentile_gap_nm": "percentile_gap_nm",
    "depletion_frontier_nn_gap_nm": "frontier_nn_gap_nm",
    "depletion_bound_tcr_cd45_nn_p10_nm": "bt_cd45_nn_p10_nm",
    "depletion_cd45_bound_tcr_nn_p10_nm": "cd45_bt_nn_p10_nm",
}

CSV_HEADER = [
    "rigidity_kT", "dt_sec", "seed", "time_sec",
    *METRIC_KEYS,
    "duration_sec", "status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_hash(repo_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _file_md5(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_snapshot_rows(snap: dict) -> dict:
    """Convert one snapshot JSON object to a flat metrics dict."""
    row = {}
    for snap_key, csv_key in SNAP_KEY_MAP.items():
        v = snap.get(snap_key)
        row[csv_key] = v if v is not None else ""
    return row


def load_existing_snapshots(run_dir: Path) -> list[dict] | None:
    """Load snapshot rows from an existing segregation.json, or None."""
    seg = run_dir / "segregation.json"
    if not seg.exists():
        return None
    try:
        data = json.loads(seg.read_text())
        snaps = data.get("snapshots", [])
        if not snaps:
            return None
        rows = []
        for s in snaps:
            metrics = extract_snapshot_rows(s)
            metrics["time_sec"] = s["time_sec"]
            rows.append(metrics)
        return rows
    except Exception:
        return None


def run_dir_for(base: Path, kappa: float, dt: float, seed: int) -> Path:
    dt_tag = f"dt{dt * 1e6:.2f}us"
    k_tag = f"k{kappa:.2f}"
    return base / dt_tag / k_tag / f"seed{seed}"


def run_single(kappa: float, dt: float, seed: int,
               run_dir: Path) -> tuple[list[dict] | None, float]:
    """Run one 10s simulation with snapshots.

    Returns (list_of_snapshot_row_dicts_or_None, wall_clock_duration_sec).
    Each dict has time_sec + 8 metric columns.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BINARY),
        "--rigidity_kT", str(kappa),
        "--dt", str(dt),
        "--seed", str(seed),
        *FIXED_ARGS,
        "--run-dir", str(run_dir),
    ]
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC,
        )
        duration = time.monotonic() - t0
        if result.returncode != 0:
            return None, duration
        data = json.loads(result.stdout.strip())
        (run_dir / "segregation.json").write_text(json.dumps(data, indent=2))
        snaps = data.get("snapshots", [])
        if not snaps:
            return None, duration
        rows = []
        for s in snaps:
            metrics = extract_snapshot_rows(s)
            metrics["time_sec"] = s["time_sec"]
            rows.append(metrics)
        return rows, duration
    except Exception:
        return None, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def write_provenance(spec_payload: dict, n_seeds: int, design_points: list):
    """Write provenance.json to OUT_DIR."""
    try:
        pkg_version = importlib.metadata.version("bayesian-metamodeling")
    except importlib.metadata.PackageNotFoundError:
        pkg_version = "not-installed"

    prov = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": {
            "package": "bayesian-metamodeling",
            "version": pkg_version,
            "repo_commit": _git_hash(REPO_ROOT),
            "submodule_commit": _git_hash(SUBMODULE_DIR),
        },
        "binary": {
            "path": str(BINARY),
            "md5": _file_md5(BINARY),
        },
        "platform": {
            "system": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "spec": spec_payload,
        "fixed_args": FIXED_ARGS,
        "sim_time_sec": SIM_TIME,
        "snapshot_interval_sec": SNAPSHOT_INTERVAL,
        "n_seeds": n_seeds,
        "seed_start": SEED_START,
        "n_design_points": len(design_points),
        "total_runs": len(design_points) * n_seeds,
        "snapshots_per_run": int(SIM_TIME / SNAPSHOT_INTERVAL),
        "total_csv_rows": len(design_points) * n_seeds * int(SIM_TIME / SNAPSHOT_INTERVAL),
        "script": str(SCRIPT_DIR / "run.py"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


# ---------------------------------------------------------------------------
# Progress plotting
# ---------------------------------------------------------------------------

def make_progress_plot(results_csv: Path, output: Path,
                       done: int, total: int, elapsed: float):
    """2x4 panel: each metric, x=time, colored by dt, lines=mean over seeds at K=1 and K=40."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = []
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return

    dt_vals = sorted(set(float(r["dt_sec"]) for r in rows))
    k_vals = sorted(set(float(r["rigidity_kT"]) for r in rows))
    # Show dynamics for min and max K
    k_show = [k_vals[0], k_vals[-1]] if len(k_vals) >= 2 else k_vals[:1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dt_vals)))
    dt_color = {d: c for d, c in zip(dt_vals, colors)}

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle(
        f"KS Behavior Sweep ({done}/{total} runs, {elapsed / 60:.1f} min)\n"
        f"patch=500nm, grid=100, dx=5nm, {SNAPSHOT_INTERVAL}s snapshots, "
        f"showing K={', '.join(f'{k:.1f}' for k in k_show)}",
        fontsize=13,
    )

    for idx, (mkey, mname) in enumerate(METRIC_NAMES):
        ax = axes[idx // 4, idx % 4]
        for ki, k_val in enumerate(k_show):
            ls = "-" if ki == 0 else "--"
            for dt_val in dt_vals:
                filt = [r for r in rows
                        if abs(float(r["rigidity_kT"]) - k_val) < 0.01
                        and float(r["dt_sec"]) == dt_val
                        and r["status"] == "ok" and r[mkey] != ""]
                if not filt:
                    continue
                # Group by time, compute mean over seeds
                t_groups: dict[float, list[float]] = {}
                for r in filt:
                    t = float(r["time_sec"])
                    v = float(r[mkey])
                    t_groups.setdefault(t, []).append(v)
                ts = sorted(t_groups.keys())
                means = [np.mean(t_groups[t]) for t in ts]
                label = f"K={k_val:.0f} dt={dt_val * 1e6:.1f}us" if idx == 0 else None
                ax.plot(ts, means, ls, color=dt_color[dt_val], alpha=0.7,
                        linewidth=1.2, label=label)
        ax.set_xlabel("time (s)")
        ax.set_title(mname, fontsize=10)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KS behavior sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without running")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT,
                        help=f"Seeds per DOE point (default {N_SEEDS_DEFAULT})")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Stop after this many new runs (0=unlimited)")
    args = parser.parse_args()

    # --- Load spec and plan DOE via bayesian_metamodeling ---
    spec_payload = json.loads(SPEC_PATH.read_text())
    try:
        from bayesian_metamodeling.designs import plan_points
        from bayesian_metamodeling.spec import load_and_validate_modelspec

        spec = load_and_validate_modelspec(spec_payload)
        design_points = plan_points(spec)
        print(f"DOE via bayesian_metamodeling: {len(design_points)} design points "
              f"(strategy={spec.design.strategy})")
    except ImportError:
        print("Warning: bayesian_metamodeling not importable; generating grid manually")
        grid = spec_payload["design"]["grid"]
        from itertools import product
        keys = list(grid.keys())
        design_points = [
            dict(zip(keys, combo))
            for combo in product(*(grid[k] for k in keys))
        ]
        print(f"DOE (manual grid): {len(design_points)} design points")

    n_seeds = args.n_seeds
    snaps_per_run = int(SIM_TIME / SNAPSHOT_INTERVAL)
    total_runs = len(design_points) * n_seeds
    total_rows = total_runs * snaps_per_run
    print(f"Seeds per point: {n_seeds} (start={SEED_START})")
    print(f"Total runs: {total_runs} ({snaps_per_run} snapshots each = {total_rows} CSV rows)")
    print(f"Sim: {SIM_TIME}s, snapshot every {SNAPSHOT_INTERVAL}s")
    print(f"Fixed args: {' '.join(FIXED_ARGS)}")
    print(f"Binary: {BINARY} (exists={BINARY.exists()})")
    print(f"Output: {OUT_DIR}")

    # --- Build run schedule ---
    schedule = []
    for pt in design_points:
        kappa = pt["rigidity_kT"]
        dt_sec = pt["dt_sec"]
        for i in range(n_seeds):
            seed = SEED_START + i
            rd = run_dir_for(OUT_DIR / "runs", kappa, dt_sec, seed)
            schedule.append((kappa, dt_sec, seed, rd))

    if args.dry_run:
        print(f"\n--- DRY RUN: {len(schedule)} runs planned ---")
        for kappa, dt_sec, seed, rd in schedule[:10]:
            print(f"  K={kappa:6.2f}  dt={dt_sec:.2e}  seed={seed}  "
                  f"-> {snaps_per_run} snapshots  -> {rd}")
        if len(schedule) > 10:
            print(f"  ... and {len(schedule) - 10} more")
        return

    # --- Write provenance ---
    write_provenance(spec_payload, n_seeds, design_points)
    print(f"Provenance written: {OUT_DIR / 'provenance.json'}")

    # --- Prepare CSV ---
    results_csv = OUT_DIR / "results.csv"
    log_path = OUT_DIR / "run_log.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results to allow resume (keyed on run, not individual snapshots)
    existing_run_keys: set[tuple] = set()
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (float(row["rigidity_kT"]),
                       float(row["dt_sec"]), int(row["seed"]))
                existing_run_keys.add(key)
        n_existing_rows = sum(1 for _ in open(results_csv)) - 1
        print(f"Loaded {len(existing_run_keys)} existing runs "
              f"({n_existing_rows} rows) from CSV")

    # Write header if file is new
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    csv_file = open(results_csv, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    if write_header:
        csv_writer.writeheader()
        csv_file.flush()

    log_file = open(log_path, "a")

    # --- Run loop ---
    t0 = time.time()
    last_plot = t0
    new_runs = 0
    skipped = 0

    for idx, (kappa, dt_sec, seed, rd) in enumerate(schedule):
        key = (kappa, dt_sec, seed)

        # Resume: skip if already in CSV
        if key in existing_run_keys:
            skipped += 1
            continue

        # Also check disk for existing segregation.json with snapshots
        existing_snaps = load_existing_snapshots(rd)
        if existing_snaps is not None:
            for snap_row in existing_snaps:
                row = {
                    "rigidity_kT": kappa, "dt_sec": dt_sec, "seed": seed,
                    **snap_row,
                    "duration_sec": 0, "status": "ok",
                }
                csv_writer.writerow(row)
            csv_file.flush()
            existing_run_keys.add(key)
            skipped += 1
            continue

        # Run simulation
        snap_rows, duration = run_single(kappa, dt_sec, seed, rd)
        new_runs += 1
        status = "ok" if snap_rows else "failed"

        if snap_rows:
            for snap_row in snap_rows:
                row = {
                    "rigidity_kT": kappa, "dt_sec": dt_sec, "seed": seed,
                    **snap_row,
                    "duration_sec": f"{duration:.1f}", "status": status,
                }
                csv_writer.writerow(row)
        else:
            # Write a single failed row
            row = {
                "rigidity_kT": kappa, "dt_sec": dt_sec, "seed": seed,
                "time_sec": SIM_TIME,
                **{k: "" for k in METRIC_KEYS},
                "duration_sec": f"{duration:.1f}", "status": "failed",
            }
            csv_writer.writerow(row)
        csv_file.flush()
        existing_run_keys.add(key)

        # Log detail
        n_snaps = len(snap_rows) if snap_rows else 0
        log_entry = {
            "rigidity_kT": kappa, "dt_sec": dt_sec, "seed": seed,
            "status": status, "n_snapshots": n_snaps,
            "duration_sec": round(duration, 1), "run_dir": str(rd),
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # Progress
        done_runs = len(existing_run_keys)
        if new_runs % 5 == 0 or new_runs == 1:
            elapsed = time.time() - t0
            rate = new_runs / elapsed if elapsed > 0 else 0
            remaining = total_runs - done_runs
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{done_runs}/{total_runs}] {new_runs} new, "
                  f"{elapsed:.0f}s elapsed, ETA {eta / 60:.1f}min | "
                  f"K={kappa:.1f} dt={dt_sec:.2e} s={seed} "
                  f"-> {status} ({n_snaps} snaps, {duration:.1f}s)")

        # Progress plot
        now = time.time()
        if now - last_plot >= PLOT_INTERVAL_SEC:
            make_progress_plot(results_csv, OUT_DIR / "progress_latest.png",
                               done_runs, total_runs, now - t0)
            print(f"  Plot saved ({done_runs}/{total_runs} runs complete)")
            last_plot = now

        if args.max_runs > 0 and new_runs >= args.max_runs:
            print(f"Stopping after {args.max_runs} new runs (--max-runs)")
            break

    csv_file.close()
    log_file.close()

    # --- Final plot ---
    elapsed = time.time() - t0
    done_runs = len(existing_run_keys)
    make_progress_plot(results_csv, OUT_DIR / "progress_latest.png",
                       done_runs, total_runs, elapsed)
    make_progress_plot(results_csv, OUT_DIR / "behavior_sweep_final.png",
                       done_runs, total_runs, elapsed)

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"Completed: {done_runs}/{total_runs} runs "
          f"({new_runs} new, {skipped} skipped) in {elapsed:.0f}s")
    print(f"Results: {results_csv}")
    print(f"Provenance: {OUT_DIR / 'provenance.json'}")
    print(f"Log: {log_path}")
    print(f"Plot: {OUT_DIR / 'behavior_sweep_final.png'}")


if __name__ == "__main__":
    main()
