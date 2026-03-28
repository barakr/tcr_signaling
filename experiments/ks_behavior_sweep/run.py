#!/usr/bin/env python3
"""KS behavior sweep: learn 8 segregation metrics as f(K, dt).

Runs 10-second simulations across 10 log-spaced rigidity (K) values and
4 timestep (dt) values.  time_sec is fixed at 10s and tracked in the output
for provenance but is NOT a swept parameter.

Uses bayesian_metamodeling DOE planning (plan_points) for reproducible
grid design; execution is direct binary invocation (same pattern as
sweep_comprehensive.py) to handle multi-seed runs and all 8 metric
extractions.

Usage
-----
    cd projects/tcr_signaling
    python experiments/ks_behavior_sweep/run.py              # full sweep
    python experiments/ks_behavior_sweep/run.py --dry-run     # show plan only
    python experiments/ks_behavior_sweep/run.py --n-seeds 1 --max-runs 5  # quick test

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
# Fixed physics parameters (same as sweep_comprehensive.py, doubled patch+grid)
# ---------------------------------------------------------------------------
SIM_TIME = 10.0  # fixed simulation time (not swept)

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
]

N_SEEDS_DEFAULT = 5
SEED_START = 42
PLOT_INTERVAL_SEC = 180
TIMEOUT_SEC = 1200  # 20 min max per run

# The 8 metrics we extract from each run
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


def extract_metrics(data: dict) -> dict:
    """Extract 8 metrics from a segregation.json dict."""
    ts = data.get("binding_timeseries", [])
    half = len(ts) // 2
    diag = data.get("diagnostics", {})
    return {
        "bound_fraction": float(np.mean(ts[half:])) if ts else None,
        "depletion_width_nm": data.get("depletion_width_nm"),
        "overlap_coeff": diag.get("depletion_overlap_coeff"),
        "ks_statistic": diag.get("depletion_ks_statistic"),
        "percentile_gap_nm": diag.get("depletion_percentile_gap_nm"),
        "frontier_nn_gap_nm": diag.get("depletion_frontier_nn_gap_nm"),
        "bt_cd45_nn_p10_nm": diag.get("depletion_bound_tcr_cd45_nn_p10_nm"),
        "cd45_bt_nn_p10_nm": diag.get("depletion_cd45_bound_tcr_nn_p10_nm"),
    }


def load_existing(run_dir: Path) -> dict | None:
    seg = run_dir / "segregation.json"
    if not seg.exists():
        return None
    try:
        return extract_metrics(json.loads(seg.read_text()))
    except Exception:
        return None


def run_dir_for(base: Path, kappa: float, dt: float, seed: int) -> Path:
    dt_tag = f"dt{dt * 1e6:.2f}us"
    k_tag = f"k{kappa:.2f}"
    return base / dt_tag / k_tag / f"seed{seed}"


def run_single(kappa: float, dt: float, seed: int,
               run_dir: Path) -> tuple[dict | None, float]:
    """Run one simulation. Returns (metrics_dict_or_None, duration_sec)."""
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
        return extract_metrics(data), duration
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
        "n_seeds": n_seeds,
        "seed_start": SEED_START,
        "n_design_points": len(design_points),
        "total_runs": len(design_points) * n_seeds,
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
    """2x4 panel: each metric, x=K (log), colored by dt, lines=mean over seeds."""
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
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dt_vals)))
    dt_color = {d: c for d, c in zip(dt_vals, colors)}

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle(
        f"KS Behavior Sweep ({done}/{total} runs, {elapsed / 60:.1f} min)\n"
        f"patch=500nm, grid=100, dx=5nm, brownian/gaussian",
        fontsize=13,
    )

    for idx, (mkey, mname) in enumerate(METRIC_NAMES):
        ax = axes[idx // 4, idx % 4]
        for dt_val in dt_vals:
            dt_rows = [r for r in rows if float(r["dt_sec"]) == dt_val
                       and r["status"] == "ok" and r[mkey] != ""]
            if not dt_rows:
                continue
            # Group by K, compute mean
            k_groups: dict[float, list[float]] = {}
            for r in dt_rows:
                k = float(r["rigidity_kT"])
                v = float(r[mkey])
                k_groups.setdefault(k, []).append(v)
            ks = sorted(k_groups.keys())
            means = [np.mean(k_groups[k]) for k in ks]
            stds = [np.std(k_groups[k], ddof=1) if len(k_groups[k]) > 1 else 0
                    for k in ks]
            label = f"dt={dt_val * 1e6:.1f}us"
            ax.errorbar(ks, means, yerr=stds, fmt="o-", markersize=4,
                        color=dt_color[dt_val], label=label, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("K (kT)")
        ax.set_title(mname, fontsize=10)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

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
        # Fallback: generate grid manually if package not installed
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
    total_runs = len(design_points) * n_seeds
    print(f"Seeds per point: {n_seeds} (start={SEED_START})")
    print(f"Total runs: {total_runs}")
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
            print(f"  K={kappa:6.2f}  t={SIM_TIME:.0f}s(fixed)  dt={dt_sec:.2e}  "
                  f"seed={seed}  -> {rd}")
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

    # Load existing results to allow resume
    existing_keys: set[tuple] = set()
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (float(row["rigidity_kT"]),
                       float(row["dt_sec"]), int(row["seed"]))
                existing_keys.add(key)
        print(f"Loaded {len(existing_keys)} existing results from CSV")

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
        if key in existing_keys:
            skipped += 1
            continue

        # Also skip if segregation.json exists on disk
        existing = load_existing(rd)
        if existing is not None:
            row = {
                "rigidity_kT": kappa, "dt_sec": dt_sec,
                "seed": seed, "time_sec": SIM_TIME,
                **{k: existing.get(k, "") for k in METRIC_KEYS},
                "duration_sec": 0, "status": "ok",
            }
            csv_writer.writerow(row)
            csv_file.flush()
            existing_keys.add(key)
            skipped += 1
            continue

        # Run simulation
        metrics, duration = run_single(kappa, dt_sec, seed, rd)
        new_runs += 1
        status = "ok" if metrics else "failed"

        row = {
            "rigidity_kT": kappa, "dt_sec": dt_sec,
            "seed": seed, "time_sec": SIM_TIME,
            **{k: (metrics.get(k, "") if metrics else "") for k in METRIC_KEYS},
            "duration_sec": f"{duration:.1f}", "status": status,
        }
        csv_writer.writerow(row)
        csv_file.flush()
        existing_keys.add(key)

        # Log detail
        log_entry = {
            "rigidity_kT": kappa, "dt_sec": dt_sec, "seed": seed,
            "time_sec": SIM_TIME, "status": status,
            "duration_sec": round(duration, 1), "run_dir": str(rd),
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # Progress
        done_total = len(existing_keys)
        if new_runs % 10 == 0 or new_runs == 1:
            elapsed = time.time() - t0
            rate = new_runs / elapsed if elapsed > 0 else 0
            remaining = total_runs - done_total
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{done_total}/{total_runs}] {new_runs} new, "
                  f"{elapsed:.0f}s elapsed, ETA {eta / 60:.1f}min | "
                  f"K={kappa:.1f} dt={dt_sec:.2e} s={seed} "
                  f"-> {status} ({duration:.1f}s)")

        # Progress plot
        now = time.time()
        if now - last_plot >= PLOT_INTERVAL_SEC:
            make_progress_plot(results_csv, OUT_DIR / "progress_latest.png",
                               done_total, total_runs, now - t0)
            print(f"  Plot saved ({done_total}/{total_runs} complete)")
            last_plot = now

        if args.max_runs > 0 and new_runs >= args.max_runs:
            print(f"Stopping after {args.max_runs} new runs (--max-runs)")
            break

    csv_file.close()
    log_file.close()

    # --- Final plot ---
    elapsed = time.time() - t0
    done_total = len(existing_keys)
    make_progress_plot(results_csv, OUT_DIR / "progress_latest.png",
                       done_total, total_runs, elapsed)
    make_progress_plot(results_csv, OUT_DIR / "behavior_sweep_final.png",
                       done_total, total_runs, elapsed)

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"Completed: {done_total}/{total_runs} runs "
          f"({new_runs} new, {skipped} skipped) in {elapsed:.0f}s")
    print(f"Results: {results_csv}")
    print(f"Provenance: {OUT_DIR / 'provenance.json'}")
    print(f"Log: {log_path}")
    print(f"Plot: {OUT_DIR / 'behavior_sweep_final.png'}")


if __name__ == "__main__":
    main()
