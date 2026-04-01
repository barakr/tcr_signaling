#!/usr/bin/env python3
"""Fit surrogate models on the completed KS behavior sweep data.

Reads ~/Downloads/metamodel_ks/results.csv (20,000 rows: 3 inputs × 8 metrics),
averages over seeds, and fits one surrogate per metric via the
bayesian_metamodeling framework.

Usage
-----
    cd projects/tcr_signaling

    # SBI (recommended — flexible neural density estimator):
    conda activate py312_bayesmm_sbi
    python experiments/ks_behavior_sweep/fit_surrogates.py --backend sbi_npe

    # PyMC (fast Bayesian linear regression):
    conda activate py312_bayesmm_pymc
    python experiments/ks_behavior_sweep/fit_surrogates.py --backend pymc_gp

    # Options:
    #   --no-average      Use all 20k rows (don't average over seeds)
    #   --results-csv     Override path to results CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from bayesian_metamodeling.surrogates.backends import (
    fit_backend_model,
    save_backend_payload,
)

# ---------------------------------------------------------------------------
# Constants (must match run.py)
# ---------------------------------------------------------------------------
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

INPUT_NAMES = ["rigidity_kT", "dt_sec", "time_sec"]

DEFAULT_CSV = Path.home() / "Downloads" / "metamodel_ks" / "results.csv"
OUT_DIR = Path.home() / "Downloads" / "metamodel_ks" / "surrogates"

SBI_CONFIG = {
    "density_estimator": "maf",
    "max_num_epochs": 200,
    "training_batch_size": 64,
    "learning_rate": 5e-4,
    "stop_after_epochs": 30,
    "summary_samples": 512,
}

PYMC_CONFIG = {
    "draws": 500,
    "tune": 300,
    "chains": 2,
    "target_accept": 0.9,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path, average_seeds: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load CSV, filter, optionally average over seeds.

    Returns (X, metrics_dict) where:
        X: shape (n, 3) — rigidity_kT, dt_sec, time_sec
        metrics_dict: {metric_key: array of shape (n,)}
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok":
                continue
            # Skip rows with missing metric values
            if any(row.get(k, "") == "" for k in METRIC_KEYS):
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} valid rows from {csv_path}")

    if average_seeds:
        # Group by (rigidity_kT, dt_sec, time_sec), average metrics
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            key = (float(r["rigidity_kT"]), float(r["dt_sec"]), float(r["time_sec"]))
            groups.setdefault(key, []).append(r)

        avg_rows = []
        for (k, dt, t), group in sorted(groups.items()):
            avg = {"rigidity_kT": k, "dt_sec": dt, "time_sec": t}
            for mkey in METRIC_KEYS:
                vals = [float(r[mkey]) for r in group if r[mkey] != ""]
                avg[mkey] = np.mean(vals) if vals else np.nan
            avg_rows.append(avg)
        print(f"Averaged over seeds: {len(avg_rows)} training points "
              f"(from {len(rows)} raw rows)")
        rows = avg_rows

    X = np.column_stack([
        [float(r["rigidity_kT"]) for r in rows],
        [float(r["dt_sec"]) for r in rows],
        [float(r["time_sec"]) for r in rows],
    ])

    metrics = {}
    for mkey in METRIC_KEYS:
        metrics[mkey] = np.array([float(r[mkey]) for r in rows])

    return X, metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diagnostics(X: np.ndarray, metrics: dict, models: dict, out_path: Path):
    """2×4 predicted-vs-actual scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping diagnostics plot")
        return

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle("Surrogate Diagnostics: Predicted vs Actual", fontsize=14)

    inputs = {
        "rigidity_kT": X[:, 0],
        "dt_sec": X[:, 1],
        "time_sec": X[:, 2],
    }

    for idx, (mkey, mname) in enumerate(METRIC_NAMES):
        ax = axes[idx // 4, idx % 4]
        if mkey not in models:
            ax.set_title(f"{mname}\n(no model)", fontsize=10)
            continue

        model = models[mkey]
        y_actual = metrics[mkey]
        summary = model.summary(inputs)
        y_pred = np.array(summary["mean"])

        ax.scatter(y_actual, y_pred, alpha=0.15, s=5, c="steelblue")
        lo = min(y_actual.min(), y_pred.min())
        hi = max(y_actual.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5)

        rmse = float(np.sqrt(np.mean((y_pred - y_actual) ** 2)))
        ss_res = np.sum((y_pred - y_actual) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.set_title(f"{mname}\nRMSE={rmse:.3f}  R²={r2:.3f}", fontsize=10)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Diagnostics plot: {out_path}")


def plot_dynamics(X: np.ndarray, models: dict, out_path: Path):
    """2×4 dynamics plot: t on x-axis, lines per (K, dt)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Prediction grid
    k_vals = np.geomspace(1, 40, 6)
    dt_vals = np.unique(X[:, 1])
    t_vals = np.linspace(0.1, 10, 50)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_vals)))

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle("Surrogate Dynamics: metric(t) for selected K and dt", fontsize=14)

    for idx, (mkey, mname) in enumerate(METRIC_NAMES):
        ax = axes[idx // 4, idx % 4]
        if mkey not in models:
            ax.set_title(f"{mname}\n(no model)", fontsize=10)
            continue

        model = models[mkey]
        # Plot for smallest and largest dt
        for dt_i, dt_val in enumerate([dt_vals[0], dt_vals[-1]]):
            ls = "-" if dt_i == 0 else "--"
            for ki, k_val in enumerate(k_vals):
                inputs = {
                    "rigidity_kT": np.full(len(t_vals), k_val),
                    "dt_sec": np.full(len(t_vals), dt_val),
                    "time_sec": t_vals,
                }
                summary = model.summary(inputs)
                mean = np.array(summary["mean"])
                label = (f"K={k_val:.1f} dt={dt_val * 1e6:.1f}us"
                         if idx == 0 and dt_i == 0 else None)
                ax.plot(t_vals, mean, ls, color=colors[ki], alpha=0.7,
                        linewidth=1.2, label=label)

        ax.set_xlabel("time (s)")
        ax.set_title(mname, fontsize=10)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dynamics plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit surrogates on KS sweep data")
    parser.add_argument("--backend", choices=["sbi_npe", "pymc_gp"],
                        default="sbi_npe", help="Surrogate backend (default: sbi_npe)")
    parser.add_argument("--no-average", action="store_true",
                        help="Use all rows (don't average over seeds)")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_CSV,
                        help="Path to results CSV")
    args = parser.parse_args()

    if not args.results_csv.exists():
        print(f"Error: results CSV not found: {args.results_csv}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend_config = SBI_CONFIG if args.backend == "sbi_npe" else PYMC_CONFIG

    # Load data
    X, metrics = load_and_prepare(args.results_csv, average_seeds=not args.no_average)
    print(f"Training data: X.shape={X.shape}, backend={args.backend}")

    # Fit one surrogate per metric
    models = {}
    summary_data = {}

    for mkey, mname in METRIC_NAMES:
        y = metrics[mkey]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            print(f"  Skipping {mname}: only {valid.sum()} valid samples")
            continue

        print(f"\n{'=' * 60}")
        print(f"Fitting: {mname} ({valid.sum()} samples)")
        t0 = time.monotonic()

        model = fit_backend_model(
            backend=args.backend,
            x=X[valid],
            y=y[valid],
            input_names=INPUT_NAMES,
            output_name=mkey,
            backend_config=backend_config,
            seed=42,
        )

        elapsed = time.monotonic() - t0
        models[mkey] = model

        # Train error
        train_summary = model.summary({
            "rigidity_kT": X[valid, 0],
            "dt_sec": X[valid, 1],
            "time_sec": X[valid, 2],
        })
        y_pred = np.array(train_summary["mean"])
        rmse = float(np.sqrt(np.mean((y_pred - y[valid]) ** 2)))
        ss_res = np.sum((y_pred - y[valid]) ** 2)
        ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"  Train RMSE: {rmse:.4f}  R²: {r2:.4f}  ({elapsed:.1f}s)")

        # Save model artifact
        artifact_path = OUT_DIR / f"surrogate_{mkey}.json"
        save_backend_payload(model, artifact_path)
        print(f"  Artifact: {artifact_path}")

        summary_data[mkey] = {
            "metric_name": mname,
            "n_train": int(valid.sum()),
            "train_rmse": rmse,
            "train_r2": r2,
            "fit_time_sec": round(elapsed, 1),
            "backend": args.backend,
            "artifact_path": str(artifact_path),
        }

    # Save summary
    summary_path = OUT_DIR / "surrogate_summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f"\nSummary: {summary_path}")

    # Plots
    plot_diagnostics(X, metrics, models, OUT_DIR / "surrogate_diagnostics.png")
    plot_dynamics(X, models, OUT_DIR / "surrogate_dynamics.png")

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"{'Metric':<30s} {'RMSE':>10s} {'R²':>8s} {'Time':>8s}")
    print("-" * 70)
    for mkey, mname in METRIC_NAMES:
        if mkey in summary_data:
            s = summary_data[mkey]
            print(f"{mname:<30s} {s['train_rmse']:10.4f} {s['train_r2']:8.4f} "
                  f"{s['fit_time_sec']:7.1f}s")
        else:
            print(f"{mname:<30s} {'SKIPPED':>10s}")
    print(f"\nAll artifacts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
