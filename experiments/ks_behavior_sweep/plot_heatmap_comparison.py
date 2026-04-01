#!/usr/bin/env python3
"""Heatmap comparison: simulation mean vs surrogate prediction for BT→CD45 NN P10.

Produces a figure with rows = selected time steps, columns = [sim mean, surrogate pred].
Each heatmap has dt on x-axis, K on y-axis, color = metric value.

Usage
-----
    cd projects/tcr_signaling
    conda activate py312_bayesmm_sbi
    python experiments/ks_behavior_sweep/plot_heatmap_comparison.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesian_metamodeling.surrogates.backends import load_backend_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METRIC_KEY = "bt_cd45_nn_p10_nm"
METRIC_LABEL = "BT→CD45 NN P10 (nm)"
BACKEND = "sbi_npe"

RESULTS_CSV = Path.home() / "Downloads" / "metamodel_ks" / "results.csv"
SURROGATE_PATH = (Path.home() / "Downloads" / "metamodel_ks" / "surrogates"
                  / f"surrogate_{METRIC_KEY}.json")
OUT_DIR = Path.home() / "Downloads" / "metamodel_ks" / "surrogates"

# Time snapshots to show as rows (seconds)
TIME_SNAPSHOTS = [0.5, 1.0, 2.0, 5.0, 10.0]

INPUT_NAMES = ["rigidity_kT", "dt_sec", "time_sec"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sim_data() -> dict[float, np.ndarray]:
    """Load simulation results, average over seeds.

    Returns {time_sec: 2D array of shape (n_K, n_dt)} with mean metric values.
    Also returns (k_vals, dt_vals) for axis labels.
    """
    rows = []
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or row[METRIC_KEY] == "":
                continue
            rows.append(row)

    k_vals = sorted(set(float(r["rigidity_kT"]) for r in rows))
    dt_vals = sorted(set(float(r["dt_sec"]) for r in rows))

    # Group by (K, dt, t), average over seeds
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        key = (float(r["rigidity_kT"]), float(r["dt_sec"]), float(r["time_sec"]))
        groups.setdefault(key, []).append(float(r[METRIC_KEY]))

    grids = {}
    for t_snap in TIME_SNAPSHOTS:
        # Find closest available time
        available_t = sorted(set(k[2] for k in groups.keys()))
        closest_t = min(available_t, key=lambda x: abs(x - t_snap))
        grid = np.full((len(k_vals), len(dt_vals)), np.nan)
        for ki, k in enumerate(k_vals):
            for di, dt in enumerate(dt_vals):
                vals = groups.get((k, dt, closest_t), [])
                if vals:
                    grid[ki, di] = np.mean(vals)
        grids[t_snap] = grid

    return grids, k_vals, dt_vals


def predict_surrogate(model, k_vals, dt_vals) -> dict[float, np.ndarray]:
    """Evaluate surrogate on the same grid as simulation data."""
    grids = {}
    for t_snap in TIME_SNAPSHOTS:
        grid = np.full((len(k_vals), len(dt_vals)), np.nan)
        for di, dt in enumerate(dt_vals):
            inputs = {
                "rigidity_kT": np.array(k_vals),
                "dt_sec": np.full(len(k_vals), dt),
                "time_sec": np.full(len(k_vals), t_snap),
            }
            summary = model.summary(inputs)
            grid[:, di] = np.array(summary["mean"])
        grids[t_snap] = grid
    return grids


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(sim_grids, surr_grids, k_vals, dt_vals, out_path):
    n_rows = len(TIME_SNAPSHOTS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Shared color scale across all panels
    all_vals = []
    for g in list(sim_grids.values()) + list(surr_grids.values()):
        all_vals.append(g[~np.isnan(g)])
    all_vals = np.concatenate(all_vals)
    vmin, vmax = np.nanpercentile(all_vals, [2, 98])

    dt_labels = [f"{d * 1e6:.1f}" for d in dt_vals]
    k_labels = [f"{k:.1f}" for k in k_vals]

    for ri, t_snap in enumerate(TIME_SNAPSHOTS):
        for ci, (grid, title_prefix) in enumerate([
            (sim_grids[t_snap], "Simulation (seed mean)"),
            (surr_grids[t_snap], "Surrogate (SBI NPE)"),
        ]):
            ax = axes[ri, ci]
            im = ax.imshow(grid, aspect="auto", origin="lower",
                           vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_xticks(range(len(dt_labels)))
            ax.set_xticklabels(dt_labels, fontsize=8)
            ax.set_yticks(range(len(k_labels)))
            ax.set_yticklabels(k_labels, fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel("dt (µs)", fontsize=10)
            if ci == 0:
                ax.set_ylabel("K (kT)", fontsize=10)
            ax.set_title(f"{title_prefix}\nt = {t_snap:.1f}s", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{METRIC_LABEL}: Simulation vs Surrogate", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading simulation data from {RESULTS_CSV}")
    sim_grids, k_vals, dt_vals = load_sim_data()

    print(f"Loading surrogate from {SURROGATE_PATH}")
    model = load_backend_model(BACKEND, SURROGATE_PATH,
                               expected_inputs=INPUT_NAMES,
                               expected_output=METRIC_KEY)

    print("Generating surrogate predictions...")
    surr_grids = predict_surrogate(model, k_vals, dt_vals)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "heatmap_bt_cd45_nn_p10_comparison.png"
    plot_comparison(sim_grids, surr_grids, k_vals, dt_vals, out_path)


if __name__ == "__main__":
    main()
