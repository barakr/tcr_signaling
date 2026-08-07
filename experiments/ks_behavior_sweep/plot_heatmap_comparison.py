#!/usr/bin/env python3
"""Heatmap comparison: simulation mean vs dense surrogate prediction for BT→CD45 NN P10.

Left column: simulation seed-averaged data on the original sparse grid (10 K × 4 dt).
Right column: surrogate predictions on a dense log-spaced grid (30 K × 30 dt),
              interpolating/extrapolating beyond training points.
Rows = selected time snapshots.
Training points are overlaid as white dots on the surrogate panels.

Usage
-----
    cd projects/tcr_signaling
    conda activate py312_bayesmm_sbi
    python experiments/ks_behavior_sweep/plot_heatmap_comparison.py
"""

from __future__ import annotations

import csv
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
SURROGATE_PATH = (
    Path.home() / "Downloads" / "metamodel_ks" / "surrogates" / f"surrogate_{METRIC_KEY}.json"
)
OUT_DIR = Path.home() / "Downloads" / "metamodel_ks" / "surrogates"

TIME_SNAPSHOTS = [0.5, 1.0, 2.0, 5.0, 10.0]
INPUT_NAMES = ["rigidity_kT", "dt_sec", "time_sec"]

# Dense surrogate grid (log-spaced)
N_K_DENSE = 30
N_DT_DENSE = 30
K_RANGE = (1.0, 40.0)
DT_RANGE = (1.0e-6, 30e-6)  # slightly beyond training range [1.25, 25] µs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sim_data():
    """Load simulation results, average over seeds.

    Returns (grids_dict, k_vals, dt_vals) where grids_dict maps
    time_sec → 2D array of shape (n_K, n_dt).
    """
    rows = []
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or row[METRIC_KEY] == "":
                continue
            rows.append(row)

    k_vals = sorted(set(float(r["rigidity_kT"]) for r in rows))
    dt_vals = sorted(set(float(r["dt_sec"]) for r in rows))

    groups: dict[tuple, list[float]] = {}
    for r in rows:
        key = (float(r["rigidity_kT"]), float(r["dt_sec"]), float(r["time_sec"]))
        groups.setdefault(key, []).append(float(r[METRIC_KEY]))

    available_t = sorted(set(k[2] for k in groups.keys()))
    grids = {}
    for t_snap in TIME_SNAPSHOTS:
        closest_t = min(available_t, key=lambda x: abs(x - t_snap))
        grid = np.full((len(k_vals), len(dt_vals)), np.nan)
        for ki, k in enumerate(k_vals):
            for di, dt in enumerate(dt_vals):
                vals = groups.get((k, dt, closest_t), [])
                if vals:
                    grid[ki, di] = np.mean(vals)
        grids[t_snap] = grid

    return grids, np.array(k_vals), np.array(dt_vals)


def predict_dense(model, k_dense, dt_dense):
    """Evaluate surrogate on a dense K × dt grid for each time snapshot."""
    grids = {}
    for t_snap in TIME_SNAPSHOTS:
        grid = np.empty((len(k_dense), len(dt_dense)))
        for di, dt in enumerate(dt_dense):
            inputs = {
                "rigidity_kT": k_dense,
                "dt_sec": np.full(len(k_dense), dt),
                "time_sec": np.full(len(k_dense), t_snap),
            }
            summary = model.summary(inputs)
            grid[:, di] = np.array(summary["mean"])
        grids[t_snap] = grid
    return grids


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(sim_grids, surr_grids, k_sim, dt_sim, k_dense, dt_dense, out_path):
    n_rows = len(TIME_SNAPSHOTS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Shared color scale from both sim and surrogate data
    all_vals = []
    for g in list(sim_grids.values()) + list(surr_grids.values()):
        all_vals.append(g[~np.isnan(g)])
    all_vals = np.concatenate(all_vals)
    vmin, vmax = np.nanpercentile(all_vals, [2, 98])

    for ri, t_snap in enumerate(TIME_SNAPSHOTS):
        # --- Left: simulation (sparse grid, imshow) ---
        ax_sim = axes[ri, 0]
        sim_grid = sim_grids[t_snap]
        im = ax_sim.imshow(
            sim_grid, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis"
        )
        ax_sim.set_xticks(range(len(dt_sim)))
        ax_sim.set_xticklabels([f"{d * 1e6:.1f}" for d in dt_sim], fontsize=7)
        ax_sim.set_yticks(range(len(k_sim)))
        ax_sim.set_yticklabels([f"{k:.1f}" for k in k_sim], fontsize=7)
        if ri == n_rows - 1:
            ax_sim.set_xlabel("dt (µs)", fontsize=10)
        ax_sim.set_ylabel("K (kT)", fontsize=10)
        ax_sim.set_title(f"Simulation (seed mean)\nt = {t_snap:.1f}s", fontsize=10)
        fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)

        # --- Right: surrogate (dense grid, pcolormesh with log axes) ---
        ax_surr = axes[ri, 1]
        dt_us = dt_dense * 1e6  # convert to µs for axis
        # pcolormesh needs edge arrays (n+1 values)
        k_edges = _log_edges(k_dense)
        dt_edges = _log_edges(dt_us)
        pcm = ax_surr.pcolormesh(
            dt_edges,
            k_edges,
            surr_grids[t_snap],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            shading="flat",
        )
        ax_surr.set_xscale("log")
        ax_surr.set_yscale("log")
        # Overlay training points
        dt_sim_us = dt_sim * 1e6
        kk, dd = np.meshgrid(k_sim, dt_sim_us, indexing="ij")
        ax_surr.scatter(
            dd.ravel(),
            kk.ravel(),
            c="white",
            s=12,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            label="training points" if ri == 0 else None,
        )
        if ri == n_rows - 1:
            ax_surr.set_xlabel("dt (µs)", fontsize=10)
        ax_surr.set_ylabel("K (kT)", fontsize=10)
        ax_surr.set_title(
            f"Surrogate (SBI NPE, {N_K_DENSE}×{N_DT_DENSE} dense)\nt = {t_snap:.1f}s", fontsize=10
        )
        fig.colorbar(pcm, ax=ax_surr, fraction=0.046, pad=0.04)
        if ri == 0:
            ax_surr.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"{METRIC_LABEL}: Simulation vs Dense Surrogate Interpolation", fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _log_edges(centers):
    """Build pcolormesh edge array for log-spaced centers."""
    log_c = np.log10(centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (log_c[:-1] + log_c[1:])
    edges[0] = log_c[0] - (edges[1] - log_c[0])
    edges[-1] = log_c[-1] + (log_c[-1] - edges[-2])
    return 10**edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Loading simulation data from {RESULTS_CSV}")
    sim_grids, k_sim, dt_sim = load_sim_data()

    print(f"Loading surrogate from {SURROGATE_PATH}")
    model = load_backend_model(
        BACKEND, SURROGATE_PATH, expected_inputs=INPUT_NAMES, expected_output=METRIC_KEY
    )

    # Dense log-spaced grid for surrogate evaluation
    k_dense = np.geomspace(*K_RANGE, N_K_DENSE)
    dt_dense = np.geomspace(*DT_RANGE, N_DT_DENSE)
    print(
        f"Dense surrogate grid: {N_K_DENSE} K × {N_DT_DENSE} dt "
        f"(K: {K_RANGE[0]}-{K_RANGE[1]} kT, dt: {DT_RANGE[0] * 1e6:.1f}-{DT_RANGE[1] * 1e6:.1f} µs)"
    )

    print("Generating dense surrogate predictions...")
    surr_grids = predict_dense(model, k_dense, dt_dense)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "heatmap_bt_cd45_nn_p10_comparison.png"
    plot_comparison(sim_grids, surr_grids, k_sim, dt_sim, k_dense, dt_dense, out_path)


if __name__ == "__main__":
    main()
