#!/usr/bin/env python3
"""Heatmap dynamics: all 8 KS metrics as f(time, K) at the most conservative dt.

Left column: simulation seed-averaged data on the original grid.
Right column: surrogate predictions on a dense grid (50 time × 30 K, log K).
8 rows, one per metric.  Fixed dt = 1.25 µs (most conservative).

Usage
-----
    cd projects/tcr_signaling
    conda activate py312_bayesmm_sbi
    python experiments/ks_behavior_sweep/plot_heatmap_dynamics.py
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

BACKEND = "sbi_npe"
INPUT_NAMES = ["rigidity_kT", "dt_sec", "time_sec"]

RESULTS_CSV = Path.home() / "Downloads" / "metamodel_ks" / "results.csv"
SURROGATE_DIR = Path.home() / "Downloads" / "metamodel_ks" / "surrogates"
OUT_DIR = SURROGATE_DIR

DT_FIXED = 1.25e-6  # most conservative dt

# Dense surrogate grid
N_T_DENSE = 50
N_K_DENSE = 30
T_RANGE = (0.1, 10.0)
K_RANGE = (1.0, 40.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sim_data():
    """Load simulation results at DT_FIXED, average over seeds.

    Returns (t_vals, k_vals, grids) where grids maps metric_key →
    2D array of shape (n_K, n_t).
    """
    rows = []
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            if abs(float(row["dt_sec"]) - DT_FIXED) > 1e-8:
                continue
            rows.append(row)

    k_vals = sorted(set(float(r["rigidity_kT"]) for r in rows))
    t_vals = sorted(set(float(r["time_sec"]) for r in rows))

    # Group by (K, t), average over seeds per metric
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (float(r["rigidity_kT"]), float(r["time_sec"]))
        groups.setdefault(key, []).append(r)

    grids = {}
    for mkey in METRIC_KEYS:
        grid = np.full((len(k_vals), len(t_vals)), np.nan)
        for ki, k in enumerate(k_vals):
            for ti, t in enumerate(t_vals):
                recs = groups.get((k, t), [])
                vals = [float(r[mkey]) for r in recs if r.get(mkey, "") != ""]
                if vals:
                    grid[ki, ti] = np.mean(vals)
        grids[mkey] = grid

    return np.array(t_vals), np.array(k_vals), grids


def predict_dense(models, t_dense, k_dense):
    """Evaluate each surrogate on the dense t × K grid at fixed dt."""
    grids = {}
    for mkey in METRIC_KEYS:
        if mkey not in models:
            continue
        model = models[mkey]
        grid = np.empty((len(k_dense), len(t_dense)))
        for ti, t in enumerate(t_dense):
            inputs = {
                "rigidity_kT": k_dense,
                "dt_sec": np.full(len(k_dense), DT_FIXED),
                "time_sec": np.full(len(k_dense), t),
            }
            summary = model.summary(inputs)
            grid[:, ti] = np.array(summary["mean"])
        grids[mkey] = grid
    return grids


def _log_edges(centers):
    """Build pcolormesh edge array for log-spaced centers."""
    log_c = np.log10(centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (log_c[:-1] + log_c[1:])
    edges[0] = log_c[0] - (edges[1] - log_c[0])
    edges[-1] = log_c[-1] + (log_c[-1] - edges[-2])
    return 10**edges


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_all(sim_grids, surr_grids, t_sim, k_sim, t_dense, k_dense, out_path):
    n_rows = len(METRIC_NAMES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.2 * n_rows))

    for ri, (mkey, mname) in enumerate(METRIC_NAMES):
        sim_grid = sim_grids.get(mkey)
        surr_grid = surr_grids.get(mkey)

        # Shared color scale per metric
        all_vals = []
        if sim_grid is not None:
            all_vals.append(sim_grid[~np.isnan(sim_grid)])
        if surr_grid is not None:
            all_vals.append(surr_grid[~np.isnan(surr_grid)])
        if all_vals:
            combined = np.concatenate(all_vals)
            vmin, vmax = np.nanpercentile(combined, [2, 98])
        else:
            vmin, vmax = 0, 1

        # --- Left: simulation (sparse) ---
        ax_sim = axes[ri, 0]
        if sim_grid is not None:
            im = ax_sim.imshow(
                sim_grid,
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                extent=[t_sim[0], t_sim[-1], 0, len(k_sim) - 1],
            )
            ax_sim.set_yticks(range(len(k_sim)))
            ax_sim.set_yticklabels([f"{k:.1f}" for k in k_sim], fontsize=6)
            fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
        else:
            ax_sim.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sim.transAxes)
        if ri == n_rows - 1:
            ax_sim.set_xlabel("time (s)", fontsize=10)
        ax_sim.set_ylabel("K (kT)", fontsize=9)
        ax_sim.set_title(f"{mname}\nSimulation (seed mean)", fontsize=9)

        # --- Right: surrogate (dense, log K axis) ---
        ax_surr = axes[ri, 1]
        if surr_grid is not None:
            k_edges = _log_edges(k_dense)
            # linear edges for time
            t_edges = np.empty(len(t_dense) + 1)
            t_edges[1:-1] = 0.5 * (t_dense[:-1] + t_dense[1:])
            t_edges[0] = t_dense[0] - (t_edges[1] - t_dense[0])
            t_edges[-1] = t_dense[-1] + (t_dense[-1] - t_edges[-2])

            pcm = ax_surr.pcolormesh(
                t_edges, k_edges, surr_grid, vmin=vmin, vmax=vmax, cmap="viridis", shading="flat"
            )
            ax_surr.set_yscale("log")
            # Overlay training points
            tt, kk = np.meshgrid(t_sim[::10], k_sim, indexing="ij")
            ax_surr.scatter(
                tt.ravel(), kk.ravel(), c="white", s=6, edgecolors="black", linewidths=0.3, zorder=5
            )
            fig.colorbar(pcm, ax=ax_surr, fraction=0.046, pad=0.04)
        else:
            ax_surr.text(
                0.5, 0.5, "No model", ha="center", va="center", transform=ax_surr.transAxes
            )
        if ri == n_rows - 1:
            ax_surr.set_xlabel("time (s)", fontsize=10)
        ax_surr.set_ylabel("K (kT)", fontsize=9)
        ax_surr.set_title(f"{mname}\nSurrogate ({N_T_DENSE}×{N_K_DENSE} dense)", fontsize=9)

    fig.suptitle(
        f"KS Metric Dynamics (dt = {DT_FIXED * 1e6:.2f} µs): Simulation vs Surrogate",
        fontsize=13,
        y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Loading simulation data (dt={DT_FIXED * 1e6:.2f} µs) from {RESULTS_CSV}")
    t_sim, k_sim, sim_grids = load_sim_data()
    print(f"  Sim grid: {len(k_sim)} K × {len(t_sim)} t")

    # Load all 8 surrogate models
    models = {}
    for mkey, mname in METRIC_NAMES:
        path = SURROGATE_DIR / f"surrogate_{mkey}.json"
        if path.exists():
            models[mkey] = load_backend_model(
                BACKEND, path, expected_inputs=INPUT_NAMES, expected_output=mkey
            )
            print(f"  Loaded surrogate: {mname}")
        else:
            print(f"  Missing surrogate: {mname} ({path})")

    # Dense grid
    t_dense = np.linspace(*T_RANGE, N_T_DENSE)
    k_dense = np.geomspace(*K_RANGE, N_K_DENSE)
    print(
        f"Dense grid: {N_T_DENSE} t × {N_K_DENSE} K "
        f"(t: {T_RANGE[0]}-{T_RANGE[1]}s, K: {K_RANGE[0]}-{K_RANGE[1]} kT)"
    )

    print("Generating dense surrogate predictions (8 metrics)...")
    surr_grids = predict_dense(models, t_dense, k_dense)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "heatmap_all_metrics_dynamics.png"
    plot_all(sim_grids, surr_grids, t_sim, k_sim, t_dense, k_dense, out_path)


if __name__ == "__main__":
    main()
