"""KS sweep example — pure Python API, no CLI dependency.

Runs the kinetic-segregation Monte Carlo model directly via ``simulate_ks()``,
sweeps over (time, rigidity) parameter pairs, and produces a heatmap PNG.

Profiles
--------
    --profile fast        3x3 grid, grid=16, 10 steps   (~5 sec)
    --profile regular     4x4 grid, grid=32, 20 steps   (~1-3 min)
    --profile extensive   7x8 grid, grid=64, auto steps  (~20-30 min)

Usage
-----
    cd projects/tcr_signaling
    python examples/ks_example.py --profile fast

    # With surrogate (requires PyMC or SBI conda env):
    python examples/ks_example.py --profile fast --with-surrogate

Output
------
    artifacts/ks_sweep_<profile>.csv   — sweep results
    artifacts/ks_sweep_<profile>.png   — depletion width heatmap
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so we can import the KS model directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from plot_sweep import pivot_to_grid, plot_heatmap, plot_sweep_and_surrogate  # noqa: E402

from projects.tcr_signaling.models.kinetic_segregation.model import simulate_ks  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# DOE grids per profile (must match the spec JSON files)
PROFILE_CONFIGS = {
    "fast": {
        "time_sec": [5.0, 30.0, 100.0],
        "rigidity_kT_nm2": [1.0, 20.0, 100.0],
        "n_tcr": 10,
        "n_cd45": 20,
        "n_steps": 10,
        "grid_size": 16,
    },
    "regular": {
        "time_sec": [5.0, 20.0, 50.0, 100.0],
        "rigidity_kT_nm2": [1.0, 10.0, 30.0, 100.0],
        "n_tcr": 30,
        "n_cd45": 60,
        "n_steps": 20,
        "grid_size": 32,
    },
    "extensive": {
        "time_sec": [5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0],
        "rigidity_kT_nm2": [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0],
        "n_tcr": 50,
        "n_cd45": 100,
        "n_steps": None,  # auto from time_sec
        "grid_size": 64,
    },
}

BASE_SEED = 42


def run_sweep(profile: str) -> tuple[list[dict], float]:
    """Run in-process DOE sweep, return (rows, elapsed_sec)."""
    cfg = PROFILE_CONFIGS[profile]
    time_vals = cfg["time_sec"]
    rig_vals = cfg["rigidity_kT_nm2"]
    n_points = len(time_vals) * len(rig_vals)

    print(f"=== KS sweep (profile: {profile}, {n_points} points) ===")
    rows: list[dict] = []
    t0 = time.monotonic()

    for t in time_vals:
        for r in rig_vals:
            # Per-point seed (same logic as __main__.py)
            raw = struct.pack("dd", t, r)
            input_hash = int(hashlib.md5(raw).hexdigest()[:8], 16)
            point_seed = BASE_SEED + input_hash

            result = simulate_ks(
                time_sec=t,
                rigidity_kT_nm2=r,
                n_tcr=cfg["n_tcr"],
                n_cd45=cfg["n_cd45"],
                n_steps=cfg["n_steps"],
                grid_size=cfg["grid_size"],
                seed=point_seed,
            )

            row = {
                "time_sec": t,
                "rigidity_kT_nm2": r,
                "depletion_width_nm": result["depletion_width_nm"],
                "accept_rate": result["accept_rate"],
                "n_steps_actual": result["n_steps_actual"],
            }
            rows.append(row)
            print(
                f"  t={t:6.1f}s  rig={r:6.1f}kT  "
                f"depletion={result['depletion_width_nm']:7.2f}nm  "
                f"accept={result['accept_rate']:.3f}"
            )

    elapsed = time.monotonic() - t0
    return rows, elapsed


def save_csv(rows: list[dict], csv_path: Path) -> None:
    """Write sweep rows to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {csv_path}")


def try_surrogate(
    rows: list[dict],
    profile: str,
    sweep_time: np.ndarray,
    sweep_rig: np.ndarray,
    sweep_grid: np.ndarray,
    dense_n: int = 25,
) -> None:
    """Attempt surrogate fit+eval+plot via Python API. Skip gracefully if deps missing."""
    try:
        from bayesian_metamodeling.surrogates.backends import fit_backend_model
    except ImportError:
        print("bayesian_metamodeling not installed — skipping surrogate step.")
        return

    X = np.array([[r["time_sec"], r["rigidity_kT_nm2"]] for r in rows])
    y = np.array([r["depletion_width_nm"] for r in rows])
    input_names = ["time_sec", "rigidity_kT_nm2"]
    output_name = "depletion_width_nm"

    # Try PyMC first, then SBI
    for backend_name in ("pymc_gp", "sbi_npe"):
        print(f"\n=== Surrogate fit ({backend_name}) ===")
        try:
            model = fit_backend_model(
                backend=backend_name,
                x=X,
                y=y,
                input_names=input_names,
                output_name=output_name,
                seed=BASE_SEED,
            )
        except Exception as exc:
            print(f"  {backend_name}: unavailable or failed ({exc})")
            continue

        # Train-set RMSE
        train_summary = model.summary({"time_sec": X[:, 0], "rigidity_kT_nm2": X[:, 1]})
        mean_pred = np.array(train_summary["mean"])
        rmse = float(np.sqrt(np.mean((mean_pred - y) ** 2)))
        print(f"  Train RMSE: {rmse:.3f} nm")

        # Dense eval for heatmap
        cfg = PROFILE_CONFIGS[profile]
        time_dense = np.linspace(min(cfg["time_sec"]), max(cfg["time_sec"]), dense_n)
        rig_dense = np.linspace(min(cfg["rigidity_kT_nm2"]), max(cfg["rigidity_kT_nm2"]), dense_n)
        tt, rr = np.meshgrid(time_dense, rig_dense, indexing="ij")
        dense_inputs = {"time_sec": tt.ravel(), "rigidity_kT_nm2": rr.ravel()}

        summary = model.summary(dense_inputs)
        mean_grid = np.array(summary["mean"]).reshape(dense_n, dense_n)

        if "std" in summary:
            std_grid = np.array(summary["std"]).reshape(dense_n, dense_n)
        else:
            sigma = float(summary.get("sigma", 0.0))
            std_grid = np.full((dense_n, dense_n), sigma)

        # 3-panel plot
        png_path = ARTIFACTS_DIR / f"ks_surrogate_{backend_name}_{profile}.png"
        plot_sweep_and_surrogate(
            sweep_time, sweep_rig, sweep_grid,
            time_dense, rig_dense, mean_grid, std_grid,
            png_path, backend=backend_name,
        )
        print(f"  Saved 3-panel heatmap to {png_path}")
        return  # Only fit one backend


def main() -> None:
    parser = argparse.ArgumentParser(description="KS sweep example (pure Python API)")
    parser.add_argument(
        "--profile",
        choices=["fast", "regular", "extensive"],
        default="fast",
        help="Simulation profile (default: fast)",
    )
    parser.add_argument(
        "--with-surrogate",
        action="store_true",
        help="Attempt surrogate fit after sweep (requires PyMC or SBI env)",
    )
    args = parser.parse_args()

    # Run sweep
    rows, elapsed = run_sweep(args.profile)
    print(f"\nSweep complete: {len(rows)} points in {elapsed:.1f}s\n")

    # Save CSV
    csv_path = ARTIFACTS_DIR / f"ks_sweep_{args.profile}.csv"
    save_csv(rows, csv_path)

    # Plot heatmap
    times = np.array([r["time_sec"] for r in rows])
    rigidities = np.array([r["rigidity_kT_nm2"] for r in rows])
    depletions = np.array([r["depletion_width_nm"] for r in rows])

    time_vals, rig_vals, grid = pivot_to_grid(times, rigidities, depletions)
    png_path = ARTIFACTS_DIR / f"ks_sweep_{args.profile}.png"
    plot_heatmap(
        time_vals,
        rig_vals,
        grid,
        png_path,
        title=f"KS depletion width ({args.profile} profile)",
    )
    print(f"Saved heatmap to {png_path}")

    # Optional surrogate
    if args.with_surrogate:
        try_surrogate(rows, args.profile, time_vals, rig_vals, grid)

    print("\nDone.")


if __name__ == "__main__":
    main()
