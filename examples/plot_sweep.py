"""Shared heatmap plotting for KS sweep results.

Loads sweep data from a CSV file and produces a depletion-width heatmap PNG.
Used by both ``ks_example.py`` (import) and ``ks_example_cli.sh`` (subprocess).

Usage (standalone):
    python plot_sweep.py --csv path/to/sweep.csv --output artifacts/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_sweep_csv(
    csv_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load sweep CSV and return (time, rigidity, depletion) arrays."""
    times: list[float] = []
    rigidities: list[float] = []
    depletions: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_sec"]))
            rigidities.append(float(row["rigidity_kT_nm2"]))
            depletions.append(float(row["depletion_width_nm"]))

    return np.array(times), np.array(rigidities), np.array(depletions)


def pivot_to_grid(
    times: NDArray[np.float64],
    rigidities: NDArray[np.float64],
    depletions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pivot flat arrays into 2-D grid (time x rigidity)."""
    time_vals = np.sort(np.unique(times))
    rig_vals = np.sort(np.unique(rigidities))
    grid = np.full((len(time_vals), len(rig_vals)), np.nan)

    time_idx = {v: i for i, v in enumerate(time_vals)}
    rig_idx = {v: i for i, v in enumerate(rig_vals)}

    for t, r, d in zip(times, rigidities, depletions):
        grid[time_idx[t], rig_idx[r]] = d

    return time_vals, rig_vals, grid


def plot_heatmap(
    time_vals: NDArray[np.float64],
    rig_vals: NDArray[np.float64],
    grid: NDArray[np.float64],
    output_path: Path,
    title: str = "KS depletion width",
) -> Path:
    """Create a single-panel heatmap and save as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(rig_vals, time_vals, grid, shading="auto", cmap="viridis")
    ax.set_xlabel("rigidity (kT/nm\u00b2)")
    ax.set_ylabel("time (sec)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="depletion width (nm)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sweep_and_surrogate(
    sweep_time: NDArray[np.float64],
    sweep_rig: NDArray[np.float64],
    sweep_grid: NDArray[np.float64],
    dense_time: NDArray[np.float64],
    dense_rig: NDArray[np.float64],
    mean_grid: NDArray[np.float64],
    std_grid: NDArray[np.float64],
    output_path: Path,
    backend: str = "surrogate",
) -> Path:
    """Create a 3-panel heatmap: model output | surrogate mean | surrogate std."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    im0 = ax.pcolormesh(sweep_rig, sweep_time, sweep_grid, shading="auto", cmap="viridis")
    ax.set_xlabel("rigidity (kT/nm\u00b2)")
    ax.set_ylabel("time (sec)")
    ax.set_title("Model output (sweep)")
    fig.colorbar(im0, ax=ax, label="depletion width (nm)")

    ax = axes[1]
    im1 = ax.pcolormesh(dense_rig, dense_time, mean_grid, shading="auto", cmap="viridis")
    ax.set_xlabel("rigidity (kT/nm\u00b2)")
    ax.set_ylabel("time (sec)")
    ax.set_title(f"Surrogate mean ({backend})")
    fig.colorbar(im1, ax=ax, label="depletion width (nm)")

    ax = axes[2]
    im2 = ax.pcolormesh(dense_rig, dense_time, std_grid, shading="auto", cmap="magma")
    ax.set_xlabel("rigidity (kT/nm\u00b2)")
    ax.set_ylabel("time (sec)")
    ax.set_title(f"Surrogate std ({backend})")
    fig.colorbar(im2, ax=ax, label="std (nm)")

    fig.suptitle("Kinetic segregation: model vs surrogate", fontsize=14, fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def fit_and_plot_surrogate(
    csv_path: Path,
    surrogate_spec_path: Path,
    output_dir: Path,
    dense_n: int = 25,
) -> Path:
    """Fit a surrogate from sweep data and produce 3-panel comparison plot."""
    import json

    from bayesian_metamodeling.spec import SurrogateSpec
    from bayesian_metamodeling.surrogates.backends import fit_backend_model

    payload = json.loads(surrogate_spec_path.read_text())
    spec = SurrogateSpec.model_validate(payload)

    times, rigidities, depletions = load_sweep_csv(csv_path)
    X = np.column_stack([times, rigidities])
    y = depletions

    print(f"Fitting surrogate ({spec.backend}) on {len(y)} points...")
    model = fit_backend_model(
        backend=spec.backend, x=X, y=y,
        input_names=spec.inputs, output_name=spec.outputs[0],
        backend_config=spec.backend_config, seed=spec.seed,
    )

    # Train RMSE
    train_summary = model.summary({"time_sec": X[:, 0], "rigidity_kT_nm2": X[:, 1]})
    mean_pred = np.array(train_summary["mean"])
    rmse = float(np.sqrt(np.mean((mean_pred - y) ** 2)))
    print(f"  Train RMSE: {rmse:.1f} nm")

    # Dense evaluation grid
    time_dense = np.linspace(times.min(), times.max(), dense_n)
    rig_dense = np.linspace(rigidities.min(), rigidities.max(), dense_n)
    tt, rr = np.meshgrid(time_dense, rig_dense, indexing="ij")
    dense_inputs = {"time_sec": tt.ravel(), "rigidity_kT_nm2": rr.ravel()}

    summary = model.summary(dense_inputs)
    mean_grid = np.array(summary["mean"]).reshape(dense_n, dense_n)
    if "std" in summary:
        std_grid = np.array(summary["std"]).reshape(dense_n, dense_n)
    else:
        std_grid = np.full((dense_n, dense_n), float(summary.get("sigma", 0.0)))

    # Sweep grid for left panel
    time_vals, rig_vals, sweep_grid = pivot_to_grid(times, rigidities, depletions)

    png_path = output_dir / f"ks_surrogate_{spec.backend}_heatmap.png"
    plot_sweep_and_surrogate(
        time_vals, rig_vals, sweep_grid,
        time_dense, rig_dense, mean_grid, std_grid,
        png_path, backend=spec.backend,
    )
    print(f"  Saved 3-panel heatmap to {png_path}")
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KS sweep heatmap from CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to sweep CSV file")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts",
        help="Output directory for PNG (default: artifacts/)",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument(
        "--surrogate-spec",
        type=str,
        default=None,
        help="Path to SurrogateSpec JSON — fit surrogate and produce 3-panel comparison",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Try to find latest sweep_rows.csv in a store directory
        store = Path(args.csv)
        candidates = sorted(store.glob("sweeps/**/sweep_rows.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV found at {args.csv}")
        csv_path = candidates[-1]

    times, rigidities, depletions = load_sweep_csv(csv_path)
    time_vals, rig_vals, grid = pivot_to_grid(times, rigidities, depletions)

    out_dir = Path(args.output)
    out_path = out_dir / "ks_sweep_heatmap.png"
    title = args.title or "KS depletion width (sweep)"

    plot_heatmap(time_vals, rig_vals, grid, out_path, title=title)
    print(f"Saved heatmap to {out_path}")

    if args.surrogate_spec:
        fit_and_plot_surrogate(csv_path, Path(args.surrogate_spec), out_dir)


if __name__ == "__main__":
    main()
