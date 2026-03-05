"""KS sweep + surrogate heatmap example.

Demonstrates the full bayesian_metamodeling workflow for the kinetic-segregation
model: parameter sweep via ``bayesmm run``, surrogate fitting via
``bayesmm surrogate fit``, and 3-panel heatmap comparison (model output,
surrogate mean, surrogate std).

Profiles
--------
Three spec tiers control runtime vs accuracy:

    --profile fast        9 DOE points, grid=16, 10 steps   (~10 sec)
    --profile regular    16 DOE points, grid=32, 20 steps   (~1-3 min)
    --profile extensive  56 DOE points, grid=64, auto steps  (~20-30 min)

Two-env workflow
----------------
The pymc_gp and sbi_npe backends require different conda environments.
Run one backend per invocation:

    # PyMC backend (runs sweep + fit + heatmap)
    conda activate py312_bayesmm_pymc
    cd projects/tcr_signaling
    python examples/ks_sweep_and_surrogate.py --backend pymc_gp --profile fast

    # SBI backend (reuse existing sweep data)
    conda activate py312_bayesmm_sbi
    cd projects/tcr_signaling
    python examples/ks_sweep_and_surrogate.py --backend sbi_npe --skip-sweep

Output
------
PNG heatmaps are saved to ``artifacts/ks_heatmap_<backend>.png``.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to the tcr_signaling project root)
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent.parent

MODEL_SPECS = {
    "fast": EXAMPLE_DIR / "specs" / "model.kinetic_segregation.fast.json",
    "regular": EXAMPLE_DIR / "specs" / "model.kinetic_segregation.regular.json",
    "extensive": EXAMPLE_DIR / "specs" / "model.kinetic_segregation.extensive.json",
}
SURROGATE_SPECS = {
    "pymc_gp": EXAMPLE_DIR / "specs" / "surrogate.kinetic_segregation.pymc_gp.json",
    "sbi_npe": EXAMPLE_DIR / "specs" / "surrogate.kinetic_segregation.sbi_npe.json",
}
STORE_DIR = PROJECT_ROOT / "store"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# DOE grid values per profile (must match the spec JSON files)
PROFILE_GRIDS = {
    "fast": {
        "time": np.array([5.0, 30.0, 100.0]),
        "rigidity": np.array([1.0, 20.0, 100.0]),
    },
    "regular": {
        "time": np.array([5.0, 20.0, 50.0, 100.0]),
        "rigidity": np.array([1.0, 10.0, 30.0, 100.0]),
    },
    "extensive": {
        "time": np.array([5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0]),
        "rigidity": np.array([1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0]),
    },
}


# ---------------------------------------------------------------------------
# Step 1: Sweep
# ---------------------------------------------------------------------------


def run_sweep(profile: str) -> None:
    """Run the kinetic-segregation DOE sweep via ``bayesmm run``."""
    model_spec = MODEL_SPECS[profile]
    print(f"=== Running KS parameter sweep (profile: {profile}) ===")
    result = subprocess.run(
        ["bayesmm", "run", str(model_spec)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(f"bayesmm run exited with code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    print("Sweep complete.\n")


# ---------------------------------------------------------------------------
# Step 2: Load sweep data
# ---------------------------------------------------------------------------


def load_sweep_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the latest sweep CSV and return (time, rigidity, depletion) arrays."""
    sweep_csvs = sorted(STORE_DIR.glob("sweeps/**/sweep_rows.csv"))
    if not sweep_csvs:
        print("No sweep_rows.csv found. Run without --skip-sweep first.", file=sys.stderr)
        sys.exit(1)

    csv_path = sweep_csvs[-1]
    print(f"Loading sweep data from {csv_path}")

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


def sweep_to_grid(
    times: np.ndarray, rigidities: np.ndarray, depletions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot flat sweep arrays into 2-D grids aligned with DOE axes."""
    time_vals = np.sort(np.unique(times))
    rig_vals = np.sort(np.unique(rigidities))
    grid = np.full((len(time_vals), len(rig_vals)), np.nan)

    time_idx = {v: i for i, v in enumerate(time_vals)}
    rig_idx = {v: i for i, v in enumerate(rig_vals)}

    for t, r, d in zip(times, rigidities, depletions):
        grid[time_idx[t], rig_idx[r]] = d

    return time_vals, rig_vals, grid


# ---------------------------------------------------------------------------
# Step 3: Fit surrogate
# ---------------------------------------------------------------------------


def fit_surrogate(backend: str) -> None:
    """Fit a surrogate via ``bayesmm surrogate fit``."""
    spec_path = SURROGATE_SPECS[backend]
    print(f"=== Fitting surrogate ({backend}) ===")
    result = subprocess.run(
        ["bayesmm", "surrogate", "fit", str(spec_path)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(f"bayesmm surrogate fit exited with code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    print(f"Surrogate fit complete ({backend}).\n")


# ---------------------------------------------------------------------------
# Step 4: Dense eval via Python API
# ---------------------------------------------------------------------------

SPEC_NAMES = {
    "pymc_gp": "surrogate_kinetic_segregation",
    "sbi_npe": "surrogate_kinetic_segregation_sbi",
}


def dense_eval(
    backend: str, dense_n: int, profile: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the fitted surrogate on a dense meshgrid.

    Returns (time_dense, rig_dense, mean_grid, std_grid).
    """
    import os

    from bayesian_metamodeling.storage.surrogate_store import find_latest_artifact_for_spec
    from bayesian_metamodeling.surrogates.backends import load_backend_model

    grid_cfg = PROFILE_GRIDS[profile]
    time_dense = np.linspace(grid_cfg["time"].min(), grid_cfg["time"].max(), dense_n)
    rig_dense = np.linspace(grid_cfg["rigidity"].min(), grid_cfg["rigidity"].max(), dense_n)
    tt, rr = np.meshgrid(time_dense, rig_dense, indexing="ij")
    inputs = {"time_sec": tt.ravel(), "rigidity_kT_nm2": rr.ravel()}

    # Framework registry uses paths relative to repo root.
    prev_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        spec_name = SPEC_NAMES[backend]
        artifact_id, artifact_path = find_latest_artifact_for_spec(spec_name)
        print(f"Using artifact {artifact_id} for {spec_name}")

        artifact_meta = __import__("json").loads(artifact_path.read_text())
        payload_path = Path(artifact_meta["backend_payload"])

        model = load_backend_model(
            backend,
            payload_path,
            expected_inputs=["time_sec", "rigidity_kT_nm2"],
            expected_output="depletion_width_nm",
        )

        summary = model.summary(inputs)
    finally:
        os.chdir(prev_cwd)

    mean = np.array(summary["mean"]).reshape(dense_n, dense_n)

    if "std" in summary:
        std = np.array(summary["std"]).reshape(dense_n, dense_n)
    else:
        sigma = float(summary["sigma"])
        std = np.full((dense_n, dense_n), sigma)

    return time_dense, rig_dense, mean, std


# ---------------------------------------------------------------------------
# Step 5: Plot 3-panel heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    backend: str,
    sweep_time: np.ndarray,
    sweep_rig: np.ndarray,
    sweep_grid: np.ndarray,
    dense_time: np.ndarray,
    dense_rig: np.ndarray,
    mean_grid: np.ndarray,
    std_grid: np.ndarray,
) -> Path:
    """Create a 3-panel heatmap: model output | surrogate mean | surrogate std."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: model output from sweep
    ax = axes[0]
    im0 = ax.pcolormesh(sweep_rig, sweep_time, sweep_grid, shading="auto", cmap="viridis")
    ax.set_xlabel("rigidity (kT/nm\u00b2)")
    ax.set_ylabel("time (sec)")
    ax.set_title("Model output (sweep)")
    fig.colorbar(im0, ax=ax, label="depletion width (nm)")

    # Panel 2: surrogate mean (dense)
    ax = axes[1]
    im1 = ax.pcolormesh(dense_rig, dense_time, mean_grid, shading="auto", cmap="viridis")
    ax.set_xlabel("rigidity (kT/nm\u00b2)")
    ax.set_ylabel("time (sec)")
    ax.set_title(f"Surrogate mean ({backend})")
    fig.colorbar(im1, ax=ax, label="depletion width (nm)")

    # Panel 3: surrogate std (dense)
    ax = axes[2]
    im2 = ax.pcolormesh(dense_rig, dense_time, std_grid, shading="auto", cmap="magma")
    ax.set_xlabel("rigidity (kT/nm\u00b2)")
    ax.set_ylabel("time (sec)")
    ax.set_title(f"Surrogate std ({backend})")
    fig.colorbar(im2, ax=ax, label="std (nm)")

    fig.suptitle(
        "Kinetic segregation: model vs surrogate comparison",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"ks_heatmap_{backend}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="KS sweep + surrogate heatmap example")
    parser.add_argument(
        "--profile",
        choices=["fast", "regular", "extensive"],
        default="fast",
        help="Simulation profile: fast (~10s), regular (~3min), extensive (~30min)",
    )
    parser.add_argument(
        "--backend",
        choices=["pymc_gp", "sbi_npe", "both"],
        default="both",
        help="Surrogate backend(s) to fit and evaluate (default: both)",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Reuse existing sweep data instead of running a new sweep",
    )
    parser.add_argument(
        "--dense-n",
        type=int,
        default=25,
        help="Dense eval grid resolution per axis (default: 25 -> 625 points)",
    )
    args = parser.parse_args()

    # Determine backends to run
    if args.backend == "both":
        backends = ["pymc_gp", "sbi_npe"]
    else:
        backends = [args.backend]

    # Step 1: sweep
    if not args.skip_sweep:
        run_sweep(args.profile)

    # Step 2: load sweep data
    times, rigidities, depletions = load_sweep_data()
    sweep_time, sweep_rig, sweep_grid = sweep_to_grid(times, rigidities, depletions)

    # Steps 3-5: per backend
    for backend in backends:
        fit_surrogate(backend)

        dense_time, dense_rig, mean_grid, std_grid = dense_eval(backend, args.dense_n, args.profile)

        plot_heatmap(
            backend,
            sweep_time,
            sweep_rig,
            sweep_grid,
            dense_time,
            dense_rig,
            mean_grid,
            std_grid,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
