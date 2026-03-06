"""KS example — demonstrates the bayesian_metamodeling framework API.

Runs a kinetic-segregation parameter sweep using the framework's spec-driven
pipeline (DOE planning, subprocess execution, centralized sweep storage), then
optionally fits a surrogate model from the stored data.

Profiles
--------
    --profile fast        3x3 grid, grid=16, 10 steps   (~10 sec)
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
    artifacts/ks_sweep_<profile>.csv   — sweep results (copy from store)
    artifacts/ks_sweep_<profile>.png   — depletion width heatmap
    artifacts/ks_surrogate_<backend>_<profile>.png — surrogate comparison (if fitted)

    Framework store (provenance):
    store/sweeps/<run_id>/sweep_rows.csv
    store/sweeps/<run_id>/sweep_manifest.json
    store/surrogates/<artifact_id>/  (if surrogate fitted)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from bayesian_metamodeling.adapters import resolve_adapter
from bayesian_metamodeling.designs import plan_points
from bayesian_metamodeling.runners import LocalProcessRunner
from bayesian_metamodeling.spec import SurrogateSpec, load_and_validate_modelspec
from bayesian_metamodeling.storage import persist_sweep
from bayesian_metamodeling.surrogates import fit_surrogate
from bayesian_metamodeling.surrogates.backends import fit_backend_model

# Repo root (needed for subprocess entrypoint resolution)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent.parent

# Add examples dir to path for plot_sweep import
sys.path.insert(0, str(SCRIPT_DIR))
from plot_sweep import pivot_to_grid, plot_heatmap, plot_sweep_and_surrogate  # noqa: E402

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
SPECS_DIR = SCRIPT_DIR / "specs"


def execute_design_point(spec, point_index, point, run_token):
    """Execute one DOE point using the framework adapter+runner pipeline."""
    from datetime import UTC, datetime

    adapter = resolve_adapter(spec)
    runner = LocalProcessRunner(timeout_sec=spec.runner.resources.walltime_min * 60)

    run_label = f"{spec.model.name}_{point_index + 1}"
    temp_run_dir = Path(spec.storage.root) / "_active" / run_token / run_label
    temp_run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC)
    status = "failed"
    returncode = 1
    outputs = {}
    error = ""
    stdout_text = ""
    stderr_text = ""

    try:
        materialization = adapter.materialize_inputs(
            spec=spec, point=point, run_dir=temp_run_dir, repo_root=REPO_ROOT,
        )
        run_result = runner.run(materialization=materialization, run_dir=temp_run_dir)
        returncode = run_result.returncode

        if run_result.stdout_path.exists():
            stdout_text = run_result.stdout_path.read_text(errors="replace")
        if run_result.stderr_path.exists():
            stderr_text = run_result.stderr_path.read_text(errors="replace")

        if returncode == 0:
            try:
                outputs = adapter.parse_outputs(spec=spec, run_dir=temp_run_dir)
                status = "success"
            except (ValueError, FileNotFoundError, json.JSONDecodeError, KeyError, OSError) as exc:
                error = f"Output parsing failed: {exc}"
                status = "failed"
                returncode = 1
        else:
            status = "failed"
    except Exception as exc:
        error = str(exc)
        status = "failed"
    finally:
        finished_at = datetime.now(UTC)
        duration_sec = (finished_at - started_at).total_seconds()
        shutil.rmtree(temp_run_dir, ignore_errors=True)

    return {
        "point_index": point_index,
        "point": point,
        "status": status,
        "returncode": returncode,
        "outputs": outputs,
        "error": error,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_sec": duration_sec,
    }


def run_sweep(profile: str):
    """Run a spec-driven sweep using the bayesian_metamodeling pipeline."""
    spec_path = SPECS_DIR / f"model.kinetic_segregation.{profile}.json"
    if not spec_path.exists():
        print(f"Error: spec not found: {spec_path}")
        sys.exit(1)

    # Step 1: Load and validate the ModelSpec
    payload = json.loads(spec_path.read_text())
    spec = load_and_validate_modelspec(payload)
    # Ensure subprocess uses the same Python interpreter as this script
    if spec.model.artifact.entrypoint and spec.model.artifact.entrypoint[0] == "python":
        spec.model.artifact.entrypoint[0] = sys.executable
    print(f"Spec validated: model={spec.model.name}, adapter={spec.adapter.id}")

    # Step 2: Plan DOE points
    points = plan_points(spec)
    print(f"DOE plan: {len(points)} points (strategy={spec.design.strategy})")

    # Step 3: Execute each point via adapter + subprocess runner
    print(f"\n=== Running KS sweep ({profile} profile, {len(points)} points) ===")
    run_token = uuid4().hex
    t0 = time.monotonic()
    results = []

    for idx, point in enumerate(points):
        result = execute_design_point(spec, idx, point, run_token)
        results.append(result)
        if result["status"] == "success":
            out = result["outputs"].get("depletion_width_nm", {})
            dep = out.get("depletion_width_nm", out) if isinstance(out, dict) else out
            t, r = point["time_sec"], point["rigidity_kT_nm2"]
            print(f"  [{idx + 1}/{len(points)}] t={t:.0f}s rig={r:.0f}kT -> depletion={dep:.1f} nm")
        else:
            print(f"  [{idx + 1}/{len(points)}] FAILED: {result['error']}")

    elapsed = time.monotonic() - t0

    # Step 4: Persist the sweep to the centralized store (full provenance)
    stored = persist_sweep(
        spec_payload=payload, spec=spec, point_results=results, execution_mode="serial",
    )
    sweep_csv = stored.run_dir / "sweep_rows.csv"
    sweep_manifest = stored.run_dir / "sweep_manifest.json"
    print(f"\nSweep stored: run_id={stored.run_id}")
    print(f"  CSV: {sweep_csv}")
    print(f"  Manifest: {sweep_manifest}")

    failed = sum(1 for r in results if r["status"] != "success")
    print(f"  {len(results) - failed}/{len(results)} points succeeded in {elapsed:.1f}s")

    return stored, spec, elapsed


def copy_and_plot(stored, profile: str):
    """Copy sweep CSV to artifacts and generate heatmap PNG."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy CSV to artifacts for easy access
    sweep_csv = stored.run_dir / "sweep_rows.csv"
    csv_dest = ARTIFACTS_DIR / f"ks_sweep_{profile}.csv"
    shutil.copy2(sweep_csv, csv_dest)
    print("\nArtifacts:")
    print(f"  CSV copy: {csv_dest}")

    # Load and plot
    from plot_sweep import load_sweep_csv
    times, rigidities, depletions = load_sweep_csv(sweep_csv)
    time_vals, rig_vals, grid = pivot_to_grid(times, rigidities, depletions)

    png_path = ARTIFACTS_DIR / f"ks_sweep_{profile}.png"
    plot_heatmap(time_vals, rig_vals, grid, png_path,
                 title=f"KS depletion width ({profile} profile)")
    print(f"  Heatmap: {png_path}")

    return time_vals, rig_vals, grid


def try_surrogate(profile: str, sweep_time, sweep_rig, sweep_grid):
    """Fit a surrogate using the framework's surrogate API, then plot comparison."""
    # Step 1: Load surrogate spec (references the same store as the sweep)
    for backend_name in ("pymc_gp", "sbi_npe"):
        spec_path = SPECS_DIR / f"surrogate.kinetic_segregation.{backend_name}.json"
        if not spec_path.exists():
            continue

        print(f"\n=== Surrogate fit ({backend_name}) ===")
        payload = json.loads(spec_path.read_text())
        spec = SurrogateSpec.model_validate(payload)
        print(f"  Spec: name={spec.name}, backend={spec.backend}")
        print(f"  Inputs: {spec.inputs}, Outputs: {spec.outputs}")
        print(f"  Dataset: {spec.dataset_ref}")

        # Step 2: Fit the surrogate (loads training data from sweep store automatically)
        try:
            artifact = fit_surrogate(spec)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"  Fit failed: {exc}")
            continue

        print(f"  Artifact stored: id={artifact['artifact_id']}")
        print(f"  Path: {artifact['artifact_path']}")

        # Step 3: Evaluate on a dense grid for visualization
        # Use the low-level backend API for dense prediction (eval_surrogate expects
        # pre-stored artifacts and JSON input format; here we want numpy arrays)
        from plot_sweep import load_sweep_csv
        csv_path = Path(spec.dataset_ref) / "sweeps"
        csv_files = sorted(csv_path.rglob("sweep_rows.csv"))
        if not csv_files:
            print("  No sweep CSV found for surrogate evaluation")
            continue

        times, rigs, deps = load_sweep_csv(csv_files[-1])
        X = np.column_stack([times, rigs])
        y = deps

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

        # Dense eval
        cfg_time = np.sort(np.unique(times))
        cfg_rig = np.sort(np.unique(rigs))
        dense_n = 25
        time_dense = np.linspace(cfg_time.min(), cfg_time.max(), dense_n)
        rig_dense = np.linspace(cfg_rig.min(), cfg_rig.max(), dense_n)
        tt, rr = np.meshgrid(time_dense, rig_dense, indexing="ij")
        dense_inputs = {"time_sec": tt.ravel(), "rigidity_kT_nm2": rr.ravel()}

        summary = model.summary(dense_inputs)
        mean_grid = np.array(summary["mean"]).reshape(dense_n, dense_n)
        if "std" in summary:
            std_grid = np.array(summary["std"]).reshape(dense_n, dense_n)
        else:
            std_grid = np.full((dense_n, dense_n), float(summary.get("sigma", 0.0)))

        png_path = ARTIFACTS_DIR / f"ks_surrogate_{backend_name}_{profile}.png"
        plot_sweep_and_surrogate(
            sweep_time, sweep_rig, sweep_grid,
            time_dense, rig_dense, mean_grid, std_grid,
            png_path, backend=backend_name,
        )
        print(f"  3-panel heatmap: {png_path}")
        return  # fit one backend only


def main():
    parser = argparse.ArgumentParser(
        description="KS example using the bayesian_metamodeling framework API"
    )
    parser.add_argument(
        "--profile", choices=["fast", "regular", "extensive"],
        default="fast", help="Simulation profile (default: fast)",
    )
    parser.add_argument(
        "--with-surrogate", action="store_true",
        help="Fit a surrogate after sweep (requires PyMC or SBI conda env)",
    )
    args = parser.parse_args()

    # Run sweep via framework pipeline
    stored, _, _ = run_sweep(args.profile)

    # Plot results
    time_vals, rig_vals, grid = copy_and_plot(stored, args.profile)

    # Optional surrogate
    if args.with_surrogate:
        try_surrogate(args.profile, time_vals, rig_vals, grid)

    print("\nDone.")


if __name__ == "__main__":
    main()
