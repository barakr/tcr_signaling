#!/usr/bin/env python3
"""Mixed-affinity pMHC sweep: does CD45 exclusion let a TCR discriminate a
high-affinity pMHC species from a low-affinity one competing on the same
patch?

Two-species mixture per run: a "high" species at
u_assoc_low * affinity_ratio and a "low" species at u_assoc_low, mixed in
`mixture_fraction` : (1 - mixture_fraction) proportions. The DOE sweeps
(affinity_ratio, mixture_fraction); u_assoc_low and all other physics/geometry
parameters are fixed (FIXED_ARGS + BASE_U_ASSOC_LOW below).

pmhc_species is a JSON array, which the float-only ModelSpec io_schema
(spec.json) cannot express directly -- so, like the sibling
experiments/ks_behavior_sweep, this sweep only uses bayesian_metamodeling for
DOE planning over the two swept SCALARS (affinity_ratio, mixture_fraction)
and calls the compiled ks_gpu binary directly, writing a per-run temporary
--params JSON file with the resolved two-species pmhc_species table (a
CLI-flag-only path doesn't exist for this -- see main.cpp's load_params_file).

Usage
-----
    cd projects/tcr_signaling
    python experiments/ks_mixed_affinity/run.py              # full sweep
    python experiments/ks_mixed_affinity/run.py --dry-run     # show plan only
    python experiments/ks_mixed_affinity/run.py --n-seeds 1 --max-runs 1  # quick test

Output goes to ~/Downloads/metamodel_ks_mixed_affinity/ (results.csv,
provenance.json, run_log.jsonl).
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
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMODULE_DIR = SCRIPT_DIR.parent.parent  # projects/tcr_signaling
REPO_ROOT = SUBMODULE_DIR.parent.parent  # metamodeler_codex_scaffold_docs
KS_DIR = SUBMODULE_DIR / "models" / "kinetic_segregation"

# ks_build owns the platform-dependent naming (ks_gpu vs ks_gpu.exe) and the
# multi-config build subdirectories, so this sweep works on Windows too.
sys.path.insert(0, str(KS_DIR))
import ks_build  # noqa: E402

BINARY = ks_build.find_binary() or (KS_DIR / ks_build.binary_name())
SPEC_PATH = SCRIPT_DIR / "spec.json"
OUT_DIR = Path.home() / "Downloads" / "metamodel_ks_mixed_affinity"

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
SIM_TIME = 2.0  # seconds -- enough for the bound-fraction readout to settle
BASE_U_ASSOC_LOW = 5.0  # kT, the "low affinity" species; "high" = this * ratio
SIGMA_BIND = 3.0  # nm, shared by both species
BIND_THRESHOLD = 5.0  # nm, --monitor-binding

FIXED_ARGS = [
    "--time_sec",
    str(SIM_TIME),
    "--rigidity_kT",
    "5",
    "--grid_size",
    "64",
    "--patch_size",
    "500",
    "--n_tcr",
    "60",
    "--n_cd45",
    "150",
    "--n_pmhc",
    "60",
    "--pmhc_mode",
    "inner_circle",
    "--pmhc_radius",
    "62.5",
    "--step_mode",
    "brownian",
    "--binding_mode",
    "gaussian",
    "--monitor-binding",
    str(BIND_THRESHOLD),
    "--monitor-interval",
    "1",
    "--no-gpu",  # multi-species pMHC is CPU-only (main.cpp's GPU guard) --
    # explicit here rather than relying on the stub-backend fallback, so this
    # sweep behaves identically on a GPU-capable (Apple) machine.
]

N_SEEDS_DEFAULT = 5
SEED_START = 42
TIMEOUT_SEC = 600

CSV_HEADER = [
    "affinity_ratio",
    "mixture_fraction",
    "seed",
    "u_assoc_high",
    "u_assoc_low",
    "n_pmhc_high",
    "n_pmhc_low",
    "bound_fraction_high",
    "bound_fraction_low",
    "depletion_width_nm",
    "duration_sec",
    "status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_hash(repo_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
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


def run_dir_for(base: Path, ratio: float, frac: float, seed: int) -> Path:
    return base / f"ratio{ratio:.2f}" / f"frac{frac:.2f}" / f"seed{seed}"


def run_single(
    ratio: float, frac: float, seed: int, run_dir: Path
) -> tuple[dict | None, float]:
    """Run one mixed-affinity simulation. Returns (result_dict_or_None,
    wall_clock_duration_sec)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    u_high = BASE_U_ASSOC_LOW * ratio
    species = [
        {"fraction": frac, "u_assoc": u_high, "sigma_bind": SIGMA_BIND},
        {"fraction": 1.0 - frac, "u_assoc": BASE_U_ASSOC_LOW, "sigma_bind": SIGMA_BIND},
    ]

    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=run_dir, delete=False
    ) as pf:
        json.dump({"pmhc_species": species}, pf)
        params_path = pf.name

    cmd = [
        str(BINARY),
        "--seed",
        str(seed),
        *FIXED_ARGS,
        "--params",
        params_path,
        "--run-dir",
        str(run_dir),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
        duration = time.monotonic() - t0
        if result.returncode != 0:
            (run_dir / "stderr.log").write_text(result.stderr)
            return None, duration
        data = json.loads(result.stdout.strip())
        (run_dir / "segregation.json").write_text(json.dumps(data, indent=2))
        bf = data.get("bound_fraction_by_species")
        n_by_sp = data.get("n_pmhc_by_species")
        if bf is None or n_by_sp is None:
            return None, duration
        return {
            "u_assoc_high": u_high,
            "u_assoc_low": BASE_U_ASSOC_LOW,
            "n_pmhc_high": n_by_sp[0],
            "n_pmhc_low": n_by_sp[1],
            "bound_fraction_high": bf[0],
            "bound_fraction_low": bf[1],
            "depletion_width_nm": data.get("depletion_width_nm"),
        }, duration
    except Exception:
        return None, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def write_provenance(spec_payload: dict, n_seeds: int, design_points: list):
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
        "binary": {"path": str(BINARY), "md5": _file_md5(BINARY)},
        "platform": {
            "system": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "spec": spec_payload,
        "fixed_args": FIXED_ARGS,
        "base_u_assoc_low": BASE_U_ASSOC_LOW,
        "sim_time_sec": SIM_TIME,
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="KS mixed-affinity pMHC sweep")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan and exit without running"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=N_SEEDS_DEFAULT,
        help=f"Seeds per DOE point (default {N_SEEDS_DEFAULT})",
    )
    parser.add_argument(
        "--max-runs", type=int, default=0, help="Stop after this many new runs (0=unlimited)"
    )
    args = parser.parse_args()

    spec_payload = json.loads(SPEC_PATH.read_text())
    try:
        from bayesian_metamodeling.designs import plan_points
        from bayesian_metamodeling.spec import load_and_validate_modelspec

        spec = load_and_validate_modelspec(spec_payload)
        design_points = plan_points(spec)
        print(
            f"DOE via bayesian_metamodeling: {len(design_points)} design points "
            f"(strategy={spec.design.strategy})"
        )
    except ImportError:
        print("Warning: bayesian_metamodeling not importable; generating grid manually")
        grid = spec_payload["design"]["grid"]
        from itertools import product

        keys = list(grid.keys())
        design_points = [dict(zip(keys, combo)) for combo in product(*(grid[k] for k in keys))]
        print(f"DOE (manual grid): {len(design_points)} design points")

    n_seeds = args.n_seeds
    total_runs = len(design_points) * n_seeds
    print(f"Seeds per point: {n_seeds} (start={SEED_START})")
    print(f"Total runs: {total_runs}")
    print(f"Sim: {SIM_TIME}s, u_assoc_low={BASE_U_ASSOC_LOW}")
    print(f"Binary: {BINARY} (exists={BINARY.exists()})")
    print(f"Output: {OUT_DIR}")

    schedule = []
    for pt in design_points:
        ratio = pt["affinity_ratio"]
        frac = pt["mixture_fraction"]
        for i in range(n_seeds):
            seed = SEED_START + i
            rd = run_dir_for(OUT_DIR / "runs", ratio, frac, seed)
            schedule.append((ratio, frac, seed, rd))

    if args.dry_run:
        print(f"\n--- DRY RUN: {len(schedule)} runs planned ---")
        for ratio, frac, seed, rd in schedule[:10]:
            print(f"  ratio={ratio:5.2f}  frac={frac:.2f}  seed={seed}  -> {rd}")
        if len(schedule) > 10:
            print(f"  ... and {len(schedule) - 10} more")
        return

    if not BINARY.exists():
        print(f"ERROR: binary not found at {BINARY}. Build it first (see README).")
        return 1

    write_provenance(spec_payload, n_seeds, design_points)
    print(f"Provenance written: {OUT_DIR / 'provenance.json'}")

    results_csv = OUT_DIR / "results.csv"
    log_path = OUT_DIR / "run_log.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_run_keys: set[tuple] = set()
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (float(row["affinity_ratio"]), float(row["mixture_fraction"]), int(row["seed"]))
                existing_run_keys.add(key)
        print(f"Loaded {len(existing_run_keys)} existing runs from CSV")

    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    csv_file = open(results_csv, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    if write_header:
        csv_writer.writeheader()
        csv_file.flush()
    log_file = open(log_path, "a")

    t0 = time.time()
    new_runs = 0

    for ratio, frac, seed, rd in schedule:
        key = (ratio, frac, seed)
        if key in existing_run_keys:
            continue

        result, duration = run_single(ratio, frac, seed, rd)
        new_runs += 1
        status = "ok" if result else "failed"
        row = {
            "affinity_ratio": ratio,
            "mixture_fraction": frac,
            "seed": seed,
            "duration_sec": f"{duration:.1f}",
            "status": status,
            **{k: "" for k in CSV_HEADER if k not in ("affinity_ratio", "mixture_fraction", "seed", "duration_sec", "status")},
        }
        if result:
            row.update(result)
        csv_writer.writerow(row)
        csv_file.flush()
        existing_run_keys.add(key)

        log_entry = {
            "affinity_ratio": ratio,
            "mixture_fraction": frac,
            "seed": seed,
            "status": status,
            "duration_sec": round(duration, 1),
            "run_dir": str(rd),
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        done_runs = len(existing_run_keys)
        if new_runs % 5 == 0 or new_runs == 1:
            elapsed = time.time() - t0
            rate = new_runs / elapsed if elapsed > 0 else 0
            eta = (total_runs - done_runs) / rate if rate > 0 else 0
            print(
                f"  [{done_runs}/{total_runs}] {new_runs} new, {elapsed:.0f}s elapsed, "
                f"ETA {eta / 60:.1f}min | ratio={ratio:.1f} frac={frac:.2f} s={seed} "
                f"-> {status} ({duration:.1f}s)"
            )

        if args.max_runs > 0 and new_runs >= args.max_runs:
            print(f"Stopping after {args.max_runs} new runs (--max-runs)")
            break

    csv_file.close()
    log_file.close()

    elapsed = time.time() - t0
    done_runs = len(existing_run_keys)
    print(f"\n{'=' * 90}")
    print(f"Completed: {done_runs}/{total_runs} runs ({new_runs} new) in {elapsed:.0f}s")
    print(f"Results: {results_csv}")
    print(f"Provenance: {OUT_DIR / 'provenance.json'}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
