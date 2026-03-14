"""Rigidity-dependent step-size convergence test for TCR-pMHC gaussian binding.

For high rigidities (50, 75, 100 kT), sweep dt from 0.0001 ms to 0.1 ms
and measure bound TCR fraction. If bound fraction converges (plateaus) as
dt decreases, the unbinding at high rigidity is real physics. If it keeps
changing, it's a numerical artifact.

Parameters match the rigidity sweep movies:
  250 nm patch, 50x50 grid, 30 TCR/pMHC/CD45, gaussian binding, sigma_r=2 nm.
  0.1 second simulations for each (dt, rigidity) pair.

GPU enabled for speed. Outputs to ~/Downloads/ks_rigidity_convergence/.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent
_BINARY = _PKG_DIR / "ks_gpu"
_OUTPUT_DIR = Path.home() / "Downloads" / "ks_rigidity_convergence"
_RENDER_PYTHON = Path.home() / "miniconda3" / "envs" / "py314_bayesmm" / "bin" / "python"

# Simulation parameters (match rigidity sweep movies)
PATCH_SIZE = 250.0
GRID_SIZE = 50
N_TCR = 30
N_CD45 = 30
N_PMHC = 30
SIGMA_R = 2.0
PMHC_RADIUS = 21
MOL_REPULSION_EPS = 2.0
MOL_REPULSION_RCUT = 10.0
TIME_SEC = 0.1          # short runs — convergence check only
SEED = 42
N_FRAMES = 100          # target output frames per run

# Binding criterion
BIND_THRESHOLD = 3.0    # nm

# Rigidity values to test
RIGIDITY_VALUES = [50.0, 75.0, 100.0]

# dt sweep: 0.0001 ms (1e-7 s) to 0.1 ms (1e-4 s), 15 log-spaced values
DT_VALUES = np.logspace(np.log10(1e-7), np.log10(1e-4), 15).tolist()


def run_sim(tmp_dir: Path, dt: float, rigidity: float) -> Path | None:
    """Run one simulation with given dt and rigidity. Return run_dir or None on failure."""
    n_steps = int(round(TIME_SEC / dt))
    dump_interval = max(1, n_steps // N_FRAMES)
    run_dir = tmp_dir / f"rig{rigidity:.0f}_dt_{dt:.2e}"

    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT", str(rigidity),
        "--seed", str(SEED),
        "--grid_size", str(GRID_SIZE),
        "--n_tcr", str(N_TCR),
        "--n_cd45", str(N_CD45),
        "--n_pmhc", str(N_PMHC),
        "--pmhc_radius", str(PMHC_RADIUS),
        "--binding_mode", "gaussian",
        "--patch_size", str(PATCH_SIZE),
        "--sigma_r", str(SIGMA_R),
        "--dt", str(dt),
        "--mol_repulsion_eps", str(MOL_REPULSION_EPS),
        "--mol_repulsion_rcut", str(MOL_REPULSION_RCUT),
        "--run-dir", str(run_dir),
        "--dump-frames",
        "--dump-interval", str(dump_interval),
    ]

    step_nm = (2.0 * 1e4 * dt) ** 0.5
    print(f"  dt={dt*1000:.4f} ms  step={step_nm:.3f} nm  "
          f"n_steps={n_steps}  dump_every={dump_interval}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}): {result.stderr[:300]}")
        return None
    print(f"  Done in {elapsed:.1f}s")
    return run_dir


def parse_frames(run_dir: Path):
    """Read pMHC positions and per-frame TCR positions from binary dumps."""
    frames_dir = run_dir / "frames"

    pmhc_path = frames_dir / "pmhc.bin"
    pmhc_pos = np.fromfile(str(pmhc_path), dtype=np.float64).reshape(-1, 2)

    mol_files = sorted(frames_dir.glob("mol_*.bin"))
    tcr_frames = []
    for mf in mol_files:
        data = np.fromfile(str(mf), dtype=np.float64)
        tcr_data = data[: N_TCR * 2].reshape(N_TCR, 2)
        tcr_frames.append(tcr_data)

    return pmhc_pos, tcr_frames


def compute_bound_fraction(pmhc_pos, tcr_frames, threshold, patch_size):
    """For each frame, compute fraction of pMHCs with >= 1 TCR within threshold."""
    half = patch_size / 2.0
    n_pmhc = len(pmhc_pos)
    fractions = []

    for tcr_pos in tcr_frames:
        bound = 0
        for pi in range(n_pmhc):
            px, py = pmhc_pos[pi]
            is_bound = False
            for ti in range(len(tcr_pos)):
                ddx = tcr_pos[ti, 0] - px
                ddy = tcr_pos[ti, 1] - py
                # Periodic min-image
                if ddx > half:
                    ddx -= patch_size
                elif ddx < -half:
                    ddx += patch_size
                if ddy > half:
                    ddy -= patch_size
                elif ddy < -half:
                    ddy += patch_size
                r = (ddx * ddx + ddy * ddy) ** 0.5
                if r < threshold:
                    is_bound = True
                    break
            if is_bound:
                bound += 1
        fractions.append(bound / n_pmhc)

    return np.array(fractions)


def make_plot(results: dict):
    """Generate convergence plot: eq. bound fraction vs dt for each rigidity."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = {50.0: "tab:blue", 75.0: "tab:orange", 100.0: "tab:red"}

    for rigidity in sorted(results.keys()):
        dt_data = results[rigidity]
        dts = sorted(dt_data.keys())
        eq_fracs = []
        eq_stds = []
        for dt in dts:
            fracs = dt_data[dt]
            half_idx = len(fracs) // 2
            eq_fracs.append(np.mean(fracs[half_idx:]))
            eq_stds.append(np.std(fracs[half_idx:]))

        dt_ms = [d * 1000 for d in dts]
        color = colors.get(rigidity, "tab:gray")
        ax.errorbar(
            dt_ms, eq_fracs, yerr=eq_stds,
            fmt="o-", markersize=7, capsize=4, linewidth=2,
            color=color,
            label=f"κ = {rigidity:.0f} kT",
        )

    ax.set_xscale("log")
    ax.set_xlabel("dt (ms)", fontsize=13)
    ax.set_ylabel("Equilibrium bound fraction", fontsize=13)
    ax.set_title(
        f"Step-size convergence at high rigidity\n"
        f"({PATCH_SIZE:.0f} nm, {GRID_SIZE}×{GRID_SIZE}, "
        f"{N_TCR} TCR, {N_CD45} CD45, {N_PMHC} pMHC, "
        f"σ_r={SIGMA_R} nm, {TIME_SEC}s)",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    out_path = _OUTPUT_DIR / "rigidity_convergence.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(RIGIDITY_VALUES) * len(DT_VALUES)
    print(f"Rigidity convergence test")
    print(f"  Patch: {PATCH_SIZE} nm, grid {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Molecules: {N_TCR} TCR, {N_PMHC} pMHC, {N_CD45} CD45")
    print(f"  sigma_r={SIGMA_R} nm, bind threshold={BIND_THRESHOLD} nm")
    print(f"  Time: {TIME_SEC}s per run")
    print(f"  Rigidities: {RIGIDITY_VALUES} kT")
    print(f"  dt values: {len(DT_VALUES)} log-spaced from "
          f"{DT_VALUES[0]*1000:.4f} to {DT_VALUES[-1]*1000:.4f} ms")
    print(f"  Total runs: {total_runs}\n")

    # results[rigidity][dt] = fractions array
    results: dict[float, dict[float, np.ndarray]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_num = 0

        for rigidity in RIGIDITY_VALUES:
            results[rigidity] = {}
            print(f"\n{'='*60}")
            print(f"Rigidity = {rigidity:.0f} kT")
            print(f"{'='*60}")

            for dt in DT_VALUES:
                run_num += 1
                print(f"\n[{run_num}/{total_runs}] rig={rigidity:.0f} kT, dt={dt*1000:.4f} ms")

                run_dir = run_sim(tmp_path, dt, rigidity)
                if run_dir is None:
                    continue

                pmhc_pos, tcr_frames = parse_frames(run_dir)
                fracs = compute_bound_fraction(
                    pmhc_pos, tcr_frames, BIND_THRESHOLD, PATCH_SIZE
                )
                results[rigidity][dt] = fracs

                half_idx = len(fracs) // 2
                eq_mean = np.mean(fracs[half_idx:])
                eq_std = np.std(fracs[half_idx:])
                print(f"  -> {len(fracs)} frames, eq. bound = {eq_mean:.3f} +/- {eq_std:.3f}")

    # Generate plot
    if any(results[r] for r in results):
        make_plot(results)

        # Print summary table
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'rigidity':>10}  {'dt (ms)':>10}  {'step (nm)':>10}  "
              f"{'eq. bound':>10}  {'std':>8}")
        print("-" * 56)
        for rigidity in sorted(results.keys()):
            for dt in sorted(results[rigidity].keys()):
                fracs = results[rigidity][dt]
                half_idx = len(fracs) // 2
                step = (2.0 * 1e4 * dt) ** 0.5
                eq_mean = np.mean(fracs[half_idx:])
                eq_std = np.std(fracs[half_idx:])
                print(f"{rigidity:>10.0f}  {dt*1000:>10.4f}  {step:>10.3f}  "
                      f"{eq_mean:>10.3f}  {eq_std:>8.3f}")
    else:
        print("No successful runs!")


if __name__ == "__main__":
    main()
