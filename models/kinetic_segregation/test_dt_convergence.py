"""Step-size convergence test for TCR-pMHC gaussian binding.

25 nm box, 10 TCR + 10 pMHC, no CD45, gaussian binding with sigma_r=2 nm.
Sweeps dt from 0.005 ms to 0.2 ms (50 log-spaced values).
Measures fraction of "bound" pMHCs (TCR within 3 nm) using second half of run.
GPU enabled for speed.  Outputs convergence plot to ~/Downloads/ks_dt_convergence/.
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
_OUTPUT_DIR = Path.home() / "Downloads" / "ks_dt_convergence"
_RENDER_PYTHON = Path.home() / "miniconda3" / "envs" / "py314_bayesmm" / "bin" / "python"

# Simulation parameters
PATCH_SIZE = 25.0       # nm — tiny box
GRID_SIZE = 10          # dx = 2.5 nm (coarser grid for small box)
N_TCR = 10
N_PMHC = 10
N_CD45 = 0              # isolate TCR-pMHC interaction
SIGMA_R = 2.0           # nm
PMHC_RADIUS = 8.0       # placement disc radius (nm)
RIGIDITY = 20.0         # kT/nm²
TIME_SEC = 5.0          # seconds of simulation
SEED = 42
N_FRAMES = 100          # target number of output frames

# Binding criterion: TCR within this distance of a pMHC
BIND_THRESHOLD = 3.0    # nm

# dt sweep: 0.005 ms to 0.2 ms (50 log-spaced values)
# Auto-cal default: target_step = sigma_r/2 = 1nm → dt = 1²/(2·10000) = 5e-5 s = 0.05 ms
DT_VALUES = np.logspace(np.log10(5e-6), np.log10(2e-4), 50).tolist()
AUTO_CAL_DT = 5e-5  # for annotation on plot


def run_sim(tmp_dir: Path, dt: float) -> Path | None:
    """Run one simulation with given dt. Return run_dir or None on failure."""
    n_steps = int(round(TIME_SEC / dt))
    dump_interval = max(1, n_steps // N_FRAMES)
    run_dir = tmp_dir / f"dt_{dt:.6f}"

    cmd = [
        str(_BINARY),
        "--time_sec", str(TIME_SEC),
        "--rigidity_kT", str(RIGIDITY),
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
        "--run-dir", str(run_dir),
        "--dump-frames",
        "--dump-interval", str(dump_interval),
    ]

    step_nm = (2.0 * 1e4 * dt) ** 0.5
    print(f"  dt={dt*1000:.4f} ms  step={step_nm:.3f} nm  "
          f"n_steps={n_steps}  dump_every={dump_interval}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
    """Generate convergence plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bound fraction vs time for a subset of dt values (too many for full legend)
    ax1 = axes[0]
    all_dts = sorted(results.keys())
    # Show ~8 evenly-spaced dt values for readability
    show_indices = np.linspace(0, len(all_dts) - 1, min(8, len(all_dts)), dtype=int)
    for idx in show_indices:
        dt = all_dts[idx]
        times, fracs = results[dt]
        step_nm = (2.0 * 1e4 * dt) ** 0.5
        label = f"dt={dt*1000:.3f} ms (step={step_nm:.2f} nm)"
        ax1.plot(times, fracs, label=label, alpha=0.8, linewidth=1.5)
    # Vertical line at warmup boundary (half-time)
    ax1.axvline(TIME_SEC / 2, color="gray", linestyle=":", alpha=0.5, label="warmup cutoff")
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Fraction of pMHCs bound", fontsize=12)
    ax1.set_title(
        f"Binding kinetics vs step size\n"
        f"({PATCH_SIZE:.0f} nm box, {N_TCR} TCR, {N_PMHC} pMHC, "
        f"σ_r={SIGMA_R} nm, threshold={BIND_THRESHOLD} nm)",
        fontsize=11,
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Right: equilibrium bound fraction vs dt (log scale)
    ax2 = axes[1]
    dts = sorted(results.keys())
    # Average second half of frames (first half is warmup)
    eq_fracs = []
    eq_stds = []
    for dt in dts:
        _, fracs = results[dt]
        half_idx = len(fracs) // 2
        eq_fracs.append(np.mean(fracs[half_idx:]))
        eq_stds.append(np.std(fracs[half_idx:]))

    dt_ms = [d * 1000 for d in dts]
    ax2.errorbar(dt_ms, eq_fracs, yerr=eq_stds, fmt="o-", markersize=8,
                 capsize=5, linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("dt (ms)", fontsize=12)
    ax2.set_ylabel("Equilibrium bound fraction", fontsize=12)
    ax2.set_title("Step-size convergence\n(mean ± std of second half)", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Vertical line at auto-calibrated dt
    ax2.axvline(AUTO_CAL_DT * 1000, color="red", linestyle="--", alpha=0.4,
                label=f"auto-cal ({AUTO_CAL_DT*1000:.3f} ms)")
    # Annotate auto-cal point
    auto_idx = np.argmin([abs(d - AUTO_CAL_DT) for d in dts])
    step_auto = (2.0 * 1e4 * AUTO_CAL_DT) ** 0.5
    ax2.annotate(
        f"auto: step={step_auto:.2f} nm",
        (dt_ms[auto_idx], eq_fracs[auto_idx]),
        textcoords="offset points", xytext=(12, 8),
        fontsize=9, color="red", fontweight="bold",
    )
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = _OUTPUT_DIR / "dt_convergence.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Step-size convergence test")
    print(f"  Box: {PATCH_SIZE} nm, grid {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Molecules: {N_TCR} TCR, {N_PMHC} pMHC, {N_CD45} CD45")
    print(f"  sigma_r={SIGMA_R} nm, bind threshold={BIND_THRESHOLD} nm")
    print(f"  Time: {TIME_SEC}s, dt sweep: {[d*1000 for d in DT_VALUES]} ms\n")

    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for dt in DT_VALUES:
            run_dir = run_sim(tmp_path, dt)
            if run_dir is None:
                continue

            pmhc_pos, tcr_frames = parse_frames(run_dir)
            fracs = compute_bound_fraction(
                pmhc_pos, tcr_frames, BIND_THRESHOLD, PATCH_SIZE
            )
            n_frames = len(fracs)
            times = np.linspace(0, TIME_SEC, n_frames)
            results[dt] = (times, fracs)

            half_idx = n_frames // 2
            eq_mean = np.mean(fracs[half_idx:])
            print(f"  → {n_frames} frames, eq. bound (2nd half) = {eq_mean:.3f}")

    if results:
        make_plot(results)

        # Print summary table
        print("\nSummary:")
        print(f"{'dt (ms)':>10}  {'step (nm)':>10}  {'eq. bound':>10}  {'std':>8}")
        print("-" * 44)
        for dt in sorted(results.keys()):
            _, fracs = results[dt]
            half_idx = len(fracs) // 2
            step = (2.0 * 1e4 * dt) ** 0.5
            print(f"{dt*1000:>10.4f}  {step:>10.3f}  "
                  f"{np.mean(fracs[half_idx:]):>10.3f}  {np.std(fracs[half_idx:]):>8.3f}")
    else:
        print("No successful runs!")


if __name__ == "__main__":
    main()
