"""Benchmark: CPU vs GPU across all configuration combinations.

Tests the 4-config matrix: {no repulsion, repulsion} x {gaussian, forced} binding
at multiple grid sizes and step counts.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"

# Configuration matrix.
CONFIGS = {
    "no_repul_gaussian": [],
    "no_repul_forced": [
        "--binding_mode", "forced", "--n_pmhc", "125", "--pmhc_radius", "333",
    ],
    "repul_gaussian": [
        "--mol_repulsion_eps", "2.0", "--mol_repulsion_rcut", "50.0",
    ],
    "repul_forced": [
        "--mol_repulsion_eps", "2.0", "--mol_repulsion_rcut", "50.0",
        "--binding_mode", "forced", "--n_pmhc", "125", "--pmhc_radius", "333",
    ],
}


def run(tmp_path, *, grid_size, n_steps, config_name, use_gpu):
    rd = tmp_path / f"{config_name}_g{grid_size}_s{n_steps}_{'gpu' if use_gpu else 'cpu'}"
    cmd = [
        str(_BINARY),
        "--time_sec", "200",
        "--rigidity_kT_nm2", "20",
        "--seed", "42",
        "--n_steps", str(n_steps),
        "--grid_size", str(grid_size),
        "--n_tcr", "125",
        "--n_cd45", "250",
        "--run-dir", str(rd),
    ] + CONFIGS[config_name]
    if not use_gpu:
        cmd.append("--no-gpu")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, f"{config_name} failed: {result.stderr}"
    data = json.loads(result.stdout.strip())
    return elapsed, data


def main():
    import tempfile

    grid_sizes = [64, 128]
    step_counts = [1000, 10000]
    seed = 42

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for gs in grid_sizes:
            for ns in step_counts:
                for cfg in CONFIGS:
                    print(f"\n--- {cfg}  grid={gs}  steps={ns} ---")
                    row = {
                        "config": cfg, "grid_size": gs, "n_steps": ns,
                    }

                    t_cpu, d_cpu = run(tmp_path, grid_size=gs, n_steps=ns,
                                       config_name=cfg, use_gpu=False)
                    row["cpu_sec"] = round(t_cpu, 3)
                    row["depletion_cpu"] = round(d_cpu["depletion_width_nm"], 1)
                    row["accept_rate_cpu"] = round(d_cpu["diagnostics"]["accept_rate"], 4)
                    print(f"  CPU: {t_cpu:.3f}s  depl={row['depletion_cpu']}nm")

                    try:
                        t_gpu, d_gpu = run(tmp_path, grid_size=gs, n_steps=ns,
                                           config_name=cfg, use_gpu=True)
                        row["gpu_sec"] = round(t_gpu, 3)
                        row["depletion_gpu"] = round(d_gpu["depletion_width_nm"], 1)
                        row["accept_rate_gpu"] = round(d_gpu["diagnostics"]["accept_rate"], 4)
                        speedup = t_cpu / t_gpu if t_gpu > 0 else 0
                        row["gpu_speedup"] = round(speedup, 2)
                        print(f"  GPU: {t_gpu:.3f}s  depl={row['depletion_gpu']}nm  "
                              f"speedup={speedup:.1f}x")
                    except Exception as e:
                        row["gpu_sec"] = None
                        row["gpu_speedup"] = None
                        print(f"  GPU: failed ({e})")

                    results.append(row)

    # Summary table.
    print("\n\n=== SUMMARY ===")
    print(f"{'Config':<22} {'Grid':>4} {'Steps':>6} {'CPU(s)':>8} {'GPU(s)':>8} {'Speedup':>8}")
    print("-" * 60)
    for r in results:
        gpu_s = f"{r['gpu_sec']:.3f}" if r.get("gpu_sec") else "N/A"
        sp = f"{r['gpu_speedup']:.1f}x" if r.get("gpu_speedup") else "N/A"
        print(f"{r['config']:<22} {r['grid_size']:>4} {r['n_steps']:>6} "
              f"{r['cpu_sec']:>8.3f} {gpu_s:>8} {sp:>8}")

    # Save results.
    out_path = _PKG_DIR / "benchmark" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
