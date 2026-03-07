"""Benchmark: Python vs C-only vs C+Metal at multiple grid sizes."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"
_SUBMODULE_ROOT = str(Path(__file__).resolve().parents[3])

# Add submodule root to sys.path for Python model import
sys.path.insert(0, _SUBMODULE_ROOT)


def run_python(time_sec, rigidity, seed, n_steps, grid_size):
    import hashlib
    import struct

    from models.kinetic_segregation.model import simulate_ks

    raw = struct.pack("dd", time_sec, rigidity)
    input_hash = int(hashlib.md5(raw).hexdigest()[:8], 16)
    point_seed = seed + input_hash

    t0 = time.perf_counter()
    result = simulate_ks(
        time_sec=time_sec,
        rigidity_kT_nm2=rigidity,
        seed=point_seed,
        n_steps=n_steps,
        grid_size=grid_size,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, result["depletion_width_nm"]


def run_c(tmp_path, time_sec, rigidity, seed, n_steps, grid_size, use_gpu=False):
    rd = tmp_path / f"c_g{grid_size}_s{seed}_{'gpu' if use_gpu else 'cpu'}"
    cmd = [
        str(_BINARY),
        "--time_sec", str(time_sec),
        "--rigidity_kT_nm2", str(rigidity),
        "--seed", str(seed),
        "--n_steps", str(n_steps),
        "--grid_size", str(grid_size),
        "--run-dir", str(rd),
    ]
    if not use_gpu:
        cmd.append("--no-gpu")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip())
    return elapsed, data["depletion_width_nm"]


def main():
    import tempfile

    grid_sizes = [16, 32, 64]
    n_steps_list = [50]
    time_sec = 20.0
    rigidity = 20.0
    seed = 42

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for gs in grid_sizes:
            for ns in n_steps_list:
                print(f"\n--- grid_size={gs}, n_steps={ns} ---")

                t_py, w_py = run_python(time_sec, rigidity, seed, ns, gs)
                print(f"  Python:  {t_py:.2f}s  depletion={w_py:.1f}nm")

                t_c, w_c = run_c(tmp_path, time_sec, rigidity, seed, ns, gs, use_gpu=False)
                print(f"  C(CPU):  {t_c:.2f}s  depletion={w_c:.1f}nm  speedup={t_py/t_c:.1f}x")

                # Try GPU
                t_gpu, w_gpu = None, None
                try:
                    t_gpu, w_gpu = run_c(tmp_path, time_sec, rigidity, seed, ns, gs, use_gpu=True)
                    print(f"  C(GPU):  {t_gpu:.2f}s  depletion={w_gpu:.1f}nm  speedup={t_py/t_gpu:.1f}x")
                except Exception as e:
                    print(f"  C(GPU):  failed ({e})")

                results.append({
                    "grid_size": gs,
                    "n_steps": ns,
                    "python_sec": t_py,
                    "c_cpu_sec": t_c,
                    "c_gpu_sec": t_gpu,
                    "depletion_python": w_py,
                    "depletion_c_cpu": w_c,
                    "depletion_c_gpu": w_gpu,
                })

    # Save results
    out_path = _PKG_DIR / "benchmark" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
