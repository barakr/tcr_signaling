"""Benchmark: C CPU vs C+Metal GPU at multiple grid sizes."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1]
_BINARY = _PKG_DIR / "ks_gpu"


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

    grid_sizes = [16, 32, 64, 128, 256, 512]
    n_steps = 500
    time_sec = 20.0
    rigidity = 20.0
    seed = 42

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for gs in grid_sizes:
            print(f"\n--- grid_size={gs}, n_steps={n_steps} ---")

            t_c, w_c = run_c(tmp_path, time_sec, rigidity, seed, n_steps, gs, use_gpu=False)
            print(f"  C(CPU):  {t_c:.3f}s  depletion={w_c:.1f}nm")

            # Try GPU
            t_gpu, w_gpu = None, None
            try:
                t_gpu, w_gpu = run_c(tmp_path, time_sec, rigidity, seed, n_steps, gs, use_gpu=True)
                speedup = t_c / t_gpu if t_gpu > 0 else 0
                print(f"  C(GPU):  {t_gpu:.3f}s  depletion={w_gpu:.1f}nm  speedup={speedup:.1f}x")
            except Exception as e:
                print(f"  C(GPU):  failed ({e})")

            results.append({
                "grid_size": gs,
                "n_steps": n_steps,
                "c_cpu_sec": t_c,
                "c_gpu_sec": t_gpu,
                "depletion_c_cpu": w_c,
                "depletion_c_gpu": w_gpu,
            })

    # Save results
    out_path = _PKG_DIR / "benchmark" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
