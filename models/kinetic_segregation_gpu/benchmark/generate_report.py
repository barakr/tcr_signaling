"""Generate benchmark comparison plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCHMARK_DIR = Path(__file__).resolve().parent


def main():
    results_path = _BENCHMARK_DIR / "results.json"
    if not results_path.exists():
        print("No results.json found. Run run_benchmark.py first.")
        return

    results = json.loads(results_path.read_text())

    grid_sizes = [r["grid_size"] for r in results]
    py_times = [r["python_sec"] for r in results]
    c_cpu_times = [r["c_cpu_sec"] for r in results]
    c_gpu_times = [r["c_gpu_sec"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Speedup bar chart
    ax = axes[0, 0]
    x = np.arange(len(grid_sizes))
    width = 0.25
    cpu_speedup = [p / c for p, c in zip(py_times, c_cpu_times)]
    ax.bar(x - width / 2, cpu_speedup, width, label="C (CPU)", color="steelblue")
    if any(t is not None for t in c_gpu_times):
        gpu_speedup = [p / g if g else 0 for p, g in zip(py_times, c_gpu_times)]
        ax.bar(x + width / 2, gpu_speedup, width, label="C+Metal (GPU)", color="coral")
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Speedup vs Python")
    ax.set_title("Speedup over Python")
    ax.set_xticks(x)
    ax.set_xticklabels(grid_sizes)
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # Panel 2: Wall time scaling (log-log)
    ax = axes[0, 1]
    ax.loglog(grid_sizes, py_times, "o-", label="Python", color="green")
    ax.loglog(grid_sizes, c_cpu_times, "s-", label="C (CPU)", color="steelblue")
    if any(t is not None for t in c_gpu_times):
        valid = [(g, t) for g, t in zip(grid_sizes, c_gpu_times) if t is not None]
        if valid:
            ax.loglog(*zip(*valid), "^-", label="C+Metal (GPU)", color="coral")
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Scaling: wall time vs grid size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Correctness scatter
    ax = axes[1, 0]
    py_dep = [r["depletion_python"] for r in results]
    c_dep = [r["depletion_c_cpu"] for r in results]
    ax.scatter(py_dep, c_dep, color="steelblue", s=60, label="C (CPU)")
    if any(r["depletion_c_gpu"] is not None for r in results):
        gpu_dep = [r["depletion_c_gpu"] for r in results if r["depletion_c_gpu"] is not None]
        py_dep_gpu = [r["depletion_python"] for r in results if r["depletion_c_gpu"] is not None]
        ax.scatter(py_dep_gpu, gpu_dep, color="coral", s=60, marker="^", label="C+Metal (GPU)")
    lims = [0, max(max(py_dep), max(c_dep)) * 1.1 + 1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Python depletion width (nm)")
    ax.set_ylabel("C depletion width (nm)")
    ax.set_title("Correctness: depletion width")
    ax.legend()

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for r in results:
        cpu_su = r["python_sec"] / r["c_cpu_sec"]
        gpu_su = r["python_sec"] / r["c_gpu_sec"] if r["c_gpu_sec"] else "N/A"
        if isinstance(gpu_su, float):
            gpu_su = f"{gpu_su:.1f}x"
        table_data.append([
            str(r["grid_size"]),
            f"{r['python_sec']:.2f}s",
            f"{r['c_cpu_sec']:.2f}s",
            f"{r['c_gpu_sec']:.2f}s" if r["c_gpu_sec"] else "N/A",
            f"{cpu_su:.1f}x",
            gpu_su,
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Grid", "Python", "C(CPU)", "C(GPU)", "CPU speedup", "GPU speedup"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Summary", pad=20)

    plt.tight_layout()
    out_path = _BENCHMARK_DIR / "report.png"
    plt.savefig(out_path, dpi=150)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
