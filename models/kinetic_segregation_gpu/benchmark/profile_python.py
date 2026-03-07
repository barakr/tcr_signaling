"""Profile the Python KS model with per-phase timing."""
from __future__ import annotations

import hashlib
import struct
import sys
import time
from pathlib import Path

import numpy as np

_SUBMODULE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _SUBMODULE_ROOT)

from models.kinetic_segregation.model import simulate_ks  # noqa: E402


def profile_run(grid_size=64, n_steps=10, time_sec=20.0, rigidity=20.0, seed=42):
    raw = struct.pack("dd", time_sec, rigidity)
    input_hash = int(hashlib.md5(raw).hexdigest()[:8], 16)
    point_seed = seed + input_hash

    print(f"Profiling: grid_size={grid_size}, n_steps={n_steps}")
    t0 = time.perf_counter()
    result = simulate_ks(
        time_sec=time_sec,
        rigidity_kT_nm2=rigidity,
        seed=point_seed,
        n_steps=n_steps,
        grid_size=grid_size,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Total: {elapsed:.2f}s")
    print(f"  Depletion: {result['depletion_width_nm']:.1f}nm")
    print(f"  Accept rate: {result['accept_rate']:.3f}")
    print(f"  Per step: {elapsed/n_steps*1000:.1f}ms")
    return elapsed


def main():
    for gs in [16, 32, 64]:
        profile_run(grid_size=gs, n_steps=10)
        print()


if __name__ == "__main__":
    main()
