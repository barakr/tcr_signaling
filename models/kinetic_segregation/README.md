# Kinetic Segregation Model

C99 Monte Carlo implementation of the kinetic segregation model, with an optional
Metal GPU backend on Apple Silicon.

**Runs on Windows, macOS and Linux.** All three are built and tested on every push by
[`KS model CI`](../../.github/workflows/ci.yml) — MSVC, Clang and GCC respectively. The
GPU backend is Apple-only; Windows and Linux compile `src/gpu_stub.c` and run the same
C99 CPU core, which produces the same physics.

## Requirements

A **C/C++ compiler and CMake ≥ 3.20**. Nothing else — this model does not need the
`bayesian-metamodeling` framework.

Per-platform install commands (including the Visual Studio workload Windows users
need) live in one place, so they cannot drift:
**[Build prerequisites](../../README.md#build-prerequisites-windows-macos-linux)**.

If a build fails, `python -c "import ks_build; print(ks_build.toolchain_hint())"` from
this directory prints the same instructions for whichever platform you are on.

## Build

The same two commands everywhere:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

These are exactly the commands CI runs; `tests/test_docs.py` fails if this block and
the workflow ever disagree.

On macOS and Linux, `make` is a shorthand for the same thing (`make`, `make testlib`,
`make pdf`, `make clean`). There is no `make` on a stock Windows box, which is why no
test, notebook or script invokes it.

Products: `ks_gpu` (`ks_gpu.exe` on Windows) in this directory, and
`build/libks_potentials.{dylib,so}` / `build/ks_potentials.dll` for the ctypes tests.
Nothing needs to know those names — `ks_build.py` resolves them.

## Usage

```bash
./ks_gpu --time_sec 20 --rigidity_kT 20 --run-dir /tmp/test          # macOS / Linux
./ks_gpu --time_sec 20 --rigidity_kT 20 --run-dir /tmp/test --no-gpu # force CPU
```

On Windows the binary is `.\ks_gpu.exe`, and `--no-gpu` is redundant because there is
no GPU backend to disable.

Through the Python wrapper, which the `ModelSpec` entrypoint uses (run from the repo
root so the package resolves):

```bash
python -m models.kinetic_segregation --time_sec 20 --rigidity_kT 20 --run-dir /tmp/test
```

Seven flags exist only on the binary and are not reachable through the wrapper:
`--u_assoc`, `--sigma_bind`, `--sigma_r`, `--patch_size`, `--monitor-binding`,
`--monitor-interval`, `--snapshot-interval`.

## Learning the model

`notebooks/models/kinetic_segregation/KS_1` … `KS_5` are a five-notebook course on the
biology, the energy function, running it, what each parameter does, and how to read the
results without fooling yourself. They build this model for you on first run. See
[`notebooks/README.md`](../../notebooks/README.md).

## Architecture

```
Phase 1 (CPU): Molecular moves (~150 molecules, sequential)
    |
Phase 2 (GPU or CPU): Grid height updates (64x64 = 4096 cells)
    - Checkerboard decomposition: red cells, then black cells
    - GPU: Each half-sweep runs in parallel via Metal compute
    - CPU: Sequential fallback when Metal unavailable
    |
Repeat for n_steps MC sweeps
```

- **CPU fallback**: when Metal is unavailable (Windows, Linux, CI, SSH, headless),
  Phase 2 runs sequentially in C.
- **float32 on GPU**: heights are 0–50 nm; float32 is sufficient for the GPU kernels.
- **Deterministic**: the same seed produces identical output within each mode.
- **`src/ks_physics.h` is the single source of truth** for the float-precision energy
  functions; both `simulation.c` and `shaders.metal` include it.
- **`src/ks_compat.h`** holds the platform shims (`M_PI` on MSVC, a monotonic clock).
  Include it instead of `<math.h>` in any file that needs `M_PI`.

## Tests

```bash
python -m pytest -q -m "not slow" models/kinetic_segregation/tests/
```

Run from `projects/tcr_signaling/`. The suite builds the model itself if needed.
`-m "not deterministic"` skips the bit-level baselines, which are platform-specific.

## Benchmarking

```bash
python benchmark/run_benchmark.py      # Run benchmarks
python benchmark/generate_report.py    # Generate report.png
```
