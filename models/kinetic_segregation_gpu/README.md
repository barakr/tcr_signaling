# GPU-Accelerated Kinetic Segregation Model

C + Metal (Apple Silicon) implementation of the kinetic segregation Monte Carlo
model. Provides 50-500x speedup over the pure Python version.

## Build

Requires macOS with Command Line Tools (Xcode not needed):

```bash
cd models/kinetic_segregation_gpu
make          # builds ./ks_gpu binary
make testlib  # builds shared library for Python ctypes tests
```

## Usage

Identical CLI contract to the Python `kinetic_segregation` model:

```bash
./ks_gpu --time_sec 20 --rigidity_kT_nm2 20 --run-dir /tmp/test

# CPU-only mode (no Metal GPU):
./ks_gpu --time_sec 20 --rigidity_kT_nm2 20 --run-dir /tmp/test --no-gpu
```

Python wrapper (for framework compatibility):

```bash
python -m models.kinetic_segregation_gpu --time_sec 20 --rigidity_kT_nm2 20 --run-dir /tmp/test
```

## Architecture

```
Phase 1 (CPU): Molecular moves (~150 molecules, sequential)
    |
Phase 2 (GPU): Grid height updates (64x64 = 4096 cells)
    - Checkerboard decomposition: red cells, then black cells
    - Each half-sweep: 2048 threads in parallel
    - Pre-filled random buffers from CPU RNG (deterministic)
    |
Repeat for n_steps MC sweeps
```

- **CPU fallback**: When Metal unavailable (CI, SSH, headless), Phase 2 runs
  sequentially in C.
- **float32 on GPU**: Heights are 0-50 nm; float32 precision is sufficient for
  GPU kernels. CPU uses float64 for reference compatibility.
- **Deterministic**: Same seed produces identical output (CPU path).

## Tests

```bash
# From projects/tcr_signaling/
pytest models/kinetic_segregation_gpu/tests/test_potentials.py -v
pytest models/kinetic_segregation_gpu/tests/test_cli.py -v
pytest models/kinetic_segregation_gpu/tests/test_equivalence.py -v -m slow
```

## Benchmarking

```bash
python benchmark/run_benchmark.py      # Run benchmarks
python benchmark/generate_report.py    # Generate report.png
```
