# Kinetic Segregation Model

C implementation of the kinetic segregation Monte Carlo model with optional
Metal GPU acceleration on Apple Silicon.

## Build

Requires macOS with Command Line Tools (Xcode not needed):

```bash
cd models/kinetic_segregation
make          # builds ./ks_gpu binary
make testlib  # builds shared library for Python ctypes tests
```

## Usage

```bash
./ks_gpu --time_sec 20 --rigidity_kT 20 --run-dir /tmp/test

# CPU-only mode (no Metal GPU):
./ks_gpu --time_sec 20 --rigidity_kT 20 --run-dir /tmp/test --no-gpu

# Refuse to run at all if the GPU backend is unavailable, instead of
# falling back to the CPU (which produces different numbers):
./ks_gpu --time_sec 20 --rigidity_kT 20 --run-dir /tmp/test --require-gpu
```

Python wrapper (for framework compatibility):

```bash
python -m models.kinetic_segregation --time_sec 20 --rigidity_kT 20 --run-dir /tmp/test
```

### Which backend ran

The GPU and CPU paths differ in precision and RNG stream, so they give
measurably different results — on a small test case, `depletion_width_nm`
312.67 (Metal) against 283.91 (CPU). Every run therefore records the backend it
used, both on stderr (`Backend: metal`) and in the output JSON:

```json
"diagnostics": { "backend": "metal", ... }
```

The choice depends only on the flags and on whether Metal is available. It does
**not** depend on your working directory: the shader is located relative to the
executable. It was cwd-relative until 2026-08-11, which silently put every
framework-driven sweep on the CPU path — see `Status.md`.

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

- **CPU fallback**: When Metal unavailable (CI, SSH, headless), Phase 2 runs
  sequentially in C.
- **float32 on GPU**: Heights are 0-50 nm; float32 precision is sufficient for
  GPU kernels.
- **Deterministic**: Same seed produces identical output within each mode.

## Tests

```bash
# From projects/tcr_signaling/
pytest models/kinetic_segregation/tests/ -v
```

## Benchmarking

```bash
python benchmark/run_benchmark.py      # Run benchmarks
python benchmark/generate_report.py    # Generate report.png
```
