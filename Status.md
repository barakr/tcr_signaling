# Status: TCR Signaling Metamodel Project

## High-level State
- Stage: KS model validated, surrogate fitting in progress
- Current focus: Reproducing partial model sweeps and surrogate training

## Decision Log

### 2026-03-06: KS simulation accuracy and numerical stability fixes
- **P0: time_sec was ignored when n_steps explicit** — the root cause of
  non-monotonic depletion width. All DOE points ran identical step counts
  regardless of time. Fixed: `n_steps` is now a base count at `TIME_REF_SEC=20s`,
  scaled linearly with `time_sec`. E.g. n_steps=20, t=100s → 100 actual sweeps.
- **P1: Molecular step size too large** — was `dx * 0.5` (31-63nm), far larger
  than the TCR binding well (sigma=3nm). Changed to `max(sigma_bind*2, dx*0.15)`,
  giving 6-19nm depending on grid resolution.
- **P1: Height step size not adapted to rigidity** — fixed 1nm step regardless of
  kappa. Now `min(5.0, dx / (4*sqrt(kappa)))`: stiffer membranes get smaller steps.
- **P2: Periodic boundary conditions** — replaced zero-padded Laplacian with
  periodic BCs (`np.roll`). Eliminates artificial edge effects where boundary cells
  had zero bending penalty. `bending_energy_delta` updated to match.
- **P2: Depletion metric improved** — changed from noisy 75th/25th percentile gap
  to median separation (more robust with 10-30 molecules). Measurement taken from
  the final configuration (not averaged — this is a dynamics simulation, not
  equilibrium sampling).
- Added two new tests: `test_explicit_n_steps_scales_with_time` (deterministic)
  and `test_depletion_increases_with_time` (statistical, kappa=30, 10 seeds).
- All 40 KS tests pass; 279 root framework tests pass.
- Remaining limitations (future work): no excluded-volume between molecules,
  linear surrogate (pymc_gp) has high RMSE on this nonlinear surface.

### 2026-03-05: Split examples into Python API and CLI shell script
- Replaced hybrid `ks_sweep_and_surrogate.py` (mixed CLI subprocess + Python API)
  with two clean, independent examples:
  - `ks_example.py` — pure Python API, calls `simulate_ks()` directly, no `bayesmm`
    dependency. Produces CSV + heatmap PNG in `artifacts/`.
  - `ks_example_cli.sh` — pure shell script using `bayesmm validate`, `bayesmm run`,
    then delegates to `plot_sweep.py` for the heatmap.
- Extracted shared plotting into `plot_sweep.py` (load CSV, pivot, heatmap PNG).
  Used by both examples.
- Python API example runs fast profile in ~0.6s (vs ~2.3s for CLI with subprocess overhead).
- Surrogate steps are optional (`--with-surrogate` flag in Python example), gracefully
  skipped when PyMC/SBI are unavailable.

### 2026-03-05: Local delta bending energy optimization (O(N⁴) → O(N²))
- Phase 2 grid sweep previously called `bending_energy()` twice per cell per MC
  step, recomputing the full Laplacian over the entire grid — O(grid_size⁴) total.
- Added `bending_energy_delta()` that computes ΔE locally by only evaluating the
  5 affected Laplacian cells when a single height changes — O(1) per cell update.
- Total Phase 2 cost reduced from O(grid_size⁴) to O(grid_size²) per step.
- At grid_size=64: theoretical 4096× speedup for the bending energy computation.
- Benchmark: grid_size=64, 2 steps completes in 0.21s (previously dominated by
  full Laplacian recomputation).
- Correctness verified: delta function matches full recomputation within
  floating-point tolerance across interior, boundary, edge, and corner cells.
- All 38 KS tests pass.

### 2026-03-05: Add tiered example specs (fast/regular/extensive)
- Benchmarked KS model: grid_size dominates runtime (O(grid_size^4) per step
  due to full bending energy recomputation).
- Current default spec (grid=64, 56 DOE points, auto-steps) takes ~135 min.
- Created three tiered specs in `examples/specs/`:
  - **fast**: grid=16, 10 steps, 30 molecules, 9 DOE points (~10 sec)
  - **regular**: grid=32, 20 steps, 90 molecules, 16 DOE points (~1-3 min)
  - **extensive**: grid=64, auto steps, 150 molecules, 56 DOE points (~20-30 min)
- Example script now accepts `--profile fast|regular|extensive` (default: fast).
- Model simulation params (grid_size, n_tcr, n_cd45, n_steps) passed through
  adapter as single-value DOE grid entries — no framework changes needed.
- Production spec in `specs/` left unchanged for full reproduction.

### 2026-03-05: Fix KS model MC loop + self-contained example specs
- **MC loop fix**: Changed from single-particle stepping to full sweeps — each
  `n_steps` iteration now updates every molecule and every grid cell once,
  matching standard MC convention. Previous behavior updated ~3 molecules per
  step (150 molecules / 500 steps), far too few for equilibration.
- **n_steps auto-scaling**: Changed from `max(500, time*100)` to `max(50, time*5)`
  since each step now does ~1000x more work (full sweep vs single particle).
- **Default grid_size**: Increased from 32 to 64 for better spatial resolution.
- **Per-point seed derivation**: `__main__.py` now derives a unique reproducible
  seed per DOE point via `seed + hash(inputs)`, eliminating correlated MC noise
  across the parameter sweep.
- **Self-contained example specs**: Created `examples/specs/` with model, pymc_gp,
  and sbi_npe specs. Example script now references local specs instead of
  global `specs/` directory.

### 2026-03-05: Co-locate model tests within each model subdirectory
- Moved tests into `models/kinetic_segregation/tests/` split into
  `test_potentials.py`, `test_model.py`, `test_cli.py`
- Submodule has its own `conftest.py` and `pytest.ini`
- All 54 submodule tests pass independently

### 2026-03-05: Comprehensive KS test suite
- Added tests covering potentials (LJ, harmonic, Morse), spatial exclusion,
  CLI interface, and parameter sensitivity
- Tests validate physical correctness: CD45 exclusion from tight contacts,
  energy conservation, boundary conditions

### 2026-03-05: Initial KS model implementation
- Implemented kinetic segregation with Monte Carlo spatial simulation
- CD45 molecules excluded from tight-contact zones by repulsive potential
- Grid-based spatial tracking with configurable resolution
