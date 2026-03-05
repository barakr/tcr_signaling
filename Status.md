# Status: TCR Signaling Metamodel Project

## High-level State
- Stage: KS model validated, surrogate fitting in progress
- Current focus: Reproducing partial model sweeps and surrogate training

## Decision Log

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
