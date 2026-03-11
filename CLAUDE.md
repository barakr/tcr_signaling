# CLAUDE.md — TCR Signaling Metamodel Project

## Project Overview

This is an **independent research project** that reproduces and extends
"Bayesian metamodeling of early T-cell antigen receptor signaling accounts for
its nanoscale activation patterns" (Neve-Oz, Sherman & Raveh, *Frontiers in
Immunology*, 2024).

It uses the `bayesian-metamodeling` framework as a dependency — either installed
via pip/conda or available as the parent repo when used as a Git submodule.

**This is NOT part of the framework.** It is a consumer/use-case project with
its own development lifecycle, status tracking, and conventions.

- **Repo**: https://github.com/barakr/tcr_signaling
- **Framework dependency**: `bayesian-metamodeling` (CLI: `bayesmm`)
- **Python**: >= 3.12

## Project Structure

```
models/
  kinetic_segregation/         # KS model — C99 core + C++20 CLI + Metal GPU
    src/
      simulation.c/h           #   C99 core: Phase 1 (molecules) + Phase 2 (grid MC)
      potentials.c/h           #   Energy functions (double precision, CPU)
      rng.c/h                  #   PCG64 pseudo-random number generator
      ks_physics.h             #   Shared float physics (CPU + GPU single source of truth)
      gpu_engine.h             #   Backend-neutral GPU C API (4 functions, opaque handle)
      metal_engine.m           #   Metal backend (macOS / Apple Silicon)
      gpu_stub.c               #   Stub backend (non-Apple → CPU fallback)
      shaders.metal            #   GPU compute kernels (Philox RNG, checkerboard)
      main.cpp                 #   C++20 CLI (nlohmann/json, FNV-1a seeds)
    tests/                     #   72 pytest tests (regression + physics + equivalence)
    benchmark/                 #   Performance benchmarks (CPU vs GPU)
    Methods/                   #   LaTeX methods documentation → methods.pdf
    CMakeLists.txt             #   Cross-platform build (CMake >= 3.20)
    Makefile                   #   Thin CMake wrapper (make / make pdf / make clean)
    __main__.py                #   Python wrapper for framework compatibility
    render_movie.py            #   Animation renderer (binary frame dumps → MP4)
  lck_activity/                # Lck radial decay model
  membrane_topography/         # IRM-based tight-contact geometry
  tcr_phosphorylation/         # ITAM phosphorylation model
specs/                         # JSON specs (ModelSpec, SurrogateSpec, MetaModelSpec)
data/                          # Experimental reference data
notebooks/                     # Jupyter analysis and figure reproduction
artifacts/                     # Pre-trained surrogate artifacts
store/                         # Run artifacts (gitignored)
examples/                      # Example scripts and specs
conftest.py                    # Root conftest for pytest
pytest.ini                     # Pytest configuration
```

## Quick Commands

```bash
# Run from projects/tcr_signaling/
pytest -q                                      # Run all model tests
pytest -q models/kinetic_segregation/tests/    # Run KS tests only
pytest -q -m "not deterministic"               # Skip bit-level tests (new platform)

# Build KS binary (CMake, cross-platform)
cd models/kinetic_segregation
make                   # Build ks_gpu binary + shared testlib
make clean && make     # Full rebuild
make pdf               # Compile Methods/methods.pdf (needs tectonic in PATH)
make testlib           # Build shared lib for ctypes tests only

# Use the framework CLI on project specs
bayesmm validate specs/model.kinetic_segregation.json
bayesmm run specs/model.kinetic_segregation.json
bayesmm surrogate fit specs/surrogate.kinetic_segregation.pymc_gp.json
```

## Key Documents

| File | Purpose |
|------|---------|
| `README.md` | Project overview, partial models, reproduction workflow |
| `Status.md` | Project-specific progress and decision log |
| `CLAUDE.md` | This file — AI development guide for the TCR project |

## Partial Models

1. **Membrane Topography** — IRM-derived tight-contact geometry
2. **Kinetic Segregation (KS)** — C implementation with CPU and Metal GPU modes;
   spatial exclusion of CD45 from tight contacts
3. **Lck Activity** — Radially-symmetric exponential decay of active Lck
4. **TCR Phosphorylation** — Lck* phosphorylates TCR ITAMs

## Development Rules

1. **This project has its own Status.md** — update it here, not in the parent repo
2. **Tests live with models** — each model subdirectory has a `tests/` folder
3. **Framework is a dependency** — do not modify `src/bayesian_metamodeling/` from here;
   if the framework needs changes, switch to the parent repo context
4. **Specs are self-contained** — each JSON spec should work with `bayesmm` CLI
   without needing to know about the repo structure
5. **Seeds and provenance** — all model runs must be reproducible; record seeds
   in specs and decision log
6. **Commit regularly** — make git commits at logical milestones (fix verified, feature
   complete, refactor done). Do not accumulate large uncommitted changesets
7. **Test change policy** — Existing tests should not be modified to accommodate
   code changes unless there is a well-defined logical reason (e.g., the feature
   being tested was intentionally redesigned, or the test's reference
   implementation was removed). Any such change must be:
   - Documented in the plan with explicit rationale
   - Approved by the user before implementation
   - Prefer to ask for user input before modifying existing test assertions

## KS Model Rules

These rules apply specifically to the kinetic segregation model
(`models/kinetic_segregation/`):

1. **`ks_physics.h` is the single source of truth** for float-precision energy
   functions (TCR binding, CD45 repulsion, Laplacian, bending delta). Both
   `simulation.c` (CPU) and `shaders.metal` (GPU) include it. Never duplicate
   energy logic — update the header.

2. **GPU backends must implement `gpu_engine.h`** — the 4-function C API
   (`gpu_engine_create`, `gpu_engine_destroy`, `gpu_engine_h_ptr`,
   `gpu_engine_grid_update`) with an opaque `void*` handle.

3. **Reference value protection** — `tests/reference_values.json` holds exact
   numerical baselines for GPU and CPU. If tests fail after a code change,
   **do not silently re-record**. Report the failure and diff to the user for
   approval. Re-recording is only allowed after explicit sign-off that the
   output change is intentional (e.g., physics formula update, seed derivation
   change). Bit-level determinism tests are marked `@pytest.mark.deterministic`
   and can be skipped on a new platform with `-m "not deterministic"` while
   regression + statistical tests validate the physics.

4. **Statistical regression tests** — In addition to exact-value baselines,
   the test suite includes statistical regression tests that run multiple
   seeds and verify key observables (depletion width, acceptance rate) remain
   within expected margins. These catch subtle bugs that shift distributions
   without breaking a single reference seed. Coverage includes default
   Brownian mode, pMHC gating modes, binding modes, and step modes.

5. **Methods documentation** — Keep `Methods/methods.tex` in sync with the
   code. Don't update mid-refactor — update once changes stabilize, before
   major commits or pushes. Recompile with `make pdf`.

6. **Build with CMake** via the `make` wrapper. The `CMakeLists.txt` handles
   Apple (Metal) vs Linux (stub → CPU fallback) automatically.

7. **Conda environments** — use the same environments as the parent framework:

   | Environment | Purpose |
   |-------------|---------|
   | `py314_bayesmm` | Main dev: building, testing, rendering (Python 3.14, tectonic) |
   | `py312_bayesmm_pymc` | PyMC surrogate fitting (PyMC 5.27.1 + ArviZ) |
   | `py312_bayesmm_sbi` | SBI surrogate fitting (SBI 0.23.3 + Torch 2.10.0) |

   Always activate the appropriate env before running commands. Default for
   KS model work is `py314_bayesmm`.

8. **Frame dump format** — Binary frame dumps follow a fixed contract:
   `h_XXXXX.bin` (float32 height field), `mol_XXXXX.bin` (float64 molecule
   positions), `pmhc.bin` (float64, static pMHC positions). Don't change the
   format without updating `render_movie.py`.

9. **All KS tests must pass** before committing changes to the model.

## Relationship to Parent Repo

When this repo is used as a submodule of `metamodeler_codex_scaffold_docs`:
- It lives at `projects/tcr_signaling/`
- The parent's pytest can discover submodule tests (optional)
- The parent's `CLAUDE.md` does NOT govern this project
- Changes here should be committed to the submodule's own Git history

When used standalone:
- Install the framework: `pip install bayesian-metamodeling`
- All specs and models work identically


## Conda Environments during development

| Environment | Purpose |
|-------------|---------|
| `py314_bayesmm` | Main dev (Python 3.14) |
| `py312_bayesmm_pymc` | PyMC 5.27.1 + ArviZ |
| `py312_bayesmm_sbi` | SBI 0.23.3 + Torch 2.10.0 |

