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
models/                  # Partial model implementations
  kinetic_segregation/   #   KS model (C + Metal GPU), CPU and GPU modes
    src/                 #     C/Objective-C source code
    tests/               #     pytest test suite
    benchmark/           #     Performance benchmarks
    Methods/             #     LaTeX methods documentation
    Makefile             #     Build system (produces ks_gpu binary)
    __main__.py          #     Python wrapper for framework compatibility
    render_movie.py      #     Animation renderer from binary frame dumps
  lck_activity/          #   Lck radial decay model
  membrane_topography/   #   IRM-based tight-contact geometry
  tcr_phosphorylation/   #   ITAM phosphorylation model
specs/                   # JSON specs (ModelSpec, SurrogateSpec, MetaModelSpec)
data/                    # Experimental reference data
notebooks/               # Jupyter analysis and figure reproduction
artifacts/               # Pre-trained surrogate artifacts
store/                   # Run artifacts (gitignored)
examples/                # Example scripts and specs
conftest.py              # Root conftest for pytest
pytest.ini               # Pytest configuration
```

## Quick Commands

```bash
# Run from projects/tcr_signaling/
pytest -q                                      # Run all model tests
pytest -q models/kinetic_segregation/tests/    # Run KS tests only

# Build KS binary (requires macOS Command Line Tools)
cd models/kinetic_segregation && make

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

## Relationship to Parent Repo

When this repo is used as a submodule of `metamodeler_codex_scaffold_docs`:
- It lives at `projects/tcr_signaling/`
- The parent's pytest can discover submodule tests (optional)
- The parent's `CLAUDE.md` does NOT govern this project
- Changes here should be committed to the submodule's own Git history

When used standalone:
- Install the framework: `pip install bayesian-metamodeling`
- All specs and models work identically
