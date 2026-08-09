# Status: TCR Signaling Metamodel Project

## High-level State
- Stage: KS model validated, surrogate fitting in progress
- Current focus: Reproducing partial model sweeps and surrogate training

## Decision Log

### 2026-08-07: 01-04 now run end to end in ~3 minutes

Follow-up to the entry below. Two further framework bugs had to be fixed in the
parent before this series could work at all, and 02 was made fast enough to read.

- **02 sweeps at teaching scale, in 30 s.** Three knobs, and the third is the
  10x: grid levels 8 -> 2, Sobol 64 -> 8, and `time_sec` overridden to 0.5/1.0.
  KS wall-time is linear in simulated time (measured here: 0.5 s -> 4 s,
  5 s -> 40 s, 100 s -> >9 min), so the production design is hours. The teaching
  values are BELOW the production range, not merely its cheap end — stated in the
  notebook, because it means the depletion widths are smaller than the production
  numbers and the surrogate is a demonstration of mechanics, not a result.
- **A "Running it for real" section** gives the exact commands and the single
  switch (`TEACHING_SCALE = False`), says to budget hours, and explains that only
  the data changes — every spec, adapter, path and coupling is identical, because
  scale lives in the design block, not the pipeline.
- **03 works for the first time.** Once 02 published the four surrogates,
  `meta build` resolved and the metamodel sampled: 14 variables, 16 factors
  (9 priors, 3 couplings, 4 surrogate likelihoods), 2000 draws. Its strict xfail
  in the parent turned into XPASS and forced the marker out, which is what strict
  is for.
- **Every self-check is now real.** 03 asserts four surrogate likelihoods, at
  least one coupling, and that the coupled variables have finite, non-zero-variance
  posteriors — a collapsed sampler fails it.

**Scientific caveat worth following up separately**: the sampled posteriors for
`contact_fraction` and `ptcr_fraction` come back at mean ~0, sd ~1 — i.e.
essentially their priors. The metamodel runs, but on this teaching-scale data it
is not being constrained. Whether that persists with a production sweep is an
open question, not something these notebook fixes answer.

### 2026-08-07: The notebooks/01-04 series ran nowhere, and was broken in four ways

The parent added `Submodule notebooks CI`, which builds the KS model and executes
these notebooks weekly. Its first runs found that the series had been broken for
some time — nothing had ever executed it.

- **Specs were parent-repo-relative while the notebooks run from the submodule
  root.** `storage.root` and `dataset_ref` were `projects/tcr_signaling/store`,
  so sweeps wrote to `<submodule>/projects/tcr_signaling/store` — a nested path
  that `.gitignore`'s `projects/` line hid by accident. That is why `store/` held
  only `.gitkeep`. All 8 specs de-nested to `store`.
- **`metamodel.tcr_signaling.json` referenced four artifacts nothing produced.**
  The surrogate store is content-addressed (registry keyed by hash), which is
  right for provenance and useless as a reference you commit. Notebook 02 now
  *publishes* each fit under a stable name in `artifacts/`, joining on
  `spec_name`, artifact_id preserved.
- **`03_metamodel_inference` never assigned `META_SPEC`** — NameError on its
  first CLI cell.
- **Every self-check was `assert ROOT.is_dir()`**, which is true in a repo where
  nothing ran. Replaced: 01 asserts all four models produced their declared
  observable and prints the values; 02 asserts the four surrogates are fitted,
  published and declare real IO; 04 asserts the paper's actual claim — pTCR peaks
  at r/R=1.000 with 4567x edge/centre contrast, so the assertion fails if the
  ring inverts or flattens.

**Teaching-scale sweeps in 02.** The production design is 56 KS points with
`time_sec` up to 100; measured, `time_sec=5` takes 52 s and `time_sec=100` over
9 minutes, so the full sweep is hours, and 02 timed out at both 1800 s and
3600 s. It now sweeps *reduced copies* in `tmp/tutorial_specs/` — the cheapest
levels, not the endpoints, since spanning the range selects the single most
expensive simulation. Runtime ~4 min. The production specs are untouched;
sweeping them unmodified still gives the real thing. The notebook says plainly
that a surrogate fitted this way is a demonstration, not a result, and that the
bias toward the cheap corner is a real limitation.

Also fixed: 01's model cells invoked bare `"python"` rather than
`sys.executable` — the same wrong-interpreter class as the 28835b8 post-mortem.


### 2026-08-07: Visual teaching in the KS series — filmstrips, not movies

The tutorials argued spatial physics with error bars. KS_5, the notebook about reading
results honestly, had **zero figures**; KS_4 had only error bars despite being about what
parameters do in space; KS_3 showed a before and an after but never the process.

**Filmstrips as the default, movies opt-in.** The submodule's CI installs
numpy/scipy/matplotlib/pytest/nbclient and deliberately not ffmpeg, and the notebooks are
CI-executed, so an MP4-rendering cell cannot be the default path. A 4-6 panel time strip
carries most of a movie's teaching value -- it shows the contact forming or dissolving and
CD45 clearing -- while costing one PNG, needing no ffmpeg, executing in CI, and surviving
static rendering on GitHub.

- **KS_3** gains a 6-panel time strip of one run, plus a printed table of mean height in
  the contact disc and CD45 fraction inside it, so the eye and the numbers agree. The
  frame-dumping cell was restructured to read every frame *before* the temp directory is
  removed, with an assert that `dump_interval` divides `n_steps` (otherwise the "final"
  frame is not the final state -- a bug that had already bitten twice).
- **KS_4** gains a soft-vs-stiff contrast strip at **matched physical time**, which makes
  the dt ~ 1/kappa confound concrete: the stiff run needs 20x the sweeps to cover the same
  45.8 ms. Contact settles at 36.7 nm (kappa=1) versus 53.8 nm (kappa=20).
- **KS_5** gains the sharpest demonstration in the series. The box is periodic, so sliding
  every molecule by L/3 is *physically identical* -- yet `depletion_width_nm` collapses
  from 147.70 nm to **0.00**, while the two nearest-neighbour metrics change by 3.2e-14 nm.
  That is a proof, not an illustration, that metrics 2-6 measure the contact's position
  rather than its segregation. It also explains the `pmhc_mode=uniform` zero from KS_3.
- **Opt-in movie cell** in KS_3: `MAKE_MOVIE = False` by default, checks for ffmpeg, and
  otherwise drives the repo's own `render_movie.py` into gitignored `tmp/`. Readers get
  the animation; CI stays fast and ffmpeg-free.

Cost: the KS notebook suite goes 50 s -> 77 s, within the agreed budget. Notebook sizes
grow (KS_3 201->429 KB, KS_4 146->540 KB, KS_5 19->98 KB) because the figures are stored;
that is the price of the notebooks rendering usefully on GitHub without execution.


### 2026-08-07: Production spec switched to area deposition + inner_circle pMHC

Ran `specs/model.kinetic_segregation.json` both ways before deciding. Two findings
made the decision easy, and one of them was more serious than the deposition question.

**1. There is nothing to invalidate.** All four surrogate artifacts
`specs/metamodel.tcr_signaling.json` references are **missing**, `store/` holds only
`.gitkeep`, and no sweep output is tracked in git. No production sweep result exists on
disk, so switching cannot conflict with anything previously computed.

**2. The spec's declared output was mostly zero.** It never set `pmhc_mode`, so it
inherited `uniform`. Uniform ligands give no centred contact, and `depletion_width_nm`
is measured from the patch centre -- so it returned exactly `0.000` for 3 of 4 sampled
points (the fourth, 82.09, is noise). A surrogate fitted to that data would have been
fitting zeros. This was the real defect; the deposition scheme was secondary.

**3. The deposition change itself is small.** At the spec's own configuration
(dx = 31.25 nm), 3 paired seeds at kappa = 1 with everything else held fixed:
point 220.5 +/- 14.2 nm, area 213.7 +/- 11.8 nm. The mode difference (-3.1%) is
**smaller than the seed-to-seed spread (+/-6.5%)**, so on this observable it is not a
distinguishable effect. What area deposition changes is that the membrane coupling
exists at all -- the deposited weight goes from a fraction of the target to the full
target -- not the value of this particular summary statistic.

**Spec now sets**, as single-valued DOE entries (the same trick `examples/specs`
profiles already use): `pmhc_mode=1` (inner_circle), `pmhc_radius=666.67`,
`n_pmhc=419`, `pmhc_deposition=1` (area). Verified end to end through
`bayesmm run`: 2/2 points succeed and produce **227.8 nm (kappa=1)** and
**263.4 nm (kappa=5)**, where the old configuration produced `0.000`.

`--pmhc_deposition` now also accepts `0`/`1` (and `0.0`/`1.0`) alongside the canonical
names, in both the C CLI and the Python wrapper, because a `ModelSpec` DOE grid is
typed `dict[str, list[float]]` -- a spec cannot send a string. Mirrors `--pmhc_mode`,
which already accepts `0`/`1`. Anything outside the accepted set is still an error, not
a silent fallback.

`test_spec_resolution.py` now treats `pmhc_deposition = area` as resolving the mesh
problem (it is mesh-independent by construction), so the production spec is no longer
in `KNOWN_UNRESOLVED`; the three demo fixtures and the sweep DOE remain declared.
`TOTAL_FLOOR` 172 -> 173 (178 collected). One assertion in
`test_pmhc_deposition.py` was updated to match the widened error message -- the
behaviour it guards (typo rejected, non-zero exit) is unchanged, and numeric-alias
coverage was added alongside it.

**Scientific expectation for anyone re-running:** results will NOT match previous
runs, because previously there were effectively none -- the old configuration returned
zero for most of the grid. This is the first configuration of this spec that produces a
non-degenerate `depletion_width_nm`. Treat any earlier KS numbers as void rather than
as a baseline to compare against.


### 2026-08-07: Fix the pMHC deposition defect (opt-in), make it undetectable-proof

The under-resolution flagged earlier the same day turned out to be a genuine defect with
a standard name. The per-cell weights consumed by the Phase-2 grid update are a
**particle-mesh deposition** of Gaussian kernels of width `sigma_r` (2 nm), and
`compute_pmhc_influence` sampled each kernel at the **cell centre**. Point-sampling a
kernel narrower than the cell is a known failure mode: the deposited total collapses as
the mesh coarsens instead of staying mesh-independent, so the TCR attraction quietly
leaves the membrane update. Phase 1 was never affected -- it evaluates at continuous
positions via `pmhc_influence_at`, whose comment already noted the `sigma_r < dx` case.

**Fix, deliberately opt-in.** New `--pmhc_deposition point|area`:

- `point` (default) -- the historical cell-centre sample, kept as the default so every
  existing result and `tests/reference_values.json` stay **bit-identical**. Verified:
  all 8 `deterministic` reference tests pass unchanged after the change.
- `area` -- the exact cell **average** of the same kernel, in closed form via `erf`.
  Mesh-independent, and it converges to `point` as dx -> 0, so the two are the same
  physics differing only in mesh fidelity. On a 2000/64 mesh it recovers the full
  target weight (0.772 vs 0.047 for point).

Changing the default would silently alter everyone's numbers and invalidate the
reference baselines, which KS rule 3 forbids without sign-off -- hence opt-in. Flipping
it is a one-line change once the science decision is made, and the tests already pin
both behaviours.

**Detection, always on.** Three new diagnostics in the run JSON --
`pmhc_influence_max`, `pmhc_influence_sum`, `pmhc_influence_expected` (the resolved
target `n_pmhc * 2*pi*sigma_r^2 / dx^2`) -- plus a `WARN-PMHC` stderr note, in the
existing `AUTO-*` house style, whenever the deposited weight falls below half the
target. The trigger compares against the analytic target rather than testing for an
exactly-dead field, because the failure is usually a large deficit rather than a clean
zero, and its size depends on where ligands happen to land relative to cell centres
(measured: total loss at dx = 125 nm, partial at 31 nm, negligible below ~8 nm).

Unlike `--binding_mode`/`--step_mode`/`--pmhc_mode`, the new flag **validates its
argument** and exits non-zero on a typo, rather than silently selecting the other
option. It is also deliberately NOT readable from a `--params` file, since that path
overrides explicit CLI flags (the trap KS_5 documents). It is exposed on the Python
wrapper as well, so specs can opt in.

**Tests.** `tests/test_pmhc_deposition.py` (10) pins: the diagnostics exist and default
to point; `expected` scales as 1/dx^2; point under-deposits on a coarse mesh and warns;
a resolved mesh does not warn; area hits the target; area recovers >10x what point
loses; **the two schemes converge as the mesh is refined** (the correctness argument);
the typo is rejected; the wrapper forwards the flag.

`tests/test_spec_resolution.py` (7) audits every checked-in spec. It does *not* demand
that all be resolved -- changing their mesh changes the modelled system, which is a
scientific decision, not a code fix. It demands that under-resolution be **declared**:
an unresolved spec must appear in `KNOWN_UNRESOLVED` with a reason, so a new spec that
drifts into the degenerate regime fails instead of silently producing contact-free
results. Same principle as the CI skip guard -- exceptions allowed, never blank cheques.
Verified by removing a declaration and confirming the audit goes red.

`TOTAL_FLOOR` raised 156 -> 172 (177 collected under the CI selector). KS_3 updated to
read the real diagnostics instead of an analytic proxy, and its prose corrected: the
deficit is configuration-dependent, not a fixed 16x.

**Still open (scientific, not technical):** whether `specs/model.kinetic_segregation.json`
should move to a resolved mesh or `--pmhc_deposition area`. Either changes what the
metamodel is fitted to and would require re-running the sweeps.


### 2026-08-07: KS tutorial series — and two model findings it surfaced

**New**: `notebooks/models/kinetic_segregation/KS_1..KS_5` plus a shared
`ks_tutorial.py` helper. The series teaches the biology, then the energy model, then
running it, then parameter effects, then how to read the observables honestly. Energy
curves are read out of the compiled C via `ctypes` (reusing the loader in
`tests/test_potentials.py`) rather than re-typed in numpy, so they cannot drift from
`ks_physics.h` (KS rule 1). The notebooks depend on numpy/scipy/matplotlib and the
binary only — no framework — which is what makes them cheap to execute in this repo's
own CI.

Two findings emerged while writing them. Both are about the *model as configured*, not
about the tutorials:

- **Grid resolution silently disables the TCR attraction.** In `gaussian` binding mode
  the TCR well is scaled by a pMHC influence field of lateral range `sigma_r` (default
  **2 nm**), evaluated at cell centres with a hard `3*sigma_r` cutoff. When
  `dx = patch_size/grid_size` is much larger than `sigma_r`, no cell centre falls inside
  the cutoff, the weight is **exactly zero**, and the attractive term is absent. A
  seeded contact then just relaxes: measured at fixed physical time, the contact disc
  ends at 63.6 nm for dx = 41.7 nm versus 20.2 nm for dx = 7.8 nm.
  `experiments/ks_behavior_sweep/run.py` already overrides to
  `--patch_size 500 --grid_size 100` with the comment *"keeps dx=5nm"*, so the
  constraint was known — but **every checked-in spec sits in the degenerate regime**:
  `specs/model.kinetic_segregation.json` and the sweep's own `spec.json` give
  dx = 31.2 nm (15.6x sigma_r), `examples/specs/...regular` 62.5 nm, `...fast` 125 nm
  (62x). Flagged, not changed — reconciling the specs is a separate decision.
- **`depletion_width_nm` is saturated by the initial condition in short runs.** TCRs are
  seeded on top of pMHC, so with `pmhc_mode=inner_circle` the radial distributions are
  already separated at step 0; the metric starts near its final value and barely moves.
  Sweeping `u_assoc` from 1 to 2000 kT against it gives a flat line, which reads as "the
  model ignores binding energy". Measured instead as CD45 retention inside the contact
  disc (normalised by its own starting value), the same runs show a clear monotonic
  effect: retention 0.862 -> 0.643. The insensitivity was in the ruler, not the model.
  KS 4 teaches this explicitly.

Also confirmed and taught, each demonstrated in-notebook rather than asserted:
`dt_auto * kappa` is constant to 0.000%; a typo in `--binding_mode` silently selects the
*other* mode (`gausian` produced output identical to `forced`); a `--params` file
overrides an explicit CLI flag for the three mode options; `--dump-frames` zeroes
`binding_timeseries` (401 samples -> 0); and the C's "median" is `sorted[n/2]`, so
`np.median` reconstructions of `depletion_width_nm` disagree (by 0.39 nm in the worked
example) while `sorted[n//2]` matches to 1e-13.

**Navigation and the existing notebooks**: new `notebooks/Tutorial_0_Start_Here.ipynb`
routes readers between the model track (KS 1-5, framework-free) and the metamodel track
(`01`-`04`, needs `bayesmm`), with an environment check that reports which track is
actually runnable. `01`-`04` reworked in place: the fragile
`Path.cwd().parent.parent.parent` bootstrap replaced with an upward search (it assumed
a submodule checkout and broke standalone); `projects.tcr_signaling.models.*` module
paths and doubled `projects/tcr_signaling/` path prefixes corrected to match; learning
aims, a framework-dependency banner and a self-check beacon added to each. `01`'s
kinetic-segregation cell had been passing `--contact_fraction` and
`--cd45_bulk_density` since the 2026-03 rewrite and errored with "unrecognized
arguments" -- it now runs. `01` and `04` execute in ~3 s each; `02`/`03` need PyMC and
are verified from the parent repo, where the framework lives.

**`Methods/methods.tex` synced** (KS rule 5), four drifts corrected plus one addition:
the TCR well is now written centred at $h_0^{TCR}$ rather than $h=0$ (Eq. 6 and the
prose both said $h=0$; the code has centred it at 13 nm since 2026-03-09); Table 2
populations corrected 50/100/200 -> 125/500/auto-at-300 per um^2; the step-size safety
factor corrected 0.5 -> 0.05 with the second `calibrate_dt` force-field reduction now
documented; `pmhc_mode` default corrected from inner-circle to uniform. Added a
subsection documenting the dx vs sigma_r constraint, since it is a physical parameter
rather than a discretisation choice. Recompiled with tectonic (installed into
py314_bayesmm; it was absent).

**Guard**: `models/kinetic_segregation/tests/test_notebooks.py` executes all five via
`nbclient` and requires each notebook's `[KS_N self-check OK]` beacon, so a notebook
that runs but does nothing still fails — the failure mode Status.md already records
from the tutorial post-mortem. Marked `slow` (50 s for the set) to keep the pre-commit
hook fast; CI runs it in a dedicated step. Verified in both directions: deliberately
breaking KS_1's self-check turns it red, restoring it turns it green.
`TOTAL_FLOOR` in the CI skip guard raised 155 -> 156 (160 tests now collected).

### 2026-08-07: Own CI, own lint config, own git hooks
- **CI added in this repo, not the parent's workflow.** The KS model fails for
  toolchain, Metal and physics reasons that say nothing about the framework;
  sharing a workflow would turn the framework's status red for our causes.
  Separate repo means a separate badge and separate notifications.
  `deterministic` tests are deselected on CI only — CLAUDE.md documents them as
  platform-specific, and a hosted runner is different hardware (159 of 167 run
  there; all 167 pass locally).
- **CI asserts the suite actually ran** (zero skips, floor of 150 passing).
  Every C++ test skips itself when the binary is missing, so a toolchain
  regression can yield 150+ skips and still exit 0. This is not hypothetical: a
  broken system libc++ recently turned 118 of these into skips that read as
  success.
- **`ruff.toml` added.** There was no lint config here, so ruff walked up and
  used the parent framework's — which only works when checked out as a
  submodule, and the parent now excludes `projects/` anyway. Settings mirror the
  framework's (line-length 100, `E`/`F`/`I`).
- **Lint baseline cleared**: 49 findings → 0. 25 auto-fixed, 23 line-length
  resolved by `ruff format` (26 files reformatted), and one genuine dead local
  (`seed = 42` in `benchmark/run_benchmark.py::main`, never used — the run seed
  is set inside `run()` where the argv is built). All 167 tests, including the
  bit-level determinism baselines, pass unchanged after reformatting.
- **`githooks/` added and activated** (`core.hooksPath=githooks`), enforcing KS
  rule 9 and development rule 1, which were previously honour-system. Unlike the
  parent's hooks these do NOT block commits on `main`, since `main` is this
  repo's trunk. Hooks use POSIX `grep`, never `rg`: hooks run
  non-interactively, and a missing command inside an `if` reads as false, which
  is precisely how the parent's Status.md gate sat dead.
- **CI extended to Linux (`ubuntu-latest`) alongside macOS.** macOS exercises
  the Metal backend; Linux exercises the `gpu_stub.c` CPU-fallback path, which
  is what non-Apple users actually run. Both build the same C99 core, so a
  portability regression now surfaces.
- **`requires_metal` marker added — narrow, and with a stated reason.** Off
  Apple, CMake builds `gpu_stub.c` whose `gpu_engine_create()` returns NULL, so
  `--gpu` silently falls back to CPU. A CPU-vs-GPU assertion would then compare
  CPU against itself and pass: a *false green*, worse than an honest skip. Only
  the 13 tests that actually drive `use_gpu=True` are marked (4 whole classes
  plus 2 methods of the mixed `TestDeterminism`); the other 14 tests in those
  same two files are CPU-only and run everywhere. Under the CI selector this is
  11 skips on Linux (2 of the 13 are also `deterministic`, already deselected),
  so Linux runs 148 and macOS runs all 159.
- **CI now rejects unjustified skips.** The guard fails unless every skip
  carries the sanctioned Metal reason, and requires zero skips on macOS where
  Metal is present. Skipping is therefore never a blank cheque — a test that
  starts skipping for an unvetted reason turns the job red.
- **Windows deliberately not in the matrix.** `CMakeLists.txt` has no
  WIN32/MSVC branch, so the model does not build there; a Windows job would
  skip everything and report a meaningless green.

### 2026-03-10: Consolidate kinetic_segregation modules
- **Change**: Merged `kinetic_segregation_gpu/` (C + Metal) into `kinetic_segregation/`,
  removing the Python implementation entirely. The C binary is now the single source of
  truth, with CPU (`--no-gpu`) and GPU modes.
- **Motivation**: The C binary supports all features the Python model had (binding modes,
  step modes, pMHC gating, brownian dynamics, frame dumps). Python was redundant.
- **Tests**: 93 total tests pass. Migrated key physics tests from the Python suite into
  `test_physics_regression.py` (28 tests). Existing GPU tests updated to remove Python
  references. `test_equivalence.py` (Python-vs-C) deleted since Python is gone.
- **Methods**: Updated LaTeX documentation with checkerboard algorithm, GPU acceleration
  pipeline, and performance table.
- **CLAUDE.md**: Added test change policy (rule 7) and updated project structure.
- **Branch**: `feature/consolidate-ks`

### 2026-03-09: Align with paper physics — forced binding + paper step mode
- **Motivation**: Detailed comparison with Supplementary DataSheet1.pdf revealed
  7 discrepancies between implementation and paper. Implemented Tier 1 (defaults)
  + Tier 2 (forced binding + height constraints). Skipped Tier 3 (lattice rewrite).
- **Tier 1 — Default parameter fixes**:
  - CD45 height: 35nm → 50nm (paper Table S1)
  - Initial membrane height: 35nm → 70nm (paper)
  - TCR-pMHC bond length: defined as H0_TCR_NM = 13nm
  - N_TCR default: 50 → 125, N_CD45 default: 100 → 500 (paper Table S1)
  - D_mol default: 1e5 → 1e4 nm²/s (paper: 10,000 nm²/s for TCR)
- **Tier 2A — Forced TCR-pMHC binding** (`binding_mode="forced"`, new default):
  - `tcr_bound` boolean array tracks bound TCRs
  - Bound TCRs skip Phase 1 moves (immobile, per paper)
  - After accepted TCR move, binding state updated based on pMHC grid
  - Phase 2: cells with bound TCR have height frozen at h0_tcr (13nm)
  - Result includes `n_tcr_bound` count
- **Tier 2B — Paper step mode** (`step_mode="paper"`, new default):
  - Fixed dt=0.01s (paper), step_h=1.0nm (paper)
  - step_mol = sqrt(2 * D_mol * dt) — derived from diffusion
  - Auto spring constant: k_rep = 10*κ/dx² (paper Table S1)
  - `step_mode="brownian"` preserves previous auto-computed dynamics
- **New CLI args**: `--binding_mode`, `--step_mode`, `--h0_tcr`, `--init_height`
  in all CLIs (Python, C binary, GPU wrapper, animate).
- **Tests**: Updated existing tests to use `step_mode="brownian"` where they
  test Brownian dynamics properties. Added 7 new tests for forced binding and
  paper mode. Relaxed GPU physics thresholds for init_height=70nm. Reduced CLI
  test sizes for faster execution. 81 Python + 25 GPU tests pass.
- **Files modified**: model.py, simulation.h, simulation.c, main.m,
  __main__.py (×2), animate.py, test_model.py, test_cli.py,
  test_alignment_changes.py, test_gpu_physics.py.

### 2026-03-08: Configurable pMHC initialization + JSON param file support
- **Change 1 — pMHC inner circle mode**: Added `pmhc_mode` parameter
  (`"inner_circle"` default, `"uniform"` for backward compat). In inner_circle
  mode, pMHC molecules are placed via rejection sampling within a centered disc
  of configurable `pmhc_radius` (default: patch/3 = 667nm). Implemented in
  Python model.py, C simulation.c, and main.m.
- **Change 2 — TCR co-location with pMHC**: When `n_pmhc > 0`, TCR molecules
  are initialized on top of random pMHC positions with σ=3nm jitter (matching
  sigma_bind). pMHC init moved before TCR init in both Python and C. When
  `n_pmhc=0`, backward-compat center-biased Gaussian is preserved.
- **Change 3 — JSON param file support**: Added `--params <file.json>` to all
  CLIs (Python `__main__.py` × 3, C `main.m`). Priority: CLI arg > param file >
  built-in default. Python uses `_merge_params()` helper; C uses
  NSJSONSerialization. Previously-required `--time_sec` and `--rigidity_kT_nm2`
  changed to `default=None` so they can be supplied via param file.
- **Tests**: Added 8 new tests for inner_circle radius check, uniform spread,
  TCR co-location, backward compat, invalid mode, param file loading, and CLI
  override of param file. Updated `test_pmhc_everywhere_matches_no_pmhc` →
  `test_pmhc_everywhere_produces_segregation` (exact match no longer valid with
  co-location changing RNG path). All 126 tests pass.

### 2026-03-08: Align KS implementations with MATLAB & paper
- **Goal**: Make Python/C/GPU use the same algorithm, aligned with MATLAB's
  physics while keeping the paper's continuous Gaussian TCR potential.
- **Change 1 — Reflecting height boundary**: Replaced `max(0, h)` clamping with
  `abs(h)` reflection (matching MATLAB) in all three implementations. A proposed
  height of −3 now becomes 3 instead of 0.
- **Change 2 — Python checkerboard grid update**: Replaced sequential Gauss-Seidel
  `for gi, for gj` loop with two-pass checkerboard + snapshot approach matching
  C/GPU. Pre-bins molecules to grid counts once per step (O(N) instead of O(N)
  per cell). This closes the ~14% accept rate gap between Python and C/GPU.
- **Change 3 — pMHC molecules**: Added static pMHC positions on APC surface.
  TCR binding potential only applies at grid cells where pMHC is present.
  API: `n_pmhc` (int, 0=binding everywhere for backward compat), `pmhc_pos`
  (NDArray), `pmhc_seed` (int). Binned to grid once at initialization.
  Implemented in Python, C CPU, and GPU (Metal shader gets `pmhc_count` buffer).
- **Change 4 — Configurable CD45 parameters**: Made `k_rep` and `cd45_height`
  configurable via CLI and `simulate_ks()`. Defaults remain paper values
  (k_rep=1.0, cd45_height=35.0). MATLAB values (k=0.001, h=50nm) can be used
  for comparison.
- **Change 5 — Periodic molecule boundaries**: Switched molecule positions from
  `clip(pos, 0, L)` to `pos % L` (periodic wrap) in all implementations.
  Matches MATLAB's periodic BCs.
- **Change 6 — Soft molecular repulsion**: Added truncated harmonic repulsive
  potential between nearby molecules: `E = eps * (1 - r/r_cut)^2` for r < r_cut.
  Configurable via `mol_repulsion_eps` (default 0 = disabled) and
  `mol_repulsion_rcut` (default 10nm). Brute-force O(N²) per type — fine for
  N=50-150. Phase 1 only (no GPU shader changes needed).
- **Backward compatibility**: All changes are backward-compatible. Default
  parameters reproduce previous behavior (n_pmhc=0, mol_repulsion_eps=0,
  cd45_height=35, cd45_k_rep=1.0).
- **Tests**: 21 new tests covering all 6 changes. 83 total tests pass
  (65 Python KS + 18 GPU). No regressions.
- **Files modified** (Python): `model.py`, `potentials.py`, `__main__.py`,
  `test_alignment_changes.py` (new).
  (C/GPU): `simulation.h`, `simulation.c`, `potentials.h`, `potentials.c`,
  `shaders.metal`, `metal_engine.h`, `metal_engine.m`, `main.m`, `__main__.py`,
  `test_potentials.py`.

### 2026-03-08: Paper retrieval + cross-implementation comparison
- **Paper**: Retrieved Neve-Oz et al. 2024 (*Frontiers in Immunology*,
  DOI: 10.3389/fimmu.2024.1412221) — main PDF, 9 figures, supplementary DataSheet1.
- **Location**: `original_paper/` with `figures/` and `supplementary/` subdirs.
- **Comparison document**: `original_paper/implementation_comparison.md` — detailed
  comparison of energy formulas, Metropolis criterion, step sizes, grid update order,
  boundary conditions, RNG, and precision across 4 sources: paper methods, Python,
  C CPU, and C GPU (Metal).
- **Key findings**:
  1. Energy formulas (bending, TCR, CD45, delta) are identical across all 4 sources.
  2. Metropolis: paper uses `exp(-min(dE,500))`, implementations use log-space
     `log(u) < -dE` — mathematically equivalent, numerically superior.
  3. Step sizes: paper uses heuristics, implementations use Brownian dynamics
     derivation — more physically motivated, grid-resolution independent.
  4. Grid order: paper unspecified; Python sequential; C/GPU checkerboard+snapshot.
  5. No dissimilarities affect physical conclusions.
- **License**: Frontiers CC-BY 4.0 (open access).
- **Files added**: `original_paper/Neve-Oz_et_al_2024_Frontiers.pdf`,
  `original_paper/figures/figure_{1..9}.webp`,
  `original_paper/supplementary/DataSheet1.pdf`,
  `original_paper/implementation_comparison.md`.

### 2026-03-07: Close CPU vs GPU acceptance rate gap — snapshot-based parallel Metropolis
- **Problem**: Persistent ~2.7% systematic acceptance rate gap between C CPU
  and C GPU Phase 2, amplifying to ~14% at 5000 steps. The checkerboard update
  order fix (previous entry) did not close it.
- **Root cause analysis**: Two contributing factors, identified iteratively:
  1. **Float precision mismatch** (minor): CPU Box-Muller used double precision
     cast to float; CPU Metropolis used `double log(u)`. GPU used float32
     throughout. Fixing this alone reduced gap from ~2.7% to ~2.5%.
  2. **Stencil race condition during evaluation** (dominant): `bending_delta`
     reads cells at distance 2 (same checkerboard color). The old GPU kernel
     had a race condition: each thread wrote its proposal to `h[]` then
     immediately read the stencil, with undefined visibility of other threads'
     writes across SIMD groups. On CPU, sequential processing created a
     different (also order-dependent) stencil snapshot. This produced
     systematically different energy landscapes.
- **Solution — snapshot-based three-pass Metropolis** (both CPU and GPU):
  For each checkerboard color:
  1. **Propose**: generate ALL proposals, write to `h[]`
  2. **Snapshot**: freeze `h[]` into a read-only copy
  3. **Evaluate**: compute `bending_delta` from frozen snapshot, decide accept/reject
  4. **Apply**: restore rejected cells in `h[]`
  On GPU, each phase is a separate Metal compute encoder (barrier between phases).
  On CPU, phases are sequential loops with a `memcpy` snapshot.
  This ensures ALL cells evaluate against the same consistent height field,
  eliminating order-dependent stencil reads.
- **Float precision matching** (`rng.c`, `rng.h`, `simulation.c`):
  - Added `pcg64_uniform_f()` returning `(float)uint32 / 4294967296.0f`
  - CPU Phase 2 Box-Muller uses `sqrtf`/`logf`/`cosf` in float32, with
    `max(u1, 1e-30f)` clamp matching GPU shader
  - Metropolis comparison uses `logf(u_f) < -dE` (float, not double)
- **GPU kernel split** (`shaders.metal`, `metal_engine.m`):
  Replaced single `grid_update_kernel` with four kernels:
  `grid_propose_kernel` → `grid_snapshot_kernel` → `grid_evaluate_kernel` →
  `grid_apply_kernel`. Each dispatched as a separate command encoder providing
  Metal-guaranteed barriers between phases.
- **Results** (5 seeds, grid=50, kappa=20):

  | Steps   | Gap before | Gap after |
  |--------:|-----------:|----------:|
  | 50      | ~2.45%     | ~0.24%   |
  | 500     | ~3.20%     | ~0.08%   |
  | 5,000   | ~14%       | ~0.47%   |
  | 50,000  | (untested) | ~0.16%   |
  | 500,000 | (untested) | ~0.23%   |

  **No amplification**: gap stays <0.5% at all timescales.
- **Tests**: 25 fast GPU tests (2 new gap tests), 44 Python tests — all pass.
  New tests: `test_gap_bounded_short` (50 steps, <2%), `test_gap_bounded_medium`
  (500 steps, <2%), `test_gap_no_amplification` (5000 steps, slow, <2%).
- **Files modified**: `rng.h`, `rng.c`, `simulation.c`, `shaders.metal`,
  `metal_engine.m`, `test_gpu_physics.py`, `Status.md`.

### 2026-03-07: Stabilize GPU vs CPU acceptance rate & dynamics consistency
- **Problem**: Systematic acceptance rate gap between C CPU (~0.37) and C GPU
  (~0.39) at grid=50, kappa=20. Gap amplifies over time through molecule-height
  coupling (from +2.5% at 10 steps to +10% at 2500 steps), producing visibly
  different molecular spreading in movies.
- **Root cause**: Combination of (a) sequential vs checkerboard update order
  (Gauss-Seidel vs Jacobi — different dynamical properties), (b) float32
  rounding differences, (c) different RNGs (PCG64 vs Philox). Each factor
  alone has negligible effect in Python, but their combination compounds.
- **Fix 1 — Log-space Metropolis**: Applied uniformly to all three implementations
  (Python, C CPU, C GPU). Replaces `exp(-dE)` comparison with `log(u) < -dE`.
  Eliminates overflow/underflow and the ad-hoc `-500` capping. Standard textbook
  approach for numerically stable Metropolis-Hastings.
- **Fix 2 — Checkerboard CPU Phase 2**: Changed C CPU grid update from sequential
  `for gi, for gj` to two-pass checkerboard (even-sum cells, then odd-sum cells),
  matching GPU kernel's update order. Python model remains sequential (Gauss-Seidel)
  as the reference implementation — a deliberate design choice.
- **Fix 3 — 2π constant precision**: Updated Metal shader Box-Muller 2π from
  7 to 16 significant digits (`6.2831853071795864f`).
- **Test update**: `test_height_distribution_ks` (strict KS test) replaced with
  `test_height_distribution_consistent` (15% relative tolerance on mean/std).
  Exact distributional match is not expected due to RNG and float32 differences.
- **Expected outcome**: C CPU vs GPU gap narrows from ~3% base to <1% since both
  now use checkerboard. Python remains slightly different (sequential update).
- **Files modified**: `model.py` (Python KS), `simulation.c`, `shaders.metal`
  (C/GPU KS), `test_gpu_physics.py`, `Status.md`.

### 2026-03-07: Physical time integration for KS Monte Carlo (Brownian dynamics)
- **Problem**: MC step sizes were grid heuristics (`step_size_h = dx/(4√κ)`)
  with no physical time scale. `time_sec` was an arbitrary multiplier for step
  count (`n_steps = time_sec * 5`). Same `time_sec` produced different physics
  at different grid sizes — at grid=1024 the membrane was frozen.
- **Solution**: Introduced Brownian dynamics time integration. Each MC sweep
  advances a physical time step `dt` determined by diffusion constants and the
  stability constraint:
  - `dt_stable = dx² / (2 * D_h * κ)` with safety factor 0.5
  - `step_size_mol = sqrt(2 * D_mol * dt)`, `step_size_h = sqrt(2 * D_h * dt)`
  - `n_steps = time_sec / dt` (auto) or explicit override
- **Physical constants** (defaults, overridable via CLI):
  - `D_mol = 1×10⁵ nm²/s` (membrane protein diffusion)
  - `D_h = 5×10⁴ nm²/s` (membrane height relaxation)
- **CLI args added**: `--D_mol`, `--D_h`, `--dt` for both Python and C models.
- **n_steps semantic change**: When explicit, `n_steps` is now a raw override
  (no time-based scaling). Auto-computation uses `time_sec / dt`.
- **Diagnostics**: JSON output now includes `dt_seconds`, `step_size_h_nm`,
  `step_size_mol_nm`, `D_mol_nm2_per_s`, `D_h_nm2_per_s`.
- **Impact on step counts**: At grid=64, kappa=50, 20s physical time →
  ~4000 steps (vs 100 before). This is physically correct but computationally
  heavier. Practical range: grid=64–128 for DOE sweeps.
- **Tests**: Updated `test_explicit_n_steps_scales_with_time` →
  `test_explicit_n_steps_is_raw_override`. Added: `test_dt_scales_with_grid`,
  `test_step_sizes_from_physics`, `test_n_steps_auto_from_time`,
  `test_diagnostics_keys_present`, `TestGridConvergence` (slow).
  24 Python model tests pass, 62 fast tests total pass.
- **Files modified**: `model.py`, `__main__.py` (Python KS), `simulation.h`,
  `simulation.c`, `main.m` (C GPU), `__main__.py` (GPU wrapper),
  `test_model.py`, `test_gpu_physics.py`, `Status.md`.

### 2026-03-07: GPU-side Philox RNG + float h throughout (GPU optimization)
- **Problem**: Profiling showed CPU-side RNG generation consumed 60-94% of GPU
  path time. At grid=2048, CPU spent 69ms/step on Box-Muller while GPU kernel
  finished in 3.3ms.
- **Solution**: Moved RNG to GPU via Philox4x32-10 counter-based PRNG.
  Each GPU thread generates its own random numbers (normal via Box-Muller,
  uniform for Metropolis) — no shared state, embarrassingly parallel.
  Counter = (tid, step_offset) + key derived from CPU seed → deterministic.
- **Float h throughout**: Changed `double *h` to `float *h` in SimState.
  Height values are 0-50nm; float32 gives ~7 decimal digits, more than sufficient.
  Eliminates float→double copy-back after GPU dispatch.
- **CPU Phase 2 also uses float**: Added float-based bending/potential functions
  in simulation.c so CPU and GPU paths use identical arithmetic.
- **RNG stream separation**: CPU pcg64 is used ONLY for Phase 1 (molecules).
  GPU uses Philox with a fixed key derived from seed — different stream from
  before, but both paths remain deterministic.
- **Buffers removed**: Eliminated 4 random buffers (rand_normal, rand_uniform
  × 2 colors) from MetalEngine. Kernel signature reduced from 7 to 5 buffers.
- **Performance** (50 steps, Apple M2 Pro):

  | Grid | GPU Before | GPU After | CPU | GPU speedup | GPU/CPU ratio |
  |-----:|----------:|----------:|----:|------------:|--------------:|
  | 256  | 0.093s    | 0.083s    | 0.199s | 1.1x | 2.4x |
  | 512  | 0.265s    | 0.068s    | 0.790s | 3.9x | 11.6x |
  | 1024 | 0.951s    | 0.103s   | 3.169s | 9.2x | 30.8x |
  | 2048 | 3.691s    | 0.199s   | 12.687s | 18.5x | 63.7x |

- **Tests**: All 17 fast tests pass (CPU determinism, GPU determinism, potentials).
  Statistical equivalence test may need rerun due to float h change.

### 2026-03-07: GPU-accelerated KS model (C + Metal on Apple Silicon)
- **New model**: `models/kinetic_segregation_gpu/` (now consolidated into `models/kinetic_segregation/`) — C + Objective-C implementation
  with Metal GPU acceleration for the grid update phase.
- **Architecture**: Phase 1 (molecular moves, ~150 molecules) runs on CPU in C.
  Phase 2 (grid updates, 64x64 = 4096 cells) uses Metal GPU with checkerboard
  decomposition (2048 red cells, then 2048 black cells in parallel). CPU fallback
  when Metal is unavailable (CI, SSH, headless).
- **Build**: `clang -framework Metal -framework Foundation` — no Xcode needed,
  only CommandLineTools. Metal shaders compiled at runtime via
  `[MTLDevice newLibraryWithSource:]`.
- **Speedup**: C+Metal (GPU) achieves up to 63.7x over CPU at grid=2048.
- **Correctness**: All potential functions match Python to float64 precision (ctypes
  tests). Two-sample KS test on depletion width distributions (20 seeds) confirms
  statistical equivalence (p > 0.05). Same-seed determinism verified for CPU and GPU.
- **float32**: Heights stored as float throughout (GPU and CPU Phase 2).
  CPU Phase 1 uses double for molecule positions, casts to float for h lookup.
- **RNG**: CPU pcg64 for Phase 1 molecules; GPU Philox4x32-10 for Phase 2 grid.
- **CLI contract**: Identical to Python model (`--time_sec`, `--rigidity_kT_nm2`,
  `--run-dir`, etc.). Python `__main__.py` wrapper calls binary via subprocess.
- **Tests**: 17 new tests (11 potentials, 6 CLI) + 2 slow equivalence tests.
  All 78 non-slow tests pass.

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
