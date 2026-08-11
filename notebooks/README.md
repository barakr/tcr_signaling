# Notebooks

Start with **[`Tutorial_0_Start_Here.ipynb`](Tutorial_0_Start_Here.ipynb)**. It takes ten
minutes, runs no simulation, and its environment check tells you which of the two tracks
below will actually work on your machine before you hit an error.

```bash
conda activate py314_bayesmm
jupyter lab Tutorial_0_Start_Here.ipynb
```

## Two tracks

**Track A — the model.** How does one biophysical model work? Currently that means
kinetic segregation: the biology, the energy function, the Monte Carlo scheme, what each
parameter does, and how to measure the result without fooling yourself.

Needs numpy, matplotlib and a C toolchain (CMake + a C++ compiler). **No framework.** The
notebooks build the simulator themselves the first time you run them.

| Notebook | Question it answers | Time |
|---|---|---|
| [`models/kinetic_segregation/KS_1_Kinetic_Segregation`](models/kinetic_segregation/KS_1_Kinetic_Segregation.ipynb) | What *is* kinetic segregation, and why does it need a simulation? | ~2 s |
| [`KS_2_The_Energy_Model`](models/kinetic_segregation/KS_2_The_Energy_Model.ipynb) | What are the three energy terms, and what is the MC scheme? | ~4 s |
| [`KS_3_First_Simulation`](models/kinetic_segregation/KS_3_First_Simulation.ipynb) | Run it; what does every output key mean? | ~18 s |
| [`KS_4_Key_Parameters`](models/kinetic_segregation/KS_4_Key_Parameters.ipynb) | What do rigidity, CD45 height and binding mode actually do? | ~33 s |
| [`KS_5_Observables_and_Pitfalls`](models/kinetic_segregation/KS_5_Observables_and_Pitfalls.ipynb) | Which of the eight metrics should you trust, and when? | ~2 s |

Timings are from a clean clone on an M-series Mac, including the one-off build of
the simulator; the series is about a minute end to end.

**Work them in order.** Each builds on the previous one, and KS 4 assumes the resolution
lesson from KS 3.

**Track B — the metamodel.** How do four separately-built models get combined into one
joint description? Sweeps, surrogates, couplings and the paper's figures.

One thing to carry in: `bayesmm meta sample` has **two modes**, and they answer
different questions.

- `--method propagate` (the default) draws from the priors and applies the couplings as a
  post-draw transform. That is forward uncertainty propagation, not conditioning — the
  surrogate likelihoods are never evaluated, so calling its output a "posterior" is a
  misnomer.
- `--method joint` runs Metropolis over the full joint density — priors, couplings *and*
  surrogate likelihoods — so both ends of a coupling move and the surrogates actually
  constrain the result.

`03` covers both. Read the word "posterior" with the method in mind.

Needs the `bayesian-metamodeling` framework (`bayesmm`), and PyMC for `02` and `03`.

| Notebook | Question it answers | Needs |
|---|---|---|
| [`01_explore_models`](01_explore_models.ipynb) | What do the four partial models each produce? | `bayesmm` |
| [`02_fit_surrogates`](02_fit_surrogates.ipynb) | How do you replace a slow model with a fast probabilistic one? | `bayesmm`, PyMC |
| [`03_metamodel_inference`](03_metamodel_inference.ipynb) | How does uncertainty propagate through coupled models? | `bayesmm`, PyMC |
| [`04_reproduce_figures`](04_reproduce_figures.ipynb) | Why does phosphorylated TCR form a *ring*? | `bayesmm` |

Use the `py312_bayesmm_pymc` environment for this track.

> **`02` runs at a reduced "teaching scale" by default**, so the series is quick enough to
> sit through. That is a deliberate choice, not a shortcut hidden from you: the notebook
> says so, and its *Running it for real* section gives the single switch
> (`TEACHING_SCALE = False`) plus the commands for the production design. At production
> scale the sweeps take hours — KS wall-time is linear in simulated time, and the full
> design spans `time_sec` up to 100 s.
>
> Worth knowing before you read the numbers: the teaching values sit **below** the
> production parameter range, not merely at its cheap end. The surrogate you fit is a
> demonstration of the mechanics, not a scientific result.

## If you are new here

Read **KS 1 first even if you came for the metamodel.** Track B treats each partial model
as a box that emits a number; Track A is where you learn what that number means and how
badly it can mislead you. Three findings in the KS series are the difference between a
result and an artefact:

- Grid spacing is a physical parameter, not a discretisation choice (KS 3).
- A parameter that appears to do nothing may mean your *observable* is saturated, not
  that your model is insensitive (KS 4).
- Two of the eight depletion metrics are `null` by default, and they happen to be the two
  that survive an off-centre contact (KS 5).

## Notes

- Every notebook ships with its outputs stored, so you can read them on GitHub without
  running anything.
- Each ends in a self-check cell printing `[... self-check OK]`. If you re-run a notebook
  and that line does not appear, something is wrong even if no cell raised —
  `models/kinetic_segregation/tests/test_notebooks.py` enforces exactly this in CI.
- KS 3 has a `MAKE_MOVIE = False` cell. Set it `True` (and have `ffmpeg` on PATH) to
  render an MP4 of a run via `render_movie.py`. It is off by default because CI has no
  ffmpeg.
