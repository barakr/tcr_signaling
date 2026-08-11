# TCR Signaling Metamodel — Neve-Oz, Sherman & Raveh 2024

Reproduction of "Bayesian metamodeling of early T-cell antigen receptor signaling
accounts for its nanoscale activation patterns" (Neve-Oz, Sherman & Raveh,
*Frontiers in Immunology*, 2024).

## New here? Start with the tutorials

```bash
conda activate py314_bayesmm
jupyter lab notebooks/Tutorial_0_Start_Here.ipynb
```

`Tutorial_0` is orientation only — ten minutes, no simulation — and its environment check
tells you which parts will run on your machine before you hit an error. From there:

- **[`notebooks/models/kinetic_segregation/`](notebooks/models/kinetic_segregation/)** —
  a five-notebook course on the kinetic-segregation model: the biology, the energy
  function, running it, what each parameter does, and how to read the results without
  fooling yourself. Framework-free (numpy + a C toolchain); the whole series runs in
  about a minute.
- **[`notebooks/`](notebooks/README.md)** — the metamodel track: sweeps, surrogates,
  coupling, uncertainty propagation and the paper's figures. Needs `bayesmm`, and PyMC for `02`/`03`.

See [`notebooks/README.md`](notebooks/README.md) for the full map and timings.

## Build prerequisites (Windows, macOS, Linux)

The kinetic-segregation model is compiled C — the notebooks build it for you on first
run, but they need a **C/C++ compiler and CMake ≥ 3.20** to do it. Everything else in
the repo is pure Python.

**Windows.** Install the Visual Studio 2022 Build Tools with the *Desktop development
with C++* workload, which supplies both MSVC and CMake:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Then **reopen your terminal** so the new tools are on `PATH`. If you already have Visual
Studio but the build fails, the usual cause is a missing or outdated C++ workload — open
the Visual Studio Installer, choose *Modify*, and tick *Desktop development with C++*.
Full Visual Studio works just as well as the Build Tools; the Build Tools are simply the
smaller download.

If CMake is still not found afterwards, `conda install -c conda-forge cmake` or
`winget install Kitware.CMake` adds it on its own.

**macOS**

```bash
xcode-select --install
conda install -c conda-forge cmake     # or: brew install cmake
```

**Linux**

```bash
sudo apt install build-essential cmake        # Debian/Ubuntu
conda install -c conda-forge cmake compilers  # without sudo
```

Then build (from `models/kinetic_segregation/`) — the same two commands on every platform:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

`make` still works on macOS and Linux as a shorthand for exactly those two commands.
There is no `make` on a stock Windows box, which is why nothing in the test suite or the
notebooks calls it.

> **Which compilers are tested.** CI builds the model on every push with **MSVC**
> (Windows), **Clang** (macOS, plus the Metal GPU backend) and **GCC** (Linux). MinGW-w64
> and MSYS2 are likely to work but are not tested — if you are on Windows and have a
> choice, use MSVC. The GPU path is Apple-only; Windows and Linux run the same C99 CPU
> core, which produces the same physics.

## Paper Reference

- **Title**: Bayesian metamodeling of early T-cell antigen receptor signaling
  accounts for its nanoscale activation patterns
- **Authors**: Y. Neve-Oz, E. Sherman, B. Barak Raveh
- **Journal**: Frontiers in Immunology, 2024
- **DOI**: 10.3389/fimmu.2024.1437672

## Partial Models

The paper couples four partial models of early TCR signaling:

1. **Membrane Topography** — IRM-derived tight-contact geometry (2 um x 2 um patches)
2. **Kinetic Segregation (KS)** — Spatial exclusion of CD45 from tight contacts
3. **Lck Activity** — Radially-symmetric exponential decay of active Lck from
   CD45 boundary positions (decay length ~70 nm)
4. **TCR Phosphorylation** — Lck* phosphorylates TCR ITAMs; validated against
   ZAP-70 super-resolution data

## Reproduction Workflow

```bash
# 1. Validate all model specs
for spec in specs/model.*.json; do bayesmm validate "$spec"; done

# 2. Run parameter sweeps for each model
for spec in specs/model.*.json; do bayesmm run "$spec"; done

# 3. Fit surrogates on sweep data
for spec in specs/surrogate.*.json; do bayesmm surrogate fit "$spec"; done

# 4. Build and sample the joint metamodel
bayesmm meta build specs/metamodel.tcr_signaling.json
bayesmm meta sample specs/metamodel.tcr_signaling.json --draws 2000 --tune 1000
```

## Directory Layout

```
models/          Python CLI scripts for each partial model
specs/           JSON specs (ModelSpec, SurrogateSpec, MetaModelSpec)
data/            Experimental reference data
notebooks/       Jupyter analysis and figure reproduction
store/           Run artifacts (gitignored)
artifacts/       Pre-trained surrogate artifacts
```

## Key Parameters

| Parameter | Typical Range | Units | Description |
|-----------|---------------|-------|-------------|
| contact_radius | 0.5 - 2.0 | um | Tight-contact patch radius |
| cd45_exclusion_threshold | 10 - 50 | nm | Height threshold for CD45 exclusion |
| lck_decay_length | 30 - 150 | nm | Exponential decay length of active Lck |
| lck_activation_rate | 0.1 - 1.0 | 1/s | Rate of Lck activation at CD45 boundary |
| tcr_density | 50 - 300 | 1/um^2 | Surface density of TCR molecules |
| phosphorylation_rate | 0.01 - 0.5 | 1/s | Rate of TCR ITAM phosphorylation by Lck* |
