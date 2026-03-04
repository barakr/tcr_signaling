# TCR Signaling Metamodel — Neve-Oz, Sherman & Raveh 2024

Reproduction of "Bayesian metamodeling of early T-cell antigen receptor signaling
accounts for its nanoscale activation patterns" (Neve-Oz, Sherman & Raveh,
*Frontiers in Immunology*, 2024).

## Paper Reference

- **Title**: Bayesian metamodeling of early T-cell antigen receptor signaling
  accounts for its nanoscale activation patterns
- **Authors**: Y. Neve-Oz, E. Sherman, B. Bhatt Raveh
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
