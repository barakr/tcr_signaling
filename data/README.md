# Experimental Reference Data

## Data Provenance

Reference data for validating the TCR signaling metamodel comes from
super-resolution microscopy of ZAP-70 recruitment patterns at the
immunological synapse.

## Sources

- ZAP-70 radial enrichment profiles: Neve-Oz, Sherman & Raveh 2024, Figure 4
- IRM contact geometry statistics: derived from Interference Reflection
  Microscopy imaging of T-cell / bilayer interfaces

## Data Files

Data files are not included in the repository. To obtain them:

1. Contact the paper authors for raw data
2. Or digitize from the published figures using WebPlotDigitizer

Place CSV files in this directory with the following naming convention:
- `zap70_radial_profile.csv` — columns: `radius_um`, `enrichment`
- `irm_contact_statistics.csv` — columns: `contact_radius_um`, `frequency`
