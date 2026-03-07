#ifndef METAL_ENGINE_H
#define METAL_ENGINE_H

#include <stdint.h>

/* Create a Metal GPU engine for grid updates. Returns NULL if Metal unavailable.
   gpu_rng_key is derived from the CPU seed for deterministic GPU RNG. */
void *metal_engine_create(int grid_size, uint64_t gpu_rng_key);

/* Destroy the Metal engine. */
void metal_engine_destroy(void *ctx);

/* Run Phase 2 grid update on GPU using checkerboard decomposition.
   Updates h (float*) in-place. Molecule positions are read-only (for binning).
   No CPU RNG consumed — GPU uses Philox counter-based PRNG internally. */
void metal_engine_grid_update(void *ctx, float *h, int grid_size,
                              double kappa, double dx, double step_size_h,
                              double u_assoc, double sigma_bind,
                              double cd45_height,
                              const double *tcr_pos, int n_tcr,
                              const double *cd45_pos, int n_cd45,
                              long *accepted, long *total_proposals);

#endif /* METAL_ENGINE_H */
