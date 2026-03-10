#ifndef GPU_ENGINE_H
#define GPU_ENGINE_H

#include <stdint.h>

/*
 * gpu_engine.h — Backend-neutral GPU engine interface.
 *
 * Current implementation: Metal (metal_engine.m, macOS/Apple Silicon).
 * Future: CUDA (cuda_engine.cu, Linux/NVIDIA).
 * The simulation core (simulation.c) calls these functions via extern linkage.
 */

/* Create a GPU engine for grid updates. Returns NULL if GPU unavailable.
   gpu_rng_key is derived from the CPU seed for deterministic GPU RNG. */
void *gpu_engine_create(int grid_size, uint64_t gpu_rng_key);

/* Destroy the GPU engine. */
void gpu_engine_destroy(void *ctx);

/* Return a pointer to the shared GPU height buffer.
   On Apple Silicon unified memory, CPU can read/write this directly,
   eliminating the need for per-step memcpy. */
float *gpu_engine_h_ptr(void *ctx);

/* Run Phase 2 grid update on GPU using checkerboard decomposition.
   Updates h (float*) in-place. Molecule positions are read-only (for binning).
   n_substeps batches multiple grid sweeps in a single command buffer commit.
   No CPU RNG consumed — GPU uses counter-based PRNG internally. */
void gpu_engine_grid_update(void *ctx, float *h, int grid_size,
                            double kappa, double dx, double step_size_h,
                            double u_assoc, double sigma_bind,
                            double cd45_height, double k_rep,
                            const double *tcr_pos, int n_tcr,
                            const double *cd45_pos, int n_cd45,
                            const int *pmhc_count,
                            long *accepted, long *total_proposals,
                            int n_substeps);

#endif /* GPU_ENGINE_H */
