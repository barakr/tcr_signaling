/*
 * gpu_stub.c — Stub GPU backend for platforms without Metal or CUDA.
 *
 * All functions return NULL / no-op, triggering CPU fallback in simulation.c.
 * Linked on Linux (or macOS without Metal) to satisfy gpu_engine.h symbols.
 */
#include "gpu_engine.h"
#include <stddef.h>

void *gpu_engine_create(int grid_size, uint64_t gpu_rng_key) {
    (void)grid_size; (void)gpu_rng_key;
    return NULL;
}

void gpu_engine_destroy(void *ctx) { (void)ctx; }

float *gpu_engine_h_ptr(void *ctx) { (void)ctx; return NULL; }

void gpu_engine_grid_update(void *ctx, float *h, int grid_size,
                            double kappa, double dx, double step_size_h,
                            double u_assoc, double sigma_bind,
                            double cd45_height, double k_rep,
                            const double *tcr_pos, int n_tcr,
                            const double *cd45_pos, int n_cd45,
                            const int *pmhc_count,
                            long *accepted, long *total_proposals,
                            int n_substeps) {
    (void)ctx; (void)h; (void)grid_size;
    (void)kappa; (void)dx; (void)step_size_h;
    (void)u_assoc; (void)sigma_bind;
    (void)cd45_height; (void)k_rep;
    (void)tcr_pos; (void)n_tcr;
    (void)cd45_pos; (void)n_cd45;
    (void)pmhc_count;
    (void)accepted; (void)total_proposals;
    (void)n_substeps;
}
