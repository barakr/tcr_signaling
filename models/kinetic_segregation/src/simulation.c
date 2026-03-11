#include "simulation.h"
#include "potentials.h"
#include "cell_list.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef KS_PROFILE
#include <time.h>
static double _profile_phase1_ms = 0.0;
static double _profile_phase2_ms = 0.0;
double _profile_bin_ms = 0.0;  /* non-static: shared with metal_engine.m */

static double _clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void sim_profile_report(int n_steps) {
    fprintf(stderr, "PROFILE: phase1=%.1fms phase2=%.1fms bin=%.1fms total=%.1fms "
            "(per-step: p1=%.3fms p2=%.3fms bin=%.3fms)\n",
            _profile_phase1_ms, _profile_phase2_ms, _profile_bin_ms,
            _profile_phase1_ms + _profile_phase2_ms + _profile_bin_ms,
            _profile_phase1_ms / n_steps, _profile_phase2_ms / n_steps,
            _profile_bin_ms / n_steps);
}
#endif

/* GPU backend (Metal on macOS, CUDA on Linux — see gpu_engine.h). */
#include "gpu_engine.h"

/* Forward declaration. */
static void bin_molecules(const double *pos, int n_mol, int n, double dx,
                          int *count_grid);

static void init_height_field(SimState *s) {
    int n = s->grid_size;
    for (int i = 0; i < n * n; i++)
        s->h[i] = (float)s->init_height;

    /* Depress center to create initial tight-contact seed. */
    int center = n / 2;
    int radius = n / 8;
    if (radius < 1) radius = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int di = i - center;
            int dj = j - center;
            if (di * di + dj * dj <= radius * radius)
                s->h[i * n + j] = (float)s->h0_tcr;
        }
    }
}

static void init_positions(double *pos, int n, double patch, pcg64_t *rng,
                           int center_bias) {
    double center = patch / 2.0;
    double spread = patch / 6.0;
    for (int i = 0; i < n; i++) {
        if (center_bias) {
            pos[2 * i] = fmod(center + pcg64_normal(rng, spread), patch);
            if (pos[2 * i] < 0.0) pos[2 * i] += patch;
            pos[2 * i + 1] = fmod(center + pcg64_normal(rng, spread), patch);
            if (pos[2 * i + 1] < 0.0) pos[2 * i + 1] += patch;
        } else {
            pos[2 * i] = pcg64_uniform(rng) * patch;
            pos[2 * i + 1] = pcg64_uniform(rng) * patch;
        }
    }
}

/* Compute initial TCR binding state and enforce height constraints. */
static void init_binding_state(SimState *s) {
    if (!s->tcr_bound || !s->pmhc_count) return;
    int n = s->grid_size;
    for (int i = 0; i < s->n_tcr; i++) {
        int gi = (int)(s->tcr_pos[2 * i] / s->dx);
        if (gi >= n) gi = n - 1;
        int gj = (int)(s->tcr_pos[2 * i + 1] / s->dx);
        if (gj >= n) gj = n - 1;
        if (s->pmhc_count[gi * n + gj] > 0) {
            s->tcr_bound[i] = 1;
            s->h[gi * n + gj] = (float)s->h0_tcr;
        }
    }
}

SimState *sim_create(int grid_size, int n_tcr, int n_cd45,
                     double kappa, double u_assoc, uint64_t seed,
                     int use_gpu,
                     double D_mol, double D_h, double dt_override,
                     double cd45_height, double k_rep,
                     double mol_repulsion_eps, double mol_repulsion_rcut,
                     int n_pmhc, uint64_t pmhc_seed,
                     int pmhc_mode, double pmhc_radius,
                     int binding_mode, int step_mode,
                     double h0_tcr, double init_height) {
    SimState *s = (SimState *)calloc(1, sizeof(SimState));
    s->grid_size = grid_size;
    s->dx = PATCH_SIZE_NM / grid_size;
    s->n_tcr = n_tcr;
    s->n_cd45 = n_cd45;
    s->kappa = kappa;
    s->u_assoc = u_assoc;
    s->cd45_height = (cd45_height > 0.0) ? cd45_height : CD45_HEIGHT_NM;
    s->h0_tcr = (h0_tcr > 0.0) ? h0_tcr : H0_TCR_NM;
    s->init_height = (init_height > 0.0) ? init_height : INIT_HEIGHT_NM;
    s->binding_mode = binding_mode;
    s->step_mode = step_mode;
    s->mol_repulsion_eps = mol_repulsion_eps;
    s->mol_repulsion_rcut = (mol_repulsion_rcut > 0.0) ? mol_repulsion_rcut : 10.0;
    s->n_pmhc = n_pmhc;
    s->pmhc_pos = NULL;
    s->pmhc_count = NULL;
    s->pmhc_mode = pmhc_mode;
    s->pmhc_radius = pmhc_radius;
    s->tcr_bound = NULL;

    /* Apply defaults if zero. */
    if (D_mol <= 0.0) D_mol = D_MOL_DEFAULT;
    if (D_h <= 0.0) D_h = D_H_DEFAULT;
    s->D_mol = D_mol;
    s->D_h = D_h;

    /* Spring constant: paper formula or explicit. */
    if (k_rep > 0.0) {
        s->k_rep = k_rep;
    } else if (step_mode == 1) {
        /* paper: k = 10*kappa/a^2 */
        s->k_rep = 10.0 * kappa / (s->dx * s->dx);
    } else {
        s->k_rep = 1.0;
    }

    /* Time step and step sizes. */
    if (dt_override > 0.0) {
        s->dt = dt_override;
        s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
        s->step_size_h = sqrt(2.0 * D_h * s->dt);
    } else if (step_mode == 1) {
        /* Paper mode: fixed dt and height step. */
        s->dt = DT_PAPER;
        s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
        s->step_size_h = STEP_H_PAPER;
    } else {
        /* Brownian dynamics time step from stability constraint. */
        double dt_stable = (s->dx * s->dx) / (2.0 * D_h * kappa);
        s->dt = dt_stable * DT_SAFETY;
        s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
        s->step_size_h = sqrt(2.0 * D_h * s->dt);
    }

    s->h = (float *)malloc(grid_size * grid_size * sizeof(float));
    s->tcr_pos = (double *)malloc(n_tcr * 2 * sizeof(double));
    s->cd45_pos = (double *)malloc(n_cd45 * 2 * sizeof(double));

    pcg64_seed(&s->rng, seed);
    init_height_field(s);

    /* --- pMHC initialization (before TCR so TCR can co-locate) --- */
    if (n_pmhc > 0) {
        s->pmhc_pos = (double *)malloc(n_pmhc * 2 * sizeof(double));
        s->pmhc_count = (int *)calloc(grid_size * grid_size, sizeof(int));
        pcg64_t pmhc_rng;
        pcg64_seed(&pmhc_rng, pmhc_seed);

        double eff_radius = (pmhc_radius > 0.0) ? pmhc_radius : PATCH_SIZE_NM / 3.0;
        double center_xy = PATCH_SIZE_NM / 2.0;

        if (pmhc_mode == 1) {
            /* inner_circle: rejection sampling within centered disc */
            int placed = 0;
            while (placed < n_pmhc) {
                double cx = pcg64_uniform(&pmhc_rng) * PATCH_SIZE_NM;
                double cy = pcg64_uniform(&pmhc_rng) * PATCH_SIZE_NM;
                double ddx = cx - center_xy;
                double ddy = cy - center_xy;
                if (ddx * ddx + ddy * ddy <= eff_radius * eff_radius) {
                    s->pmhc_pos[2 * placed] = cx;
                    s->pmhc_pos[2 * placed + 1] = cy;
                    placed++;
                }
            }
        } else {
            /* uniform: full patch */
            for (int i = 0; i < n_pmhc; i++) {
                s->pmhc_pos[2 * i] = pcg64_uniform(&pmhc_rng) * PATCH_SIZE_NM;
                s->pmhc_pos[2 * i + 1] = pcg64_uniform(&pmhc_rng) * PATCH_SIZE_NM;
            }
        }
        bin_molecules(s->pmhc_pos, n_pmhc, grid_size, s->dx, s->pmhc_count);
    }

    /* --- TCR initialization --- */
    if (n_pmhc > 0 && s->pmhc_pos) {
        /* Co-locate TCR on pMHC positions with small jitter (sigma_bind) */
        for (int i = 0; i < n_tcr; i++) {
            int pidx = (int)(pcg64_uniform(&s->rng) * n_pmhc);
            if (pidx >= n_pmhc) pidx = n_pmhc - 1;
            double tx = s->pmhc_pos[2 * pidx] + pcg64_normal(&s->rng, SIGMA_BIND_NM);
            double ty = s->pmhc_pos[2 * pidx + 1] + pcg64_normal(&s->rng, SIGMA_BIND_NM);
            tx = fmod(tx, PATCH_SIZE_NM); if (tx < 0.0) tx += PATCH_SIZE_NM;
            ty = fmod(ty, PATCH_SIZE_NM); if (ty < 0.0) ty += PATCH_SIZE_NM;
            s->tcr_pos[2 * i] = tx;
            s->tcr_pos[2 * i + 1] = ty;
        }
    } else {
        /* Backward compat: center-biased Gaussian */
        init_positions(s->tcr_pos, n_tcr, PATCH_SIZE_NM, &s->rng, 1);
    }

    /* CD45: always uniform */
    init_positions(s->cd45_pos, n_cd45, PATCH_SIZE_NM, &s->rng, 0);

    /* --- TCR binding state (forced mode) --- */
    if (binding_mode == 1 && s->pmhc_count) {
        s->tcr_bound = (int *)calloc(n_tcr, sizeof(int));
        init_binding_state(s);
    }

    s->accepted = 0;
    s->total_proposals = 0;

    /* Allocate persistent cell lists if repulsion is enabled. */
    s->tcr_cell_list = NULL;
    s->cd45_cell_list = NULL;
    if (mol_repulsion_eps > 0.0 && mol_repulsion_rcut > 0.0) {
        CellList *tcl = (CellList *)malloc(sizeof(CellList));
        cell_list_init(tcl, n_tcr, mol_repulsion_rcut, PATCH_SIZE_NM);
        s->tcr_cell_list = tcl;
        CellList *ccl = (CellList *)malloc(sizeof(CellList));
        cell_list_init(ccl, n_cd45, mol_repulsion_rcut, PATCH_SIZE_NM);
        s->cd45_cell_list = ccl;
    }

    /* Try to init Metal GPU engine. Derive GPU RNG key from CPU seed. */
    s->use_gpu = 0;
    s->metal_ctx = NULL;
    s->h_is_shared = 0;
    s->grid_substeps = 1;
    if (use_gpu) {
        /* Use a deterministic key derived from the seed for GPU Philox RNG. */
        uint64_t gpu_key = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
        s->metal_ctx = gpu_engine_create(grid_size, gpu_key);
        if (s->metal_ctx) {
            s->use_gpu = 1;
            /* Move h to the Metal shared buffer so CPU and GPU share memory
               directly (no per-step memcpy needed on unified memory). */
            float *shared_h = gpu_engine_h_ptr(s->metal_ctx);
            memcpy(shared_h, s->h, grid_size * grid_size * sizeof(float));
            free(s->h);
            s->h = shared_h;
            s->h_is_shared = 1;
        }
    }

    return s;
}

void sim_destroy(SimState *s) {
    if (!s) return;
    if (s->metal_ctx) gpu_engine_destroy(s->metal_ctx);
    if (!s->h_is_shared) free(s->h);
    free(s->tcr_pos);
    free(s->cd45_pos);
    free(s->pmhc_pos);
    free(s->pmhc_count);
    free(s->tcr_bound);
    if (s->tcr_cell_list) {
        cell_list_free((CellList *)s->tcr_cell_list);
        free(s->tcr_cell_list);
    }
    if (s->cd45_cell_list) {
        cell_list_free((CellList *)s->cd45_cell_list);
        free(s->cd45_cell_list);
    }
    free(s);
}

static float height_at_pos_f(const float *h, int n, double dx, double x, double y) {
    int ix = (int)(x / dx);
    int iy = (int)(y / dx);
    if (ix < 0) ix = 0;
    if (ix >= n) ix = n - 1;
    if (iy < 0) iy = 0;
    if (iy >= n) iy = n - 1;
    return h[ix * n + iy];
}

static void phase1_molecules(SimState *s) {
    double patch = PATCH_SIZE_NM;
    int n = s->grid_size;
    double dx = s->dx;
    int use_cell = (s->mol_repulsion_eps > 0.0 && s->mol_repulsion_rcut > 0.0);

    /* Use persistent cell lists (allocated in sim_create, rebuilt here). */
    CellList *tcr_cl = (CellList *)s->tcr_cell_list;
    CellList *cd45_cl = (CellList *)s->cd45_cell_list;
    if (use_cell) {
        cell_list_build(tcr_cl, s->tcr_pos, s->n_tcr);
        cell_list_build(cd45_cl, s->cd45_pos, s->n_cd45);
    }

    /* TCR molecules. */
    for (int idx = 0; idx < s->n_tcr; idx++) {
        /* Forced binding: bound TCRs are immobile. */
        if (s->binding_mode == 1 && s->tcr_bound && s->tcr_bound[idx]) {
            continue;
        }

        double ox = s->tcr_pos[2 * idx];
        double oy = s->tcr_pos[2 * idx + 1];
        double old_h = (double)height_at_pos_f(s->h, n, dx, ox, oy);
        int old_ix = (int)(ox / dx); if (old_ix >= n) old_ix = n - 1;
        int old_iy = (int)(oy / dx); if (old_iy >= n) old_iy = n - 1;
        int has_pmhc_old = (s->pmhc_count == NULL) || (s->pmhc_count[old_ix * n + old_iy] > 0);
        double old_e = has_pmhc_old ? tcr_pmhc_potential(old_h, s->u_assoc, SIGMA_BIND_NM) : 0.0;

        double nx_ = ox + pcg64_normal(&s->rng, s->step_size_mol);
        double ny_ = oy + pcg64_normal(&s->rng, s->step_size_mol);
        nx_ = fmod(nx_, patch); if (nx_ < 0.0) nx_ += patch;
        ny_ = fmod(ny_, patch); if (ny_ < 0.0) ny_ += patch;

        double new_h = (double)height_at_pos_f(s->h, n, dx, nx_, ny_);
        int new_ix = (int)(nx_ / dx); if (new_ix >= n) new_ix = n - 1;
        int new_iy = (int)(ny_ / dx); if (new_iy >= n) new_iy = n - 1;
        int has_pmhc_new = (s->pmhc_count == NULL) || (s->pmhc_count[new_ix * n + new_iy] > 0);
        double new_e = has_pmhc_new ? tcr_pmhc_potential(new_h, s->u_assoc, SIGMA_BIND_NM) : 0.0;

        double dE = new_e - old_e;
        if (use_cell) {
            double old_pos2[2] = {ox, oy};
            double new_pos2[2] = {nx_, ny_};
            dE += mol_repulsion_delta(old_pos2, new_pos2, idx, tcr_cl,
                                      s->tcr_pos, s->mol_repulsion_eps,
                                      s->mol_repulsion_rcut, patch);
        }

        s->total_proposals++;
        double u = pcg64_uniform(&s->rng);
        if (dE <= 0.0 || (u > 0.0 && log(u) < -dE)) {
            s->accepted++;
            int old_cell = use_cell ? cell_list_cell(tcr_cl, ox, oy) : 0;
            s->tcr_pos[2 * idx] = nx_;
            s->tcr_pos[2 * idx + 1] = ny_;
            if (use_cell) {
                int new_cell = cell_list_cell(tcr_cl, nx_, ny_);
                cell_list_move(tcr_cl, idx, old_cell, new_cell);
            }
            /* Update binding state after accepted move. */
            if (s->binding_mode == 1 && s->tcr_bound && s->pmhc_count) {
                s->tcr_bound[idx] = (s->pmhc_count[new_ix * n + new_iy] > 0) ? 1 : 0;
                if (s->tcr_bound[idx]) {
                    s->h[new_ix * n + new_iy] = (float)s->h0_tcr;
                }
            }
        }
    }

    /* CD45 molecules. */
    for (int idx = 0; idx < s->n_cd45; idx++) {
        double ox = s->cd45_pos[2 * idx];
        double oy = s->cd45_pos[2 * idx + 1];
        double old_h = (double)height_at_pos_f(s->h, n, dx, ox, oy);
        double old_e = cd45_repulsion(old_h, s->cd45_height, s->k_rep);

        double nx_ = ox + pcg64_normal(&s->rng, s->step_size_mol);
        double ny_ = oy + pcg64_normal(&s->rng, s->step_size_mol);
        nx_ = fmod(nx_, patch); if (nx_ < 0.0) nx_ += patch;
        ny_ = fmod(ny_, patch); if (ny_ < 0.0) ny_ += patch;

        double new_h = (double)height_at_pos_f(s->h, n, dx, nx_, ny_);
        double new_e = cd45_repulsion(new_h, s->cd45_height, s->k_rep);

        double dE = new_e - old_e;
        if (use_cell) {
            double old_pos2[2] = {ox, oy};
            double new_pos2[2] = {nx_, ny_};
            dE += mol_repulsion_delta(old_pos2, new_pos2, idx, cd45_cl,
                                      s->cd45_pos, s->mol_repulsion_eps,
                                      s->mol_repulsion_rcut, patch);
        }

        s->total_proposals++;
        double u = pcg64_uniform(&s->rng);
        if (dE <= 0.0 || (u > 0.0 && log(u) < -dE)) {
            s->accepted++;
            int old_cell = use_cell ? cell_list_cell(cd45_cl, ox, oy) : 0;
            s->cd45_pos[2 * idx] = nx_;
            s->cd45_pos[2 * idx + 1] = ny_;
            if (use_cell) {
                int new_cell = cell_list_cell(cd45_cl, nx_, ny_);
                cell_list_move(cd45_cl, idx, old_cell, new_cell);
            }
        }
    }
}

static void bin_molecules(const double *pos, int n_mol, int n, double dx,
                          int *count_grid) {
    memset(count_grid, 0, n * n * sizeof(int));
    for (int m = 0; m < n_mol; m++) {
        int ix = (int)(pos[2 * m] / dx);
        int iy = (int)(pos[2 * m + 1] / dx);
        if (ix < 0) ix = 0; if (ix >= n) ix = n - 1;
        if (iy < 0) iy = 0; if (iy >= n) iy = n - 1;
        count_grid[ix * n + iy]++;
    }
}

/* Build frozen cell mask for forced binding. */
static void build_frozen_mask(SimState *s, int *frozen) {
    int n = s->grid_size;
    memset(frozen, 0, n * n * sizeof(int));
    if (s->binding_mode != 1 || !s->tcr_bound || !s->pmhc_count) return;
    for (int i = 0; i < s->n_tcr; i++) {
        if (!s->tcr_bound[i]) continue;
        int gi = (int)(s->tcr_pos[2 * i] / s->dx);
        if (gi >= n) gi = n - 1;
        int gj = (int)(s->tcr_pos[2 * i + 1] / s->dx);
        if (gj >= n) gj = n - 1;
        frozen[gi * n + gj] = 1;
        s->h[gi * n + gj] = (float)s->h0_tcr;
    }
}

/* Float-precision physics shared between CPU Phase 2 and GPU shaders. */
#include "ks_physics.h"

static void phase2_grid_cpu(SimState *s) {
    int n = s->grid_size;
    int n2 = n * n;
    float dx = (float)s->dx;
    float kappa = (float)s->kappa;
    int half = n2 / 2;

    int *tcr_count = (int *)malloc(n2 * sizeof(int));
    int *cd45_count = (int *)malloc(n2 * sizeof(int));
    bin_molecules(s->tcr_pos, s->n_tcr, n, s->dx, tcr_count);
    bin_molecules(s->cd45_pos, s->n_cd45, n, s->dx, cd45_count);

    /* Build frozen cell mask. */
    int *frozen = (int *)calloc(n2, sizeof(int));
    build_frozen_mask(s, frozen);

    /* Per-cell buffers for the two-pass approach (no snapshot needed —
       checkerboard ensures same-color cells are never Laplacian neighbors). */
    float *old_vals = (float *)malloc(half * sizeof(float));
    float *u_accepts = (float *)malloc(half * sizeof(float));
    int *cell_gi = (int *)malloc(half * sizeof(int));
    int *cell_gj = (int *)malloc(half * sizeof(int));

    for (int color = 0; color < 2; color++) {
        int cidx = 0;

        /* Pass 1 (propose): generate proposals, write ALL to h[]. */
        for (int gi = 0; gi < n; gi++) {
            for (int gj = 0; gj < n; gj++) {
                if ((gi + gj) % 2 != color) continue;

                float old_h_val = s->h[gi * n + gj];

                if (frozen[gi * n + gj]) {
                    /* Frozen cell: consume RNG to stay in sync. */
                    pcg64_uniform_f(&s->rng);
                    pcg64_uniform_f(&s->rng);
                    pcg64_uniform_f(&s->rng);

                    old_vals[cidx] = old_h_val;
                    u_accepts[cidx] = 0.0f;
                    cell_gi[cidx] = gi;
                    cell_gj[cidx] = gj;
                    cidx++;

                    s->total_proposals++;
                    s->accepted++;
                    continue;
                }

                /* Float32 Box-Muller matching GPU shader precision. */
                float u1_f = pcg64_uniform_f(&s->rng);
                if (u1_f < 1e-30f) u1_f = 1e-30f;
                float u2_f = pcg64_uniform_f(&s->rng);
                float normal_f = (float)s->step_size_h
                               * sqrtf(-2.0f * logf(u1_f))
                               * cosf(6.2831853071795864f * u2_f);
                float new_h_val = old_h_val + normal_f;
                if (new_h_val < 0.0f) new_h_val = -new_h_val;

                float u_f = pcg64_uniform_f(&s->rng);

                old_vals[cidx] = old_h_val;
                u_accepts[cidx] = u_f;
                cell_gi[cidx] = gi;
                cell_gj[cidx] = gj;
                cidx++;

                s->h[gi * n + gj] = new_h_val;
            }
        }

        /* Pass 2 (evaluate + apply): read directly from h[] — no snapshot.
           Same-color cells are never neighbors in the 5-point Laplacian stencil,
           so each cell reads only opposite-color neighbors (unchanged). */
        for (int k = 0; k < cidx; k++) {
            int gi = cell_gi[k];
            int gj = cell_gj[k];

            if (frozen[gi * n + gj]) {
                continue;  /* already counted above */
            }

            float old_h_val = old_vals[k];
            float new_h_val = s->h[gi * n + gj];

            int n_tcr_cell = tcr_count[gi * n + gj];
            int n_cd45_cell = cd45_count[gi * n + gj];
            int cell_has_pmhc = (s->pmhc_count == NULL) || (s->pmhc_count[gi * n + gj] > 0);

            float tcr_e_old = cell_has_pmhc ? n_tcr_cell * ks_tcr_potential(old_h_val, (float)s->u_assoc, (float)SIGMA_BIND_NM) : 0.0f;
            float old_mol_e = tcr_e_old
                            + n_cd45_cell * ks_cd45_repulsion(old_h_val, (float)s->cd45_height, (float)s->k_rep);

            float dE_bend = ks_bending_delta(s->h, n, kappa, dx,
                                                   gi, gj, old_h_val, new_h_val);
            float tcr_e_new = cell_has_pmhc ? n_tcr_cell * ks_tcr_potential(new_h_val, (float)s->u_assoc, (float)SIGMA_BIND_NM) : 0.0f;
            float new_mol_e = tcr_e_new
                            + n_cd45_cell * ks_cd45_repulsion(new_h_val, (float)s->cd45_height, (float)s->k_rep);

            float dE = dE_bend + (new_mol_e - old_mol_e);
            s->total_proposals++;
            float u_f = u_accepts[k];
            if (dE <= 0.0f || (u_f > 0.0f && logf(u_f) < -dE)) {
                s->accepted++;
            } else {
                s->h[gi * n + gj] = old_h_val;
            }
        }
    } /* end color loop */

    free(old_vals);
    free(u_accepts);
    free(cell_gi);
    free(cell_gj);
    free(tcr_count);
    free(cd45_count);
    free(frozen);
}

void sim_step(SimState *s) {
    int K = s->grid_substeps > 1 ? s->grid_substeps : 1;
#ifdef KS_PROFILE
    double t0 = _clock_ms();
#endif
    phase1_molecules(s);
#ifdef KS_PROFILE
    double t1 = _clock_ms();
    _profile_phase1_ms += t1 - t0;
#endif
    if (s->use_gpu && s->metal_ctx) {
        /* GPU: all K substeps batched in one command buffer commit. */
        gpu_engine_grid_update(s->metal_ctx, s->h, s->grid_size,
                                 s->kappa, s->dx, s->step_size_h,
                                 s->u_assoc, SIGMA_BIND_NM, s->cd45_height,
                                 s->k_rep,
                                 s->tcr_pos, s->n_tcr,
                                 s->cd45_pos, s->n_cd45,
                                 s->pmhc_count,
                                 &s->accepted, &s->total_proposals,
                                 K);
    } else {
        /* CPU: loop K times. */
        for (int sub = 0; sub < K; sub++) {
            phase2_grid_cpu(s);
        }
    }
#ifdef KS_PROFILE
    double t2 = _clock_ms();
    _profile_phase2_ms += t2 - t1;
#endif
}

void sim_run(SimState *s, int n_steps) {
    s->n_steps = n_steps;
    for (int step = 0; step < n_steps; step++) {
        sim_step(s);
    }
}

double sim_depletion_width(const SimState *s) {
    double center = PATCH_SIZE_NM / 2.0;

    double *tcr_r = (double *)malloc(s->n_tcr * sizeof(double));
    double *cd45_r = (double *)malloc(s->n_cd45 * sizeof(double));

    for (int i = 0; i < s->n_tcr; i++) {
        double dx_ = s->tcr_pos[2 * i] - center;
        double dy_ = s->tcr_pos[2 * i + 1] - center;
        tcr_r[i] = sqrt(dx_ * dx_ + dy_ * dy_);
    }
    for (int i = 0; i < s->n_cd45; i++) {
        double dx_ = s->cd45_pos[2 * i] - center;
        double dy_ = s->cd45_pos[2 * i + 1] - center;
        cd45_r[i] = sqrt(dx_ * dx_ + dy_ * dy_);
    }

    for (int i = 0; i < s->n_tcr - 1; i++)
        for (int j = i + 1; j < s->n_tcr; j++)
            if (tcr_r[j] < tcr_r[i]) {
                double tmp = tcr_r[i]; tcr_r[i] = tcr_r[j]; tcr_r[j] = tmp;
            }
    for (int i = 0; i < s->n_cd45 - 1; i++)
        for (int j = i + 1; j < s->n_cd45; j++)
            if (cd45_r[j] < cd45_r[i]) {
                double tmp = cd45_r[i]; cd45_r[i] = cd45_r[j]; cd45_r[j] = tmp;
            }

    double tcr_median = tcr_r[s->n_tcr / 2];
    double cd45_median = cd45_r[s->n_cd45 / 2];

    free(tcr_r);
    free(cd45_r);

    double w = cd45_median - tcr_median;
    return w > 0.0 ? w : 0.0;
}

double sim_mean_r(const double *pos, int n) {
    double center = PATCH_SIZE_NM / 2.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double dx_ = pos[2 * i] - center;
        double dy_ = pos[2 * i + 1] - center;
        sum += sqrt(dx_ * dx_ + dy_ * dy_);
    }
    return sum / n;
}

int sim_count_bound_tcr(const SimState *s) {
    if (!s->tcr_bound) return 0;
    int count = 0;
    for (int i = 0; i < s->n_tcr; i++) {
        if (s->tcr_bound[i]) count++;
    }
    return count;
}
