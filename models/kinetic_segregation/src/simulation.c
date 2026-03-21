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

/* Forward declarations. */
static void bin_molecules(const double *pos, int n_mol, int n, double dx,
                          int *count_grid);
static void compute_pmhc_influence(SimState *s);
static double pmhc_influence_at(const SimState *s, double x, double y);
static void calibrate_dt(SimState *s, double D_mol, double D_h);

/* ------------------------------------------------------------------ */
/*  Initialization helpers                                             */
/* ------------------------------------------------------------------ */

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

/* Set struct fields from arguments, apply defaults for zero-valued params. */
static void init_params(SimState *s, int grid_size, int n_tcr, int n_cd45,
                        double kappa, double u_assoc,
                        double cd45_height, double k_rep,
                        double mol_repulsion_eps, double mol_repulsion_rcut,
                        int binding_mode, int step_mode,
                        double h0_tcr, double init_height,
                        double sigma_r, double sigma_bind, double patch_size,
                        int n_pmhc, int pmhc_mode, double pmhc_radius,
                        double D_mol, double D_h) {
    s->grid_size = grid_size;
    s->sigma_r = (sigma_r > 0.0) ? sigma_r : SIGMA_R_DEFAULT;
    s->sigma_bind = (sigma_bind > 0.0) ? sigma_bind : SIGMA_BIND_NM;
    s->patch_size = (patch_size > 0.0) ? patch_size : PATCH_SIZE_NM;
    s->pmhc_influence = NULL;
    s->dx = s->patch_size / grid_size;
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
    s->mol_repulsion_rcut = (mol_repulsion_rcut > 0.0) ? mol_repulsion_rcut : MOL_RCUT_DEFAULT;
    s->n_pmhc = n_pmhc;
    s->pmhc_pos = NULL;
    s->pmhc_count = NULL;
    s->pmhc_mode = pmhc_mode;
    s->pmhc_radius = pmhc_radius;
    s->tcr_bound = NULL;

    /* Apply defaults if zero. */
    s->D_mol = (D_mol > 0.0) ? D_mol : D_MOL_DEFAULT;
    s->D_h = (D_h > 0.0) ? D_h : D_H_DEFAULT;

    /* Spring constant: paper formula or explicit. */
    if (k_rep > 0.0) {
        s->k_rep = k_rep;
    } else if (step_mode == STEP_MODE_PAPER) {
        s->k_rep = K_REP_PAPER_FACTOR * kappa / (s->dx * s->dx);
    } else {
        s->k_rep = K_REP_BROWNIAN;
    }
}

/* Compute dt from stability + force-field constraints, apply user override. */
static void init_dt(SimState *s, double dt_override, double dt_factor) {
    double D_mol = s->D_mol;
    double D_h = s->D_h;

    /* Stability-constrained dt (membrane height dynamics). */
    double dt_stable = (s->dx * s->dx) / (2.0 * D_h * s->kappa);
    s->dt = dt_stable * DT_SAFETY;
    s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
    s->step_size_h = sqrt(2.0 * D_h * s->dt);

    /* Force-field calibration may reduce dt further. */
    calibrate_dt(s, D_mol, D_h);

    /* Paper mode: fixed height step (independent of dt). */
    if (s->step_mode == STEP_MODE_PAPER) {
        s->step_size_h = STEP_H_PAPER;
    }

    /* Store the physics-derived dt before any user override. */
    s->dt_auto = s->dt;
    s->dt_factor = dt_factor;

    /* Apply user override (mutually exclusive). */
    if (dt_override > 0.0) {
        s->dt = dt_override;
        s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
        if (s->step_mode != STEP_MODE_PAPER)
            s->step_size_h = sqrt(2.0 * D_h * s->dt);
    } else if (dt_factor > 0.0) {
        s->dt = s->dt_auto * dt_factor;
        s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
        if (s->step_mode != STEP_MODE_PAPER)
            s->step_size_h = sqrt(2.0 * D_h * s->dt);
    }

    /* Diagnostic: always report auto-cal result. */
    fprintf(stderr, "AUTO-DT: dt_auto=%.4g s (step=%.2f nm)",
            s->dt_auto, sqrt(2.0 * D_mol * s->dt_auto));
    if (dt_override > 0.0)
        fprintf(stderr, " → overridden to dt=%.4g s (--dt)", s->dt);
    else if (dt_factor > 0.0)
        fprintf(stderr, " → scaled to dt=%.4g s (--dt_factor %.3g)",
                s->dt, dt_factor);
    fprintf(stderr, "\n");
}

/* Allocate and place molecules (pMHC, TCR, CD45), precompute fields. */
static void init_molecules(SimState *s, uint64_t seed,
                           uint64_t pmhc_seed) {
    int grid_size = s->grid_size;

    s->h = (float *)malloc(grid_size * grid_size * sizeof(float));
    s->tcr_pos = (double *)malloc(s->n_tcr * 2 * sizeof(double));
    s->cd45_pos = (double *)malloc(s->n_cd45 * 2 * sizeof(double));

    pcg64_seed(&s->rng, seed);
    init_height_field(s);

    /* pMHC initialization (before TCR so TCR can co-locate).
     * If n_pmhc == 0, auto-compute from paper density (300/µm²). */
    double eff_radius = (s->pmhc_radius > 0.0)
        ? s->pmhc_radius
        : s->patch_size * PMHC_RADIUS_FRAC_DEFAULT;

    if (s->n_pmhc == 0) {
        /* Auto-compute from PMHC_DENSITY_PER_UM2. */
        double area_nm2;
        if (s->pmhc_mode == PMHC_MODE_INNER_CIRCLE) {
            area_nm2 = M_PI * eff_radius * eff_radius;
        } else {
            area_nm2 = s->patch_size * s->patch_size;
        }
        double area_um2 = area_nm2 / 1e6;
        s->n_pmhc = (int)(PMHC_DENSITY_PER_UM2 * area_um2 + 0.5);
        if (s->n_pmhc < 1) s->n_pmhc = 1;
        fprintf(stderr, "AUTO-PMHC: n_pmhc=%d (density=%.0f/µm², area=%.0f nm²)\n",
                s->n_pmhc, PMHC_DENSITY_PER_UM2, area_nm2);
    }

    if (s->n_pmhc > 0) {
        s->pmhc_pos = (double *)malloc(s->n_pmhc * 2 * sizeof(double));
        pcg64_t pmhc_rng;
        pcg64_seed(&pmhc_rng, pmhc_seed);
        double center_xy = s->patch_size / 2.0;

        if (s->pmhc_mode == PMHC_MODE_INNER_CIRCLE) {
            /* inner_circle: rejection sampling within centered disc */
            int placed = 0;
            while (placed < s->n_pmhc) {
                double cx = pcg64_uniform(&pmhc_rng) * s->patch_size;
                double cy = pcg64_uniform(&pmhc_rng) * s->patch_size;
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
            for (int i = 0; i < s->n_pmhc; i++) {
                s->pmhc_pos[2 * i] = pcg64_uniform(&pmhc_rng) * s->patch_size;
                s->pmhc_pos[2 * i + 1] = pcg64_uniform(&pmhc_rng) * s->patch_size;
            }
        }

        /* Always allocate pmhc_count when pMHC present (needed for rendering
         * and forced-mode gating). Gaussian mode uses pmhc_influence instead. */
        s->pmhc_count = (int *)calloc(grid_size * grid_size, sizeof(int));
        bin_molecules(s->pmhc_pos, s->n_pmhc, grid_size, s->dx, s->pmhc_count);
    }

    /* TCR initialization. */
    if (s->n_pmhc > 0 && s->pmhc_pos) {
        /* Co-locate TCR on pMHC positions with small jitter (sigma_bind) */
        for (int i = 0; i < s->n_tcr; i++) {
            int pidx = (int)(pcg64_uniform(&s->rng) * s->n_pmhc);
            if (pidx >= s->n_pmhc) pidx = s->n_pmhc - 1;
            double tx = s->pmhc_pos[2 * pidx] + pcg64_normal(&s->rng, s->sigma_bind);
            double ty = s->pmhc_pos[2 * pidx + 1] + pcg64_normal(&s->rng, s->sigma_bind);
            tx = fmod(tx, s->patch_size); if (tx < 0.0) tx += s->patch_size;
            ty = fmod(ty, s->patch_size); if (ty < 0.0) ty += s->patch_size;
            s->tcr_pos[2 * i] = tx;
            s->tcr_pos[2 * i + 1] = ty;
        }
    } else {
        /* Backward compat: center-biased Gaussian */
        init_positions(s->tcr_pos, s->n_tcr, s->patch_size, &s->rng, 1);
    }

    /* CD45: always uniform */
    init_positions(s->cd45_pos, s->n_cd45, s->patch_size, &s->rng, 0);

    /* Precompute pMHC influence field for gaussian binding mode. */
    if (s->binding_mode == BINDING_MODE_GAUSSIAN) {
        if (s->n_pmhc > 0) {
            compute_pmhc_influence(s);
        } else {
            /* No pMHC in gaussian mode: uniform weight = 1.0 everywhere. */
            s->pmhc_influence = (float *)malloc(grid_size * grid_size * sizeof(float));
            for (int i = 0; i < grid_size * grid_size; i++)
                s->pmhc_influence[i] = 1.0f;
        }
    }

    /* TCR binding state (forced mode). */
    if (s->binding_mode == BINDING_MODE_FORCED && s->pmhc_count) {
        s->tcr_bound = (int *)calloc(s->n_tcr, sizeof(int));
        init_binding_state(s);
    }

    s->accepted = 0;
    s->total_proposals = 0;

    /* Allocate persistent cell lists if repulsion is enabled. */
    s->tcr_cell_list = NULL;
    s->cd45_cell_list = NULL;
    if (s->mol_repulsion_eps > 0.0 && s->mol_repulsion_rcut > 0.0) {
        CellList *tcl = (CellList *)malloc(sizeof(CellList));
        cell_list_init(tcl, s->n_tcr, s->mol_repulsion_rcut, s->patch_size);
        s->tcr_cell_list = tcl;
        CellList *ccl = (CellList *)malloc(sizeof(CellList));
        cell_list_init(ccl, s->n_cd45, s->mol_repulsion_rcut, s->patch_size);
        s->cd45_cell_list = ccl;
    }
}

/* Initialize GPU engine if requested. */
static void init_gpu(SimState *s, int use_gpu, uint64_t seed) {
    s->use_gpu = 0;
    s->metal_ctx = NULL;
    s->h_is_shared = 0;
    s->grid_substeps = 1;
    if (use_gpu) {
        /* Use a deterministic key derived from the seed for GPU Philox RNG. */
        uint64_t gpu_key = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
        s->metal_ctx = gpu_engine_create(s->grid_size, gpu_key);
        if (s->metal_ctx) {
            s->use_gpu = 1;
            /* Move h to the Metal shared buffer so CPU and GPU share memory
               directly (no per-step memcpy needed on unified memory). */
            float *shared_h = gpu_engine_h_ptr(s->metal_ctx);
            memcpy(shared_h, s->h, s->grid_size * s->grid_size * sizeof(float));
            free(s->h);
            s->h = shared_h;
            s->h_is_shared = 1;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Public: create / destroy                                           */
/* ------------------------------------------------------------------ */

SimState *sim_create(int grid_size, int n_tcr, int n_cd45,
                     double kappa, double u_assoc, uint64_t seed,
                     int use_gpu,
                     double D_mol, double D_h, double dt_override,
                     double dt_factor,
                     double cd45_height, double k_rep,
                     double mol_repulsion_eps, double mol_repulsion_rcut,
                     int n_pmhc, uint64_t pmhc_seed,
                     int pmhc_mode, double pmhc_radius,
                     int binding_mode, int step_mode,
                     double h0_tcr, double init_height,
                     double sigma_r, double sigma_bind, double patch_size) {
    SimState *s = (SimState *)calloc(1, sizeof(SimState));
    init_params(s, grid_size, n_tcr, n_cd45, kappa, u_assoc,
                cd45_height, k_rep, mol_repulsion_eps, mol_repulsion_rcut,
                binding_mode, step_mode, h0_tcr, init_height,
                sigma_r, sigma_bind, patch_size,
                n_pmhc, pmhc_mode, pmhc_radius, D_mol, D_h);
    init_dt(s, dt_override, dt_factor);
    init_molecules(s, seed, pmhc_seed);
    init_gpu(s, use_gpu, seed);
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
    free(s->pmhc_influence);
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

/* ------------------------------------------------------------------ */
/*  Phase 1: molecular Metropolis MC                                   */
/* ------------------------------------------------------------------ */

static float height_at_pos_f(const float *h, int n, double dx, double x, double y) {
    int ix = (int)(x / dx);
    int iy = (int)(y / dx);
    if (ix < 0) ix = 0;
    if (ix >= n) ix = n - 1;
    if (iy < 0) iy = 0;
    if (iy >= n) iy = n - 1;
    return h[ix * n + iy];
}

/* Compute TCR energy at position (x,y). */
static double tcr_energy_at(const SimState *s, double x, double y, double h_val) {
    if (s->pmhc_influence) {
        double w = pmhc_influence_at(s, x, y);
        return w * tcr_pmhc_potential(h_val, s->h0_tcr, s->u_assoc, s->sigma_bind);
    }
    int n = s->grid_size;
    int ix = (int)(x / s->dx); if (ix >= n) ix = n - 1;
    int iy = (int)(y / s->dx); if (iy >= n) iy = n - 1;
    int has_pmhc = (s->pmhc_count == NULL) || (s->pmhc_count[ix * n + iy] > 0);
    return has_pmhc ? tcr_pmhc_potential(h_val, s->h0_tcr, s->u_assoc, s->sigma_bind) : 0.0;
}

static void phase1_molecules(SimState *s) {
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
        if (s->binding_mode == BINDING_MODE_FORCED && s->tcr_bound && s->tcr_bound[idx])
            continue;

        double ox = s->tcr_pos[2 * idx];
        double oy = s->tcr_pos[2 * idx + 1];
        double old_h = (double)height_at_pos_f(s->h, n, dx, ox, oy);
        double old_e = tcr_energy_at(s, ox, oy, old_h);

        /* Propose displacement — need proposed position to compute new energy. */
        double nx_, ny_;
        /* We need the proposed position before calling propose_and_accept,
         * so generate it inline here (matching the RNG sequence). */
        nx_ = ox + pcg64_normal(&s->rng, s->step_size_mol);
        ny_ = oy + pcg64_normal(&s->rng, s->step_size_mol);
        nx_ = fmod(nx_, s->patch_size); if (nx_ < 0.0) nx_ += s->patch_size;
        ny_ = fmod(ny_, s->patch_size); if (ny_ < 0.0) ny_ += s->patch_size;

        double new_h = (double)height_at_pos_f(s->h, n, dx, nx_, ny_);
        double new_e = tcr_energy_at(s, nx_, ny_, new_h);
        double dE = new_e - old_e;

        if (use_cell) {
            double old_pos2[2] = {ox, oy};
            double new_pos2[2] = {nx_, ny_};
            dE += mol_repulsion_delta(old_pos2, new_pos2, idx, tcr_cl,
                                      s->tcr_pos, s->mol_repulsion_eps,
                                      s->mol_repulsion_rcut, s->patch_size);
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
            if (s->binding_mode == BINDING_MODE_FORCED && s->tcr_bound && s->pmhc_count) {
                int new_ix = (int)(nx_ / dx); if (new_ix >= n) new_ix = n - 1;
                int new_iy = (int)(ny_ / dx); if (new_iy >= n) new_iy = n - 1;
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

        double nx_, ny_;
        nx_ = ox + pcg64_normal(&s->rng, s->step_size_mol);
        ny_ = oy + pcg64_normal(&s->rng, s->step_size_mol);
        nx_ = fmod(nx_, s->patch_size); if (nx_ < 0.0) nx_ += s->patch_size;
        ny_ = fmod(ny_, s->patch_size); if (ny_ < 0.0) ny_ += s->patch_size;

        double new_h = (double)height_at_pos_f(s->h, n, dx, nx_, ny_);
        double new_e = cd45_repulsion(new_h, s->cd45_height, s->k_rep);
        double dE = new_e - old_e;

        if (use_cell) {
            double old_pos2[2] = {ox, oy};
            double new_pos2[2] = {nx_, ny_};
            dE += mol_repulsion_delta(old_pos2, new_pos2, idx, cd45_cl,
                                      s->cd45_pos, s->mol_repulsion_eps,
                                      s->mol_repulsion_rcut, s->patch_size);
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

/* ------------------------------------------------------------------ */
/*  Molecule binning + force-field helpers                             */
/* ------------------------------------------------------------------ */

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

/* Auto-calibrate dt so step_size_mol resolves the smallest force-field scale.
 * Always called during initialization to compute dt_auto.
 * Considers: sigma_r (gaussian binding), mol_repulsion_rcut (excluded vol). */
static void calibrate_dt(SimState *s, double D_mol, double D_h) {
    double min_scale = s->dx;  /* grid cell size as upper bound */

    /* Gaussian binding: molecular step must resolve sigma_r */
    if (s->n_pmhc > 0 && s->binding_mode == BINDING_MODE_GAUSSIAN && s->sigma_r > 0.0)
        if (s->sigma_r < min_scale)
            min_scale = s->sigma_r;

    /* Excluded volume: step should resolve the repulsion cutoff */
    if (s->mol_repulsion_eps > 0.0 && s->mol_repulsion_rcut > 0.0)
        if (s->mol_repulsion_rcut < min_scale)
            min_scale = s->mol_repulsion_rcut;

    /* Target step = min_scale / 2 */
    double target_step = min_scale / 2.0;
    double dt_force = (target_step * target_step) / (2.0 * D_mol);

    if (s->dt > dt_force) {
        fprintf(stderr, "AUTO-DT: step_size_mol=%.2f nm exceeds force-field "
                "scale=%.2f nm; reducing dt %.4g -> %.4g s "
                "(step %.2f -> %.2f nm)\n",
                s->step_size_mol, min_scale,
                s->dt, dt_force,
                s->step_size_mol, target_step);
        s->dt = dt_force;
        s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
        s->step_size_h = sqrt(2.0 * D_h * s->dt);
    }
}

/* Precompute pMHC influence field: W(i,j) = clamp(sum_k exp(-r_k^2 / (2*sigma_r^2)), 0, 1). */
static void compute_pmhc_influence(SimState *s) {
    int n = s->grid_size;
    double dx = s->dx;
    double sigma_r = s->sigma_r;
    double inv_2sigma2 = 1.0 / (2.0 * sigma_r * sigma_r);
    double r_cut = PMHC_CUTOFF_SIGMAS * sigma_r;
    double r_cut2 = r_cut * r_cut;
    double patch = s->patch_size;
    double half_patch = patch / 2.0;

    s->pmhc_influence = (float *)calloc(n * n, sizeof(float));

    for (int i = 0; i < n; i++) {
        double cx = (i + 0.5) * dx;
        for (int j = 0; j < n; j++) {
            double cy = (j + 0.5) * dx;
            double sum = 0.0;
            for (int k = 0; k < s->n_pmhc; k++) {
                double ddx = cx - s->pmhc_pos[2 * k];
                double ddy = cy - s->pmhc_pos[2 * k + 1];
                if (ddx > half_patch) ddx -= patch;
                else if (ddx < -half_patch) ddx += patch;
                if (ddy > half_patch) ddy -= patch;
                else if (ddy < -half_patch) ddy += patch;
                double r2 = ddx * ddx + ddy * ddy;
                if (r2 < r_cut2) {
                    sum += exp(-r2 * inv_2sigma2);
                }
            }
            s->pmhc_influence[i * n + j] = (float)(sum > 1.0 ? 1.0 : sum);
        }
    }
}

/* Compute exact pMHC influence at a continuous (x,y) position.
 * Used in phase1 molecular moves where TCR positions are continuous,
 * providing smooth lateral decay even when sigma_r < grid cell size. */
static double pmhc_influence_at(const SimState *s, double x, double y) {
    double sigma_r = s->sigma_r;
    double inv2sig2 = 1.0 / (2.0 * sigma_r * sigma_r);
    double rcut = PMHC_CUTOFF_SIGMAS * sigma_r;
    double rcut2 = rcut * rcut;
    double p = s->patch_size;
    double half = p / 2.0;
    double sum = 0.0;
    for (int k = 0; k < s->n_pmhc; k++) {
        double ddx = x - s->pmhc_pos[2 * k];
        double ddy = y - s->pmhc_pos[2 * k + 1];
        if (ddx > half) ddx -= p;
        else if (ddx < -half) ddx += p;
        if (ddy > half) ddy -= p;
        else if (ddy < -half) ddy += p;
        double r2 = ddx * ddx + ddy * ddy;
        if (r2 < rcut2)
            sum += exp(-r2 * inv2sig2);
    }
    return sum > 1.0 ? 1.0 : sum;
}

/* ------------------------------------------------------------------ */
/*  Phase 2: grid Metropolis MC (CPU path)                             */
/* ------------------------------------------------------------------ */

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
                if (u1_f < BOX_MULLER_FLOOR) u1_f = BOX_MULLER_FLOOR;
                float u2_f = pcg64_uniform_f(&s->rng);
                float normal_f = (float)s->step_size_h
                               * sqrtf(-2.0f * logf(u1_f))
                               * cosf(TWO_PI_F * u2_f);
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

            float w_bind;
            if (s->pmhc_influence) {
                w_bind = s->pmhc_influence[gi * n + gj];
            } else {
                w_bind = ((s->pmhc_count == NULL) || (s->pmhc_count[gi * n + gj] > 0)) ? 1.0f : 0.0f;
            }

            float tcr_e_old = w_bind * n_tcr_cell * ks_tcr_potential(old_h_val, (float)s->h0_tcr, (float)s->u_assoc, (float)s->sigma_bind);
            float old_mol_e = tcr_e_old
                            + n_cd45_cell * ks_cd45_repulsion(old_h_val, (float)s->cd45_height, (float)s->k_rep);

            float dE_bend = ks_bending_delta(s->h, n, kappa, dx,
                                                   gi, gj, old_h_val, new_h_val);
            float tcr_e_new = w_bind * n_tcr_cell * ks_tcr_potential(new_h_val, (float)s->h0_tcr, (float)s->u_assoc, (float)s->sigma_bind);
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

/* ------------------------------------------------------------------ */
/*  Public: step / run                                                 */
/* ------------------------------------------------------------------ */

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
                                 s->u_assoc, s->sigma_bind, s->h0_tcr,
                                 s->cd45_height, s->k_rep,
                                 s->tcr_pos, s->n_tcr,
                                 s->cd45_pos, s->n_cd45,
                                 s->pmhc_count,
                                 s->pmhc_influence,
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

/* ------------------------------------------------------------------ */
/*  Depletion metrics                                                  */
/* ------------------------------------------------------------------ */

/* Minimum-image distance helper (same as potentials.c _mic). */
static inline double _mic_sim(double d, double half_patch, double patch_size) {
    if (d > half_patch) return d - patch_size;
    if (d < -half_patch) return d + patch_size;
    return d;
}

/* Bubble sort (small N, deterministic across platforms). */
static void _bsort(double *a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) {
                double tmp = a[i]; a[i] = a[j]; a[j] = tmp;
            }
}

/* Linearly interpolated percentile on sorted array of length n. */
static double _percentile(const double *sorted, int n, double p) {
    double idx = p * (n - 1) / 100.0;
    int lo = (int)idx;
    if (lo >= n - 1) return sorted[n - 1];
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
}

/* Compute radial distances from patch center for a set of molecules. */
static void compute_radial_distances(const double *pos, int n_mol,
                                     double center, double *out_r) {
    for (int i = 0; i < n_mol; i++) {
        double dx_ = pos[2 * i] - center;
        double dy_ = pos[2 * i + 1] - center;
        out_r[i] = sqrt(dx_ * dx_ + dy_ * dy_);
    }
}

static double compute_overlap_coeff(const double *tcr_r, int nt,
                                    const double *cd45_r, int nc,
                                    double patch_size) {
    double r_max = patch_size * DIAG_HALF_FACTOR;
    double bin_w = r_max / OVERLAP_NBINS;
    double *h_tcr = (double *)calloc(OVERLAP_NBINS, sizeof(double));
    double *h_cd45 = (double *)calloc(OVERLAP_NBINS, sizeof(double));

    for (int i = 0; i < nt; i++) {
        int b = (int)(tcr_r[i] / bin_w);
        if (b >= OVERLAP_NBINS) b = OVERLAP_NBINS - 1;
        h_tcr[b] += 1.0;
    }
    for (int i = 0; i < nc; i++) {
        int b = (int)(cd45_r[i] / bin_w);
        if (b >= OVERLAP_NBINS) b = OVERLAP_NBINS - 1;
        h_cd45[b] += 1.0;
    }
    /* Normalize to density. */
    for (int i = 0; i < OVERLAP_NBINS; i++) {
        h_tcr[i] /= (nt * bin_w);
        h_cd45[i] /= (nc * bin_w);
    }
    double overlap = 0.0;
    for (int i = 0; i < OVERLAP_NBINS; i++) {
        double m = (h_tcr[i] < h_cd45[i]) ? h_tcr[i] : h_cd45[i];
        overlap += m * bin_w;
    }
    free(h_tcr);
    free(h_cd45);
    return overlap;
}

static double compute_ks_statistic(const double *tcr_r, int nt,
                                   const double *cd45_r, int nc) {
    int it = 0, ic = 0;
    double max_d = 0.0;
    while (it < nt || ic < nc) {
        double ft = (double)it / nt;
        double fc = (double)ic / nc;
        double d = ft - fc;
        if (d < 0) d = -d;
        if (d > max_d) max_d = d;

        if (it < nt && (ic >= nc || tcr_r[it] <= cd45_r[ic]))
            it++;
        else
            ic++;
    }
    return max_d;
}

static double compute_frontier_nn_gap(const SimState *s,
                                      const double *tcr_r, int nt,
                                      const double *cd45_r, int nc,
                                      double tcr_p75, double cd45_p25) {
    double half_patch = s->patch_size / 2.0;
    int n_ft = 0, n_fc = 0;
    for (int i = 0; i < nt; i++)
        if (tcr_r[i] > tcr_p75) n_ft++;
    for (int i = 0; i < nc; i++)
        if (cd45_r[i] < cd45_p25) n_fc++;

    if (n_ft == 0 || n_fc == 0) return 0.0;

    int *ft_idx = (int *)malloc(n_ft * sizeof(int));
    int *fc_idx = (int *)malloc(n_fc * sizeof(int));
    int fi = 0, ci = 0;
    for (int i = 0; i < nt; i++)
        if (tcr_r[i] > tcr_p75) ft_idx[fi++] = i;
    for (int i = 0; i < nc; i++)
        if (cd45_r[i] < cd45_p25) fc_idx[ci++] = i;

    double *nn_dist = (double *)malloc(n_ft * sizeof(double));
    for (int i = 0; i < n_ft; i++) {
        int ti = ft_idx[i];
        double tx = s->tcr_pos[2 * ti];
        double ty = s->tcr_pos[2 * ti + 1];
        double best = 1e18;
        for (int j = 0; j < n_fc; j++) {
            int cj = fc_idx[j];
            double dx_ = _mic_sim(tx - s->cd45_pos[2 * cj], half_patch, s->patch_size);
            double dy_ = _mic_sim(ty - s->cd45_pos[2 * cj + 1], half_patch, s->patch_size);
            double d2 = dx_ * dx_ + dy_ * dy_;
            if (d2 < best) best = d2;
        }
        nn_dist[i] = sqrt(best);
    }
    _bsort(nn_dist, n_ft);

    /* Trimmed median: exclude bottom/top 10%. */
    int lo = n_ft / 10;
    int hi = n_ft - 1 - n_ft / 10;
    if (lo > hi) { lo = 0; hi = n_ft - 1; }
    double result = nn_dist[(lo + hi) / 2];

    free(nn_dist);
    free(ft_idx);
    free(fc_idx);
    return result;
}

static double compute_cross_nn_median(const SimState *s, int nt) {
    int nc = s->n_cd45;
    double half_patch = s->patch_size / 2.0;
    double *nn_dist = (double *)malloc(nt * sizeof(double));
    for (int i = 0; i < nt; i++) {
        double tx = s->tcr_pos[2 * i];
        double ty = s->tcr_pos[2 * i + 1];
        double best = 1e18;
        for (int j = 0; j < nc; j++) {
            double dx_ = _mic_sim(tx - s->cd45_pos[2 * j], half_patch, s->patch_size);
            double dy_ = _mic_sim(ty - s->cd45_pos[2 * j + 1], half_patch, s->patch_size);
            double d2 = dx_ * dx_ + dy_ * dy_;
            if (d2 < best) best = d2;
        }
        nn_dist[i] = sqrt(best);
    }
    _bsort(nn_dist, nt);
    double result = nn_dist[nt / 2];
    free(nn_dist);
    return result;
}

/* Find bound TCRs using spatial proximity to pMHC (works in both binding modes).
 * Returns count of bound TCRs; fills bound_idx (caller-allocated, size n_tcr). */
static int find_bound_tcrs(const SimState *s, int *bound_idx) {
    if (s->bind_threshold <= 0.0 || !s->pmhc_pos || s->n_pmhc <= 0)
        return 0;
    double thr2 = s->bind_threshold * s->bind_threshold;
    double half = s->patch_size / 2.0;
    double ps = s->patch_size;
    int count = 0;
    for (int t = 0; t < s->n_tcr; t++) {
        double tx = s->tcr_pos[2 * t];
        double ty = s->tcr_pos[2 * t + 1];
        for (int p = 0; p < s->n_pmhc; p++) {
            double dx_ = tx - s->pmhc_pos[2 * p];
            double dy_ = ty - s->pmhc_pos[2 * p + 1];
            if (dx_ > half) dx_ -= ps;
            else if (dx_ < -half) dx_ += ps;
            if (dy_ > half) dy_ -= ps;
            else if (dy_ < -half) dy_ += ps;
            if (dx_ * dx_ + dy_ * dy_ < thr2) {
                bound_idx[count++] = t;
                break;
            }
        }
    }
    return count;
}

/* P10 of bound-TCR→nearest-CD45 distances. Returns -1 if n_bound == 0. */
static double compute_bound_tcr_cd45_nn_p10(const SimState *s,
                                             const int *bound_idx, int n_bound) {
    if (n_bound == 0) return -1.0;
    int nc = s->n_cd45;
    double half_patch = s->patch_size / 2.0;
    double *nn_dist = (double *)malloc(n_bound * sizeof(double));
    for (int i = 0; i < n_bound; i++) {
        int ti = bound_idx[i];
        double tx = s->tcr_pos[2 * ti];
        double ty = s->tcr_pos[2 * ti + 1];
        double best = 1e18;
        for (int j = 0; j < nc; j++) {
            double dx_ = _mic_sim(tx - s->cd45_pos[2 * j], half_patch, s->patch_size);
            double dy_ = _mic_sim(ty - s->cd45_pos[2 * j + 1], half_patch, s->patch_size);
            double d2 = dx_ * dx_ + dy_ * dy_;
            if (d2 < best) best = d2;
        }
        nn_dist[i] = sqrt(best);
    }
    _bsort(nn_dist, n_bound);
    double result = nn_dist[n_bound / 10];  /* P10: index floor(n*0.1) */
    free(nn_dist);
    return result;
}

/* P10 of CD45→nearest-bound-TCR distances. Returns -1 if n_bound == 0. */
static double compute_cd45_bound_tcr_nn_p10(const SimState *s,
                                             const int *bound_idx, int n_bound) {
    if (n_bound == 0) return -1.0;
    int nc = s->n_cd45;
    double half_patch = s->patch_size / 2.0;
    double *nn_dist = (double *)malloc(nc * sizeof(double));
    for (int j = 0; j < nc; j++) {
        double cx = s->cd45_pos[2 * j];
        double cy = s->cd45_pos[2 * j + 1];
        double best = 1e18;
        for (int i = 0; i < n_bound; i++) {
            int ti = bound_idx[i];
            double dx_ = _mic_sim(cx - s->tcr_pos[2 * ti], half_patch, s->patch_size);
            double dy_ = _mic_sim(cy - s->tcr_pos[2 * ti + 1], half_patch, s->patch_size);
            double d2 = dx_ * dx_ + dy_ * dy_;
            if (d2 < best) best = d2;
        }
        nn_dist[j] = sqrt(best);
    }
    _bsort(nn_dist, nc);
    double result = nn_dist[nc / 10];  /* P10 */
    free(nn_dist);
    return result;
}

double sim_depletion_width(const SimState *s) {
    double center = s->patch_size / 2.0;
    int nt = s->n_tcr, nc = s->n_cd45;

    double *tcr_r = (double *)malloc(nt * sizeof(double));
    double *cd45_r = (double *)malloc(nc * sizeof(double));
    compute_radial_distances(s->tcr_pos, nt, center, tcr_r);
    compute_radial_distances(s->cd45_pos, nc, center, cd45_r);
    _bsort(tcr_r, nt);
    _bsort(cd45_r, nc);

    double w = cd45_r[nc / 2] - tcr_r[nt / 2];
    free(tcr_r);
    free(cd45_r);
    return w > 0.0 ? w : 0.0;
}

DepletionMetrics sim_depletion_metrics(const SimState *s) {
    DepletionMetrics dm = {0};
    double center = s->patch_size / 2.0;
    int nt = s->n_tcr, nc = s->n_cd45;

    double *tcr_r = (double *)malloc(nt * sizeof(double));
    double *cd45_r = (double *)malloc(nc * sizeof(double));
    compute_radial_distances(s->tcr_pos, nt, center, tcr_r);
    compute_radial_distances(s->cd45_pos, nc, center, cd45_r);
    _bsort(tcr_r, nt);
    _bsort(cd45_r, nc);

    /* Metric 1: median_diff. */
    double tcr_med = tcr_r[nt / 2];
    double cd45_med = cd45_r[nc / 2];
    dm.median_diff = (cd45_med > tcr_med) ? cd45_med - tcr_med : 0.0;

    /* Metric 2: percentile_gap = P25(cd45) - P75(tcr). */
    double tcr_p75 = _percentile(tcr_r, nt, 75.0);
    double cd45_p25 = _percentile(cd45_r, nc, 25.0);
    dm.percentile_gap = cd45_p25 - tcr_p75;

    /* Metric 3: overlap coefficient. */
    dm.overlap_coeff = compute_overlap_coeff(tcr_r, nt, cd45_r, nc, s->patch_size);

    /* Metric 4: KS statistic. */
    dm.ks_statistic = compute_ks_statistic(tcr_r, nt, cd45_r, nc);

    /* Metric 5: frontier nearest-neighbor gap. */
    dm.frontier_nn_gap = compute_frontier_nn_gap(s, tcr_r, nt, cd45_r, nc,
                                                  tcr_p75, cd45_p25);

    /* Metric 6: cross-type nearest-neighbor median. */
    dm.cross_nn_median = compute_cross_nn_median(s, nt);

    /* Metrics 7-8: bound-TCR cross-NN P10 (geometry-free). */
    int *bound_idx = (int *)malloc(nt * sizeof(int));
    int n_bound = find_bound_tcrs(s, bound_idx);
    dm.bound_tcr_cd45_nn_p10 = compute_bound_tcr_cd45_nn_p10(s, bound_idx, n_bound);
    dm.cd45_bound_tcr_nn_p10 = compute_cd45_bound_tcr_nn_p10(s, bound_idx, n_bound);
    free(bound_idx);

    free(tcr_r);
    free(cd45_r);
    return dm;
}

/* ------------------------------------------------------------------ */
/*  Utility functions                                                  */
/* ------------------------------------------------------------------ */

double sim_mean_r(const SimState *s, const double *pos, int n) {
    double center = s->patch_size / 2.0;
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
