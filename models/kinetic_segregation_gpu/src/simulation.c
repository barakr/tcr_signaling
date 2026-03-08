#include "simulation.h"
#include "potentials.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Forward declarations for Metal engine (defined in metal_engine.m). */
extern void *metal_engine_create(int grid_size, uint64_t gpu_rng_key);
extern void metal_engine_destroy(void *ctx);
extern void metal_engine_grid_update(void *ctx, float *h, int grid_size,
                                     double kappa, double dx, double step_size_h,
                                     double u_assoc, double sigma_bind,
                                     double cd45_height,
                                     const double *tcr_pos, int n_tcr,
                                     const double *cd45_pos, int n_cd45,
                                     long *accepted, long *total_proposals);

static void init_height_field(SimState *s) {
    int n = s->grid_size;
    for (int i = 0; i < n * n; i++)
        s->h[i] = (float)CD45_HEIGHT_NM;

    /* Depress center to create initial tight-contact seed. */
    int center = n / 2;
    int radius = n / 8;
    if (radius < 1) radius = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int di = i - center;
            int dj = j - center;
            if (di * di + dj * dj <= radius * radius)
                s->h[i * n + j] = 5.0f;
        }
    }
}

static void init_positions(double *pos, int n, double patch, pcg64_t *rng,
                           int center_bias) {
    double center = patch / 2.0;
    double spread = patch / 6.0;
    for (int i = 0; i < n; i++) {
        if (center_bias) {
            pos[2 * i] = center + pcg64_normal(rng, spread);
            pos[2 * i + 1] = center + pcg64_normal(rng, spread);
            if (pos[2 * i] < 0.0) pos[2 * i] = 0.0;
            if (pos[2 * i] > patch) pos[2 * i] = patch;
            if (pos[2 * i + 1] < 0.0) pos[2 * i + 1] = 0.0;
            if (pos[2 * i + 1] > patch) pos[2 * i + 1] = patch;
        } else {
            pos[2 * i] = pcg64_uniform(rng) * patch;
            pos[2 * i + 1] = pcg64_uniform(rng) * patch;
        }
    }
}

SimState *sim_create(int grid_size, int n_tcr, int n_cd45,
                     double kappa, double u_assoc, uint64_t seed,
                     int use_gpu,
                     double D_mol, double D_h, double dt_override) {
    SimState *s = (SimState *)calloc(1, sizeof(SimState));
    s->grid_size = grid_size;
    s->dx = PATCH_SIZE_NM / grid_size;
    s->n_tcr = n_tcr;
    s->n_cd45 = n_cd45;
    s->kappa = kappa;
    s->u_assoc = u_assoc;

    /* Apply defaults if zero. */
    if (D_mol <= 0.0) D_mol = D_MOL_DEFAULT;
    if (D_h <= 0.0) D_h = D_H_DEFAULT;
    s->D_mol = D_mol;
    s->D_h = D_h;

    /* Brownian dynamics time step from stability constraint:
     * dt_stable = dx² / (2 * D_h * κ) */
    if (dt_override > 0.0) {
        s->dt = dt_override;
    } else {
        double dt_stable = (s->dx * s->dx) / (2.0 * D_h * kappa);
        s->dt = dt_stable * DT_SAFETY;
    }

    /* Step sizes derived from physics: σ = sqrt(2 * D * dt) */
    s->step_size_mol = sqrt(2.0 * D_mol * s->dt);
    s->step_size_h = sqrt(2.0 * D_h * s->dt);

    s->h = (float *)malloc(grid_size * grid_size * sizeof(float));
    s->tcr_pos = (double *)malloc(n_tcr * 2 * sizeof(double));
    s->cd45_pos = (double *)malloc(n_cd45 * 2 * sizeof(double));

    pcg64_seed(&s->rng, seed);
    init_height_field(s);
    init_positions(s->tcr_pos, n_tcr, PATCH_SIZE_NM, &s->rng, 1);
    init_positions(s->cd45_pos, n_cd45, PATCH_SIZE_NM, &s->rng, 0);

    s->accepted = 0;
    s->total_proposals = 0;

    /* Try to init Metal GPU engine. Derive GPU RNG key from CPU seed. */
    s->use_gpu = 0;
    s->metal_ctx = NULL;
    if (use_gpu) {
        /* Use a deterministic key derived from the seed for GPU Philox RNG. */
        uint64_t gpu_key = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
        s->metal_ctx = metal_engine_create(grid_size, gpu_key);
        if (s->metal_ctx) {
            s->use_gpu = 1;
        }
    }

    return s;
}

void sim_destroy(SimState *s) {
    if (!s) return;
    if (s->metal_ctx) metal_engine_destroy(s->metal_ctx);
    free(s->h);
    free(s->tcr_pos);
    free(s->cd45_pos);
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

    /* TCR molecules. */
    for (int idx = 0; idx < s->n_tcr; idx++) {
        double ox = s->tcr_pos[2 * idx];
        double oy = s->tcr_pos[2 * idx + 1];
        double old_h = (double)height_at_pos_f(s->h, n, dx, ox, oy);
        double old_e = tcr_pmhc_potential(old_h, s->u_assoc, SIGMA_BIND_NM);

        double nx_ = ox + pcg64_normal(&s->rng, s->step_size_mol);
        double ny_ = oy + pcg64_normal(&s->rng, s->step_size_mol);
        if (nx_ < 0.0) nx_ = 0.0;
        if (nx_ > patch) nx_ = patch;
        if (ny_ < 0.0) ny_ = 0.0;
        if (ny_ > patch) ny_ = patch;

        double new_h = (double)height_at_pos_f(s->h, n, dx, nx_, ny_);
        double new_e = tcr_pmhc_potential(new_h, s->u_assoc, SIGMA_BIND_NM);

        double dE = new_e - old_e;
        s->total_proposals++;
        double u = pcg64_uniform(&s->rng);
        if (dE <= 0.0 || (u > 0.0 && log(u) < -dE)) {
            s->tcr_pos[2 * idx] = nx_;
            s->tcr_pos[2 * idx + 1] = ny_;
            s->accepted++;
        }
    }

    /* CD45 molecules. */
    for (int idx = 0; idx < s->n_cd45; idx++) {
        double ox = s->cd45_pos[2 * idx];
        double oy = s->cd45_pos[2 * idx + 1];
        double old_h = (double)height_at_pos_f(s->h, n, dx, ox, oy);
        double old_e = cd45_repulsion(old_h, CD45_HEIGHT_NM);

        double nx_ = ox + pcg64_normal(&s->rng, s->step_size_mol);
        double ny_ = oy + pcg64_normal(&s->rng, s->step_size_mol);
        if (nx_ < 0.0) nx_ = 0.0;
        if (nx_ > patch) nx_ = patch;
        if (ny_ < 0.0) ny_ = 0.0;
        if (ny_ > patch) ny_ = patch;

        double new_h = (double)height_at_pos_f(s->h, n, dx, nx_, ny_);
        double new_e = cd45_repulsion(new_h, CD45_HEIGHT_NM);

        double dE = new_e - old_e;
        s->total_proposals++;
        double u = pcg64_uniform(&s->rng);
        if (dE <= 0.0 || (u > 0.0 && log(u) < -dE)) {
            s->cd45_pos[2 * idx] = nx_;
            s->cd45_pos[2 * idx + 1] = ny_;
            s->accepted++;
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

/* Float-based bending energy delta for CPU path (mirrors GPU shader logic). */
static float lap_at_f(const float *h, int n, int i, int j, float dx2) {
    int im = (i - 1 + n) % n;
    int ip = (i + 1) % n;
    int jm = (j - 1 + n) % n;
    int jp = (j + 1) % n;
    return (h[im * n + j] + h[ip * n + j] +
            h[i * n + jm] + h[i * n + jp] -
            4.0f * h[i * n + j]) / dx2;
}

static float bending_energy_delta_f(const float *h, int n, float kappa, float dx,
                                    int gi, int gj, float old_val, float new_val) {
    float dx2 = dx * dx;
    float delta_h = new_val - old_val;
    float delta_e = 0.0f;

    int affected_i[5] = {gi, (gi - 1 + n) % n, (gi + 1) % n, gi, gi};
    int affected_j[5] = {gj, gj, gj, (gj - 1 + n) % n, (gj + 1) % n};

    for (int k = 0; k < 5; k++) {
        int ai = affected_i[k];
        int aj = affected_j[k];
        float new_lap = lap_at_f(h, n, ai, aj, dx2);
        float shift;
        if (ai == gi && aj == gj)
            shift = -4.0f * delta_h / dx2;
        else
            shift = delta_h / dx2;
        float old_lap = new_lap - shift;
        delta_e += new_lap * new_lap - old_lap * old_lap;
    }
    return 0.5f * kappa * delta_e * dx2;
}

/* Float-based potential functions for CPU Phase 2. */
static float tcr_potential_f(float h, float u_assoc, float sigma_bind) {
    return -u_assoc * expf(-(h * h) / (2.0f * sigma_bind * sigma_bind));
}

static float cd45_repulsion_f(float h, float cd45_height) {
    if (h < cd45_height) {
        float diff = cd45_height - h;
        return 0.5f * diff * diff;
    }
    return 0.0f;
}

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

    /* Per-cell buffers for the three-pass approach matching GPU. */
    float *old_vals = (float *)malloc(half * sizeof(float));
    float *u_accepts = (float *)malloc(half * sizeof(float));
    int *cell_gi = (int *)malloc(half * sizeof(int));
    int *cell_gj = (int *)malloc(half * sizeof(int));
    int *accepted_flags = (int *)malloc(half * sizeof(int));

    /* Read-only snapshot for bending_delta evaluation. */
    float *h_snapshot = (float *)malloc(n2 * sizeof(float));

    /* Three-pass checkerboard matching GPU propose→snapshot→evaluate→apply:
       Pass 1: generate all proposals for a color, write to h[].
       Snapshot: copy h[] (all proposals visible) for consistent reads.
       Pass 2: compute bending deltas against snapshot, decide accept/reject.
       Pass 3: restore rejected cells in h[].
       This ensures ALL cells evaluate against the same consistent state. */
    for (int color = 0; color < 2; color++) {
        int cidx = 0;

        /* Pass 1 (propose): generate proposals, write ALL to h[]. */
        for (int gi = 0; gi < n; gi++) {
            for (int gj = 0; gj < n; gj++) {
                if ((gi + gj) % 2 != color) continue;

                float old_h_val = s->h[gi * n + gj];

                /* Float32 Box-Muller matching GPU shader precision. */
                float u1_f = pcg64_uniform_f(&s->rng);
                if (u1_f < 1e-30f) u1_f = 1e-30f;
                float u2_f = pcg64_uniform_f(&s->rng);
                float normal_f = (float)s->step_size_h
                               * sqrtf(-2.0f * logf(u1_f))
                               * cosf(6.2831853071795864f * u2_f);
                float new_h_val = old_h_val + normal_f;
                if (new_h_val < 0.0f) new_h_val = 0.0f;

                float u_f = pcg64_uniform_f(&s->rng);

                old_vals[cidx] = old_h_val;
                u_accepts[cidx] = u_f;
                cell_gi[cidx] = gi;
                cell_gj[cidx] = gj;
                cidx++;

                s->h[gi * n + gj] = new_h_val;
            }
        }

        /* Snapshot: freeze h[] so all evaluations read the same state. */
        memcpy(h_snapshot, s->h, n2 * sizeof(float));

        /* Pass 2 (evaluate): bending deltas read from frozen snapshot. */
        for (int k = 0; k < cidx; k++) {
            int gi = cell_gi[k];
            int gj = cell_gj[k];
            float old_h_val = old_vals[k];
            float new_h_val = h_snapshot[gi * n + gj];

            int n_tcr_cell = tcr_count[gi * n + gj];
            int n_cd45_cell = cd45_count[gi * n + gj];

            float old_mol_e = n_tcr_cell * tcr_potential_f(old_h_val, (float)s->u_assoc, (float)SIGMA_BIND_NM)
                            + n_cd45_cell * cd45_repulsion_f(old_h_val, (float)CD45_HEIGHT_NM);

            float dE_bend = bending_energy_delta_f(h_snapshot, n, kappa, dx,
                                                   gi, gj, old_h_val, new_h_val);
            float new_mol_e = n_tcr_cell * tcr_potential_f(new_h_val, (float)s->u_assoc, (float)SIGMA_BIND_NM)
                            + n_cd45_cell * cd45_repulsion_f(new_h_val, (float)CD45_HEIGHT_NM);

            float dE = dE_bend + (new_mol_e - old_mol_e);
            s->total_proposals++;
            float u_f = u_accepts[k];
            accepted_flags[k] = (dE <= 0.0f || (u_f > 0.0f && logf(u_f) < -dE));
            if (accepted_flags[k]) s->accepted++;
        }

        /* Pass 3 (apply): restore rejected cells. */
        for (int k = 0; k < cidx; k++) {
            if (!accepted_flags[k]) {
                int gi = cell_gi[k];
                int gj = cell_gj[k];
                s->h[gi * n + gj] = old_vals[k];
            }
        }
    } /* end color loop */

    free(old_vals);
    free(u_accepts);
    free(cell_gi);
    free(cell_gj);
    free(accepted_flags);
    free(h_snapshot);
    free(tcr_count);
    free(cd45_count);
}

void sim_step(SimState *s) {
    phase1_molecules(s);
    if (s->use_gpu && s->metal_ctx) {
        metal_engine_grid_update(s->metal_ctx, s->h, s->grid_size,
                                 s->kappa, s->dx, s->step_size_h,
                                 s->u_assoc, SIGMA_BIND_NM, CD45_HEIGHT_NM,
                                 s->tcr_pos, s->n_tcr,
                                 s->cd45_pos, s->n_cd45,
                                 &s->accepted, &s->total_proposals);
    } else {
        phase2_grid_cpu(s);
    }
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
