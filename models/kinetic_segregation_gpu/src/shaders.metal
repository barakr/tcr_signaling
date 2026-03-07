#include <metal_stdlib>
using namespace metal;

/* Parameters passed to the grid update kernel. */
struct GridParams {
    int grid_size;
    float kappa;
    float dx;
    float step_size_h;
    float u_assoc;
    float sigma_bind;
    float cd45_height;
    int color;        /* 0 = red (even sum), 1 = black (odd sum) */
    uint rng_key0;    /* derived from CPU seed */
    uint rng_key1;    /* derived from CPU seed */
    uint rng_offset;  /* increments each half-sweep for unique streams */
};

/* Philox4x32-10 counter-based PRNG.
   Standard variant used by JAX/PyTorch/cuRAND. Passes BigCrush.
   Uses Metal's built-in mulhi() for high 32 bits of uint multiplication. */
static uint4 philox4x32_round(uint4 ctr, uint2 key) {
    uint lo0 = ctr.x * 0xD2511F53u;
    uint hi0 = mulhi(ctr.x, 0xD2511F53u);
    uint lo1 = ctr.z * 0xCD9E8D57u;
    uint hi1 = mulhi(ctr.z, 0xCD9E8D57u);
    return uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

static uint4 philox4x32_10(uint4 ctr, uint2 key) {
    for (int i = 0; i < 10; i++) {
        ctr = philox4x32_round(ctr, key);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return ctr;
}

/* TCR-pMHC attractive Gaussian well. */
static float tcr_potential(float h, float u_assoc, float sigma_bind) {
    return -u_assoc * exp(-(h * h) / (2.0f * sigma_bind * sigma_bind));
}

/* CD45 soft repulsive barrier. */
static float cd45_repulsion(float h, float cd45_height) {
    if (h < cd45_height) {
        float diff = cd45_height - h;
        return 0.5f * diff * diff;
    }
    return 0.0f;
}

/* Discrete Laplacian at (i,j) with periodic BCs. */
static float lap_at(device const float *h, int n, int i, int j, float dx2) {
    int im = (i - 1 + n) % n;
    int ip = (i + 1) % n;
    int jm = (j - 1 + n) % n;
    int jp = (j + 1) % n;
    return (h[im * n + j] + h[ip * n + j] +
            h[i * n + jm] + h[i * n + jp] -
            4.0f * h[i * n + j]) / dx2;
}

/* Bending energy delta when h[gi][gj] changes. h must already contain new value. */
static float bending_delta(device const float *h, int n, float kappa, float dx,
                           int gi, int gj, float old_val, float new_val) {
    float dx2 = dx * dx;
    float delta_h = new_val - old_val;
    float delta_e = 0.0f;

    int affected_i[5] = {gi, (gi - 1 + n) % n, (gi + 1) % n, gi, gi};
    int affected_j[5] = {gj, gj, gj, (gj - 1 + n) % n, (gj + 1) % n};

    for (int k = 0; k < 5; k++) {
        int ai = affected_i[k];
        int aj = affected_j[k];
        float new_lap = lap_at(h, n, ai, aj, dx2);
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

kernel void grid_update_kernel(
    device float *h                     [[buffer(0)]],
    device const int *tcr_count         [[buffer(1)]],
    device const int *cd45_count        [[buffer(2)]],
    constant GridParams &params         [[buffer(3)]],
    device atomic_int *accept_count     [[buffer(4)]],
    uint tid                            [[thread_position_in_grid]])
{
    int n = params.grid_size;
    int total_cells = n * n;

    if ((int)tid >= total_cells / 2) return;

    /* Decode checkerboard: map linear thread id to (gi, gj) of correct color. */
    int linear = (int)tid;
    int row = linear / (n / 2);
    int col_half = linear % (n / 2);
    int gi = row;
    int gj;
    if (params.color == 0) {
        gj = col_half * 2 + (gi % 2 == 0 ? 0 : 1);
    } else {
        gj = col_half * 2 + (gi % 2 == 0 ? 1 : 0);
    }

    if (gi >= n || gj >= n) return;

    /* Generate random numbers on-GPU via Philox4x32-10.
       Counter = (tid, rng_offset, 0, 0) ensures unique stream per thread per half-sweep. */
    uint4 ctr = uint4(tid, params.rng_offset, 0u, 0u);
    uint2 key = uint2(params.rng_key0, params.rng_key1);
    uint4 bits = philox4x32_10(ctr, key);

    /* Convert to floats: u1, u2 for Box-Muller normal, u_accept for Metropolis. */
    float u1 = (float)(bits.x) / 4294967296.0f;
    float u2 = (float)(bits.y) / 4294967296.0f;
    float u_accept = (float)(bits.z) / 4294967296.0f;

    /* Box-Muller transform for normal(0, step_size_h). */
    float normal = params.step_size_h * sqrt(-2.0f * log(max(u1, 1e-30f))) * cos(6.2831853071795864f * u2);

    int cell_idx = gi * n + gj;
    float old_h_val = h[cell_idx];

    /* Read pre-binned molecule counts (O(1) per cell). */
    int n_tcr_cell = tcr_count[cell_idx];
    int n_cd45_cell = cd45_count[cell_idx];

    float old_mol_e = n_tcr_cell * tcr_potential(old_h_val, params.u_assoc, params.sigma_bind)
                    + n_cd45_cell * cd45_repulsion(old_h_val, params.cd45_height);

    /* Propose new height using GPU-generated normal. */
    float new_h_val = old_h_val + normal;
    if (new_h_val < 0.0f) new_h_val = 0.0f;

    /* Write new height so bending_delta can read the stencil.
       Safe: checkerboard ensures no neighbor is being updated simultaneously. */
    h[cell_idx] = new_h_val;

    float dE_bend = bending_delta(h, n, params.kappa, params.dx,
                                  gi, gj, old_h_val, new_h_val);
    float new_mol_e = n_tcr_cell * tcr_potential(new_h_val, params.u_assoc, params.sigma_bind)
                    + n_cd45_cell * cd45_repulsion(new_h_val, params.cd45_height);

    float dE = dE_bend + (new_mol_e - old_mol_e);

    if (dE <= 0.0f || (u_accept > 0.0f && log(u_accept) < -dE)) {
        /* Accept: h already has new value. */
        atomic_fetch_add_explicit(accept_count, 1, memory_order_relaxed);
    } else {
        /* Reject: restore old value. */
        h[cell_idx] = old_h_val;
    }
}
