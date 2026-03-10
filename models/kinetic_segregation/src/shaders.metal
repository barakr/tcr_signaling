/* Shared physics included before metal_stdlib to get KS_DEVICE defined. */
#include "ks_physics.h"

/* Parameters passed to the grid update kernels. */
struct GridParams {
    int grid_size;
    float kappa;
    float dx;
    float step_size_h;
    float u_assoc;
    float sigma_bind;
    float cd45_height;
    float k_rep;
    int color;        /* 0 = red (even sum), 1 = black (odd sum) */
    uint rng_key0;    /* derived from CPU seed */
    uint rng_key1;    /* derived from CPU seed */
    uint rng_offset;  /* increments each half-sweep for unique streams */
};

/* Per-cell proposal data shared between kernels. */
struct CellProposal {
    float old_h;
    float u_accept;
    int accepted;     /* 1 = accepted, 0 = rejected (set by evaluate) */
};

/* Philox4x32-10 counter-based PRNG. */
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

/* Physics functions (ks_tcr_potential, ks_cd45_repulsion, ks_lap_at,
 * ks_bending_delta) are provided by ks_physics.h above. */

/* Map linear thread id to (gi, gj) for the given checkerboard color. */
static void decode_cell(int linear, int n, int color, thread int &gi, thread int &gj) {
    int row = linear / (n / 2);
    int col_half = linear % (n / 2);
    gi = row;
    if (color == 0) {
        gj = col_half * 2 + (gi % 2 == 0 ? 0 : 1);
    } else {
        gj = col_half * 2 + (gi % 2 == 0 ? 1 : 0);
    }
}

/* Kernel 1 (propose): Generate proposals and write ALL to h[].
   Saves old_h and u_accept to proposals buffer. */
kernel void grid_propose_kernel(
    device float *h                         [[buffer(0)]],
    constant GridParams &params             [[buffer(1)]],
    device CellProposal *proposals          [[buffer(2)]],
    uint tid                                [[thread_position_in_grid]])
{
    int n = params.grid_size;
    if ((int)tid >= n * n / 2) return;

    int gi, gj;
    decode_cell((int)tid, n, params.color, gi, gj);
    if (gi >= n || gj >= n) return;

    uint4 ctr = uint4(tid, params.rng_offset, 0u, 0u);
    uint2 key = uint2(params.rng_key0, params.rng_key1);
    uint4 bits = philox4x32_10(ctr, key);

    float u1 = (float)(bits.x) / 4294967296.0f;
    float u2 = (float)(bits.y) / 4294967296.0f;
    float u_accept = (float)(bits.z) / 4294967296.0f;

    float normal = params.step_size_h * sqrt(-2.0f * log(max(u1, 1e-30f)))
                 * cos(6.2831853071795864f * u2);

    int cell_idx = gi * n + gj;
    float old_h_val = h[cell_idx];

    float new_h_val = old_h_val + normal;
    if (new_h_val < 0.0f) new_h_val = -new_h_val;

    proposals[tid].old_h = old_h_val;
    proposals[tid].u_accept = u_accept;
    proposals[tid].accepted = 0;

    h[cell_idx] = new_h_val;
}

/* Kernel 2 (snapshot): Copy h[] → h_snap[] for consistent reads.
   Run as a simple copy over the full grid (not just half). */
kernel void grid_snapshot_kernel(
    device const float *h                   [[buffer(0)]],
    device float *h_snap                    [[buffer(1)]],
    uint tid                                [[thread_position_in_grid]])
{
    h_snap[tid] = h[tid];
}

/* Kernel 3 (evaluate): Compute bending deltas from frozen snapshot.
   All threads read from h_snap (immutable), write decisions to proposals. */
kernel void grid_evaluate_kernel(
    device const float *h_snap              [[buffer(0)]],
    device const int *tcr_count             [[buffer(1)]],
    device const int *cd45_count            [[buffer(2)]],
    constant GridParams &params             [[buffer(3)]],
    device atomic_int *accept_count         [[buffer(4)]],
    device CellProposal *proposals          [[buffer(5)]],
    device const int *pmhc_count            [[buffer(6)]],
    uint tid                                [[thread_position_in_grid]])
{
    int n = params.grid_size;
    if ((int)tid >= n * n / 2) return;

    int gi, gj;
    decode_cell((int)tid, n, params.color, gi, gj);
    if (gi >= n || gj >= n) return;

    int cell_idx = gi * n + gj;
    float old_h_val = proposals[tid].old_h;
    float u_accept = proposals[tid].u_accept;
    float new_h_val = h_snap[cell_idx];

    int n_tcr_cell = tcr_count[cell_idx];
    int n_cd45_cell = cd45_count[cell_idx];
    int has_pmhc = pmhc_count[cell_idx] > 0;

    float tcr_e_old = has_pmhc ? n_tcr_cell * ks_tcr_potential(old_h_val, params.u_assoc, params.sigma_bind) : 0.0f;
    float old_mol_e = tcr_e_old
                    + n_cd45_cell * ks_cd45_repulsion(old_h_val, params.cd45_height, params.k_rep);

    float dE_bend = ks_bending_delta(h_snap, n, params.kappa, params.dx,
                                  gi, gj, old_h_val, new_h_val);
    float tcr_e_new = has_pmhc ? n_tcr_cell * ks_tcr_potential(new_h_val, params.u_assoc, params.sigma_bind) : 0.0f;
    float new_mol_e = tcr_e_new
                    + n_cd45_cell * ks_cd45_repulsion(new_h_val, params.cd45_height, params.k_rep);

    float dE = dE_bend + (new_mol_e - old_mol_e);

    if (dE <= 0.0f || (u_accept > 0.0f && log(u_accept) < -dE)) {
        atomic_fetch_add_explicit(accept_count, 1, memory_order_relaxed);
        proposals[tid].accepted = 1;
    }
}

/* Kernel 4 (apply): Restore rejected cells in h[]. */
kernel void grid_apply_kernel(
    device float *h                         [[buffer(0)]],
    constant GridParams &params             [[buffer(1)]],
    device const CellProposal *proposals    [[buffer(2)]],
    uint tid                                [[thread_position_in_grid]])
{
    int n = params.grid_size;
    if ((int)tid >= n * n / 2) return;

    int gi, gj;
    decode_cell((int)tid, n, params.color, gi, gj);
    if (gi >= n || gj >= n) return;

    if (!proposals[tid].accepted) {
        h[gi * n + gj] = proposals[tid].old_h;
    }
}
