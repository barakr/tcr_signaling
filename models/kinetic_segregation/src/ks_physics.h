/*
 * ks_physics.h — Shared float-precision physics functions for CPU and GPU.
 *
 * Included by both simulation.c (CPU Phase 2 path) and shaders.metal (GPU kernels).
 * Uses preprocessor guards for Metal address-space qualifiers and math functions.
 * CUDA support: add __CUDA_ARCH__ guard when needed.
 */
#ifndef KS_PHYSICS_H
#define KS_PHYSICS_H

#if defined(__METAL_VERSION__)
  /* Metal Shading Language (C++14-based). */
  #include <metal_stdlib>
  using namespace metal;
  #define KS_DEVICE device
  #define KS_EXPF(x) exp(x)
#elif defined(__CUDA_ARCH__)
  /* CUDA device code (future). */
  #define KS_DEVICE
  #define KS_EXPF(x) expf(x)
#else
  /* CPU (C99/C++). */
  #include <math.h>
  #define KS_DEVICE
  #define KS_EXPF(x) expf(x)
#endif

/* TCR-pMHC attractive Gaussian well centered at h0_tcr:
   E = -U_assoc * exp(-(h - h0_tcr)^2 / (2*sigma^2)). */
static inline float ks_tcr_potential(float h, float h0_tcr, float u_assoc, float sigma_bind) {
    float dh = h - h0_tcr;
    return -u_assoc * KS_EXPF(-(dh * dh) / (2.0f * sigma_bind * sigma_bind));
}

/* CD45 soft repulsive barrier: E = 0.5*k_rep*(h_cd45 - h)^2 if h < h_cd45. */
static inline float ks_cd45_repulsion(float h, float cd45_height, float k_rep) {
    if (h < cd45_height) {
        float diff = cd45_height - h;
        return 0.5f * k_rep * diff * diff;
    }
    return 0.0f;
}

/* Discrete Laplacian at (i,j) with periodic boundary conditions. */
static inline float ks_lap_at(KS_DEVICE const float *h, int n, int i, int j,
                               float dx2) {
    int im = (i - 1 + n) % n;
    int ip = (i + 1) % n;
    int jm = (j - 1 + n) % n;
    int jp = (j + 1) % n;
    return (h[im * n + j] + h[ip * n + j] +
            h[i * n + jm] + h[i * n + jp] -
            4.0f * h[i * n + j]) / dx2;
}

/* Bending energy delta when h[gi][gj] changes from old_val to new_val.
 * h[] must already contain new_val at position [gi][gj]. */
static inline float ks_bending_delta(KS_DEVICE const float *h, int n,
                                      float kappa, float dx,
                                      int gi, int gj,
                                      float old_val, float new_val) {
    float dx2 = dx * dx;
    float delta_h = new_val - old_val;
    float delta_e = 0.0f;

    int affected_i[5] = {gi, (gi - 1 + n) % n, (gi + 1) % n, gi, gi};
    int affected_j[5] = {gj, gj, gj, (gj - 1 + n) % n, (gj + 1) % n};

    for (int k = 0; k < 5; k++) {
        int ai = affected_i[k];
        int aj = affected_j[k];
        float new_lap = ks_lap_at(h, n, ai, aj, dx2);
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

/* Clean up macros to avoid polluting includer's namespace. */
#undef KS_DEVICE
#undef KS_EXPF

#endif /* KS_PHYSICS_H */
