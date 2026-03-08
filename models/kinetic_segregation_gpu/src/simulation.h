#ifndef SIMULATION_H
#define SIMULATION_H

#include "rng.h"

/* Physical constants matching the Python model. */
#define PATCH_SIZE_NM    2000.0
#define CD45_HEIGHT_NM   35.0
#define SIGMA_BIND_NM    3.0
#define U_ASSOC_DEFAULT  20.0

/* Brownian dynamics defaults (nm^2/s). */
#define D_MOL_DEFAULT    1e5
#define D_H_DEFAULT      5e4
#define DT_SAFETY        0.5

typedef struct {
    /* Grid */
    int grid_size;
    double dx;
    float *h;                /* grid_size x grid_size height field (float32) */

    /* Molecules */
    int n_tcr;
    int n_cd45;
    double *tcr_pos;          /* n_tcr x 2 (x, y) */
    double *cd45_pos;         /* n_cd45 x 2 (x, y) */

    /* Parameters */
    double kappa;             /* rigidity_kT_nm2 */
    double u_assoc;
    double step_size_mol;
    double step_size_h;
    double dt;                /* physical time step (seconds) */
    double D_mol;             /* molecular diffusion coefficient (nm²/s) */
    double D_h;               /* membrane height diffusion coefficient (nm²/s) */
    double cd45_height;       /* CD45 ectodomain height (nm) */
    double k_rep;             /* CD45 repulsive spring constant (kT/nm²) */
    double mol_repulsion_eps; /* soft molecular repulsion strength (kT) */
    double mol_repulsion_rcut;/* soft molecular repulsion cutoff (nm) */

    /* pMHC: static positions on APC surface, gating TCR binding */
    int n_pmhc;
    double *pmhc_pos;         /* n_pmhc x 2 (x, y), NULL if n_pmhc=0 */
    int *pmhc_count;          /* grid_size x grid_size binned counts, NULL if n_pmhc=0 */
    int pmhc_mode;            /* 0=uniform, 1=inner_circle */
    double pmhc_radius;       /* placement disc radius (nm), 0=auto (patch/3) */

    /* Diagnostics */
    long accepted;
    long total_proposals;
    int n_steps;

    /* RNG */
    pcg64_t rng;

    /* Metal GPU availability (0 = CPU fallback, 1 = GPU) */
    int use_gpu;
    void *metal_ctx;          /* opaque pointer to MetalEngine */
} SimState;

/* Allocate and initialize simulation state.
 * D_mol, D_h: diffusion coefficients (nm²/s). Pass 0 for defaults.
 * dt_override: if > 0, use this dt instead of auto-computing.
 * cd45_height: CD45 ectodomain height (nm). Pass 0 for default (35nm).
 * k_rep: CD45 repulsive spring constant (kT/nm²). Pass 0 for default (1.0).
 * mol_repulsion_eps: soft molecular repulsion strength (kT). 0 = disabled.
 * mol_repulsion_rcut: cutoff distance (nm). 0 = default (10nm).
 * n_pmhc: number of static pMHC molecules. 0 = binding everywhere.
 * pmhc_seed: seed for pMHC random positions.
 * pmhc_mode: 0=uniform, 1=inner_circle (default).
 * pmhc_radius: placement disc radius (nm). 0 = auto (patch/3). */
SimState *sim_create(int grid_size, int n_tcr, int n_cd45,
                     double kappa, double u_assoc, uint64_t seed,
                     int use_gpu,
                     double D_mol, double D_h, double dt_override,
                     double cd45_height, double k_rep,
                     double mol_repulsion_eps, double mol_repulsion_rcut,
                     int n_pmhc, uint64_t pmhc_seed,
                     int pmhc_mode, double pmhc_radius);

/* Free simulation state. */
void sim_destroy(SimState *s);

/* Run n_steps full MC sweeps. */
void sim_run(SimState *s, int n_steps);

/* Run a single MC sweep (Phase 1 + Phase 2). */
void sim_step(SimState *s);

/* Compute depletion width from current state. */
double sim_depletion_width(const SimState *s);

/* Compute mean radial distance for TCR and CD45. */
double sim_mean_r(const double *pos, int n);

#endif /* SIMULATION_H */
