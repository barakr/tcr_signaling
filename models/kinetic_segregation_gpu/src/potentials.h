#ifndef POTENTIALS_H
#define POTENTIALS_H

/* TCR-pMHC attractive Gaussian well: E = -u_assoc * exp(-h^2 / (2*sigma^2)). */
double tcr_pmhc_potential(double h, double u_assoc, double sigma_bind);

/* CD45 soft repulsive barrier: harmonic wall when h < cd45_height. */
double cd45_repulsion(double h, double cd45_height, double k_rep);

/* Soft pairwise repulsive potential for molecule idx against all others.
   Uses minimum-image convention for periodic boundaries.
   Returns total repulsive energy contribution. */
double mol_repulsion(const double *pos, int idx, const double *all_pos, int n_mol,
                     double eps, double r_cut, double patch_size);

/* Change in bending energy when h[gi][gj] changes from old_val to new_val.
   h must already contain new_val. Uses periodic BCs. O(1). */
double bending_energy_delta(const double *h, int n, double kappa, double dx,
                            int gi, int gj, double old_val, double new_val);

#endif /* POTENTIALS_H */
