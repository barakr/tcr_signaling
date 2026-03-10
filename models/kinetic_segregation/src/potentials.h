#ifndef POTENTIALS_H
#define POTENTIALS_H

/* TCR-pMHC attractive Gaussian well: E = -u_assoc * exp(-h^2 / (2*sigma^2)). */
double tcr_pmhc_potential(double h, double u_assoc, double sigma_bind);

/* CD45 soft repulsive barrier: harmonic wall when h < cd45_height. */
double cd45_repulsion(double h, double cd45_height, double k_rep);

/* Soft pairwise repulsive potential for molecule idx against all others.
   Uses minimum-image convention for periodic boundaries.
   Returns total repulsive energy contribution.  O(N) brute-force. */
double mol_repulsion(const double *pos, int idx, const double *all_pos, int n_mol,
                     double eps, double r_cut, double patch_size);

/* Cell-list accelerated version of mol_repulsion.  O(1) amortized per call
   when molecules are uniformly distributed.  Requires a built CellList. */
struct CellList;
double mol_repulsion_cell(const double *pos, int idx,
                          const struct CellList *cl, const double *all_pos,
                          double eps, double r_cut, double patch_size);

/* Energy delta for molecule idx moving from old_pos to new_pos, computed in
   a single pass over cell-list neighbors.  Returns (new_e - old_e). */
double mol_repulsion_delta(const double *old_pos, const double *new_pos, int idx,
                           const struct CellList *cl, const double *all_pos,
                           double eps, double r_cut, double patch_size);

/* Change in bending energy when h[gi][gj] changes from old_val to new_val.
   h must already contain new_val. Uses periodic BCs. O(1). */
double bending_energy_delta(const double *h, int n, double kappa, double dx,
                            int gi, int gj, double old_val, double new_val);

#endif /* POTENTIALS_H */
