#include "potentials.h"
#include "cell_list.h"
#include <math.h>

double tcr_pmhc_potential(double h, double h0_tcr, double u_assoc, double sigma_bind) {
    double dh = h - h0_tcr;
    return -u_assoc * exp(-(dh * dh) / (2.0 * sigma_bind * sigma_bind));
}

double cd45_repulsion(double h, double cd45_height, double k_rep) {
    if (h < cd45_height) {
        double diff = cd45_height - h;
        return 0.5 * k_rep * diff * diff;
    }
    return 0.0;
}

double mol_repulsion(const double *pos, int idx, const double *all_pos, int n_mol,
                     double eps, double r_cut, double patch_size) {
    if (eps <= 0.0 || r_cut <= 0.0) return 0.0;
    double total = 0.0;
    double half_patch = patch_size / 2.0;
    double px = pos[0], py = pos[1];
    double r_cut2 = r_cut * r_cut;
    double inv_r_cut = 1.0 / r_cut;
    for (int j = 0; j < n_mol; j++) {
        if (j == idx) continue;
        double dx = px - all_pos[2 * j];
        double dy = py - all_pos[2 * j + 1];
        if (dx > half_patch) dx -= patch_size;
        else if (dx < -half_patch) dx += patch_size;
        if (dy > half_patch) dy -= patch_size;
        else if (dy < -half_patch) dy += patch_size;
        double r2 = dx * dx + dy * dy;
        if (r2 < r_cut2) {
            double r = sqrt(r2);
            double ratio = 1.0 - r * inv_r_cut;
            total += eps * ratio * ratio;
        }
    }
    return total;
}

/* Helper: pairwise repulsion contribution for a single pair at distance r. */
static inline double _repul_pair(double r2, double r_cut2, double inv_r_cut,
                                 double eps) {
    if (r2 < r_cut2) {
        double r = sqrt(r2);
        double ratio = 1.0 - r * inv_r_cut;
        return eps * ratio * ratio;
    }
    return 0.0;
}

/* Helper: minimum-image dx for periodic boundary. */
static inline double _mic(double d, double half_patch, double patch_size) {
    if (d > half_patch) return d - patch_size;
    if (d < -half_patch) return d + patch_size;
    return d;
}

double mol_repulsion_cell(const double *pos, int idx,
                          const CellList *cl, const double *all_pos,
                          double eps, double r_cut, double patch_size) {
    if (eps <= 0.0 || r_cut <= 0.0) return 0.0;
    double total = 0.0;
    double half_patch = patch_size / 2.0;
    double r_cut2 = r_cut * r_cut;
    double inv_r_cut = 1.0 / r_cut;
    double px = pos[0], py = pos[1];
    int nc = cl->nc;
    int ci = (int)(px * cl->inv_cell_size);
    int cj = (int)(py * cl->inv_cell_size);
    if (ci < 0) ci = 0; if (ci >= nc) ci = nc - 1;
    if (cj < 0) cj = 0; if (cj >= nc) cj = nc - 1;

    for (int di = -1; di <= 1; di++) {
        int ni = (ci + di + nc) % nc;
        for (int dj = -1; dj <= 1; dj++) {
            int nj = (cj + dj + nc) % nc;
            int j = cl->head[ni * nc + nj];
            while (j >= 0) {
                if (j != idx) {
                    double dx = _mic(px - all_pos[2 * j], half_patch, patch_size);
                    double dy = _mic(py - all_pos[2 * j + 1], half_patch, patch_size);
                    total += _repul_pair(dx * dx + dy * dy, r_cut2, inv_r_cut, eps);
                }
                j = cl->next[j];
            }
        }
    }
    return total;
}

double mol_repulsion_delta(const double *old_pos, const double *new_pos, int idx,
                           const CellList *cl, const double *all_pos,
                           double eps, double r_cut, double patch_size) {
    if (eps <= 0.0 || r_cut <= 0.0) return 0.0;
    double delta = 0.0;
    double half_patch = patch_size / 2.0;
    double r_cut2 = r_cut * r_cut;
    double inv_r_cut = 1.0 / r_cut;
    double ox = old_pos[0], oy = old_pos[1];
    double nx = new_pos[0], ny = new_pos[1];
    int nc = cl->nc;

    /* We need to scan neighbors reachable from EITHER the old or new position.
       Since cell_size >= r_cut, the union of 3x3 neighborhoods around the old
       and new cells covers all relevant pairs.  Collect unique cells. */
    int oci = (int)(ox * cl->inv_cell_size);
    int ocj = (int)(oy * cl->inv_cell_size);
    if (oci < 0) oci = 0; if (oci >= nc) oci = nc - 1;
    if (ocj < 0) ocj = 0; if (ocj >= nc) ocj = nc - 1;
    int nci = (int)(nx * cl->inv_cell_size);
    int ncj = (int)(ny * cl->inv_cell_size);
    if (nci < 0) nci = 0; if (nci >= nc) nci = nc - 1;
    if (ncj < 0) ncj = 0; if (ncj >= nc) ncj = nc - 1;

    /* Gather unique cell indices from both neighborhoods. */
    int cells[18];  /* max 9 + 9, but many overlap */
    int n_cells = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            int c1 = ((oci + di + nc) % nc) * nc + ((ocj + dj + nc) % nc);
            /* Add if not already present. */
            int dup = 0;
            for (int k = 0; k < n_cells; k++) { if (cells[k] == c1) { dup = 1; break; } }
            if (!dup) cells[n_cells++] = c1;

            int c2 = ((nci + di + nc) % nc) * nc + ((ncj + dj + nc) % nc);
            dup = 0;
            for (int k = 0; k < n_cells; k++) { if (cells[k] == c2) { dup = 1; break; } }
            if (!dup) cells[n_cells++] = c2;
        }
    }

    for (int ci = 0; ci < n_cells; ci++) {
        int j = cl->head[cells[ci]];
        while (j >= 0) {
            if (j != idx) {
                double jx = all_pos[2 * j], jy = all_pos[2 * j + 1];
                double dx_o = _mic(ox - jx, half_patch, patch_size);
                double dy_o = _mic(oy - jy, half_patch, patch_size);
                double dx_n = _mic(nx - jx, half_patch, patch_size);
                double dy_n = _mic(ny - jy, half_patch, patch_size);
                delta -= _repul_pair(dx_o * dx_o + dy_o * dy_o, r_cut2, inv_r_cut, eps);
                delta += _repul_pair(dx_n * dx_n + dy_n * dy_n, r_cut2, inv_r_cut, eps);
            }
            j = cl->next[j];
        }
    }
    return delta;
}

static double lap_at(const double *h, int n, int i, int j, double dx2) {
    int im = (i - 1 + n) % n;
    int ip = (i + 1) % n;
    int jm = (j - 1 + n) % n;
    int jp = (j + 1) % n;
    return (h[im * n + j] + h[ip * n + j] +
            h[i * n + jm] + h[i * n + jp] -
            4.0 * h[i * n + j]) / dx2;
}

double bending_energy_delta(const double *h, int n, double kappa, double dx,
                            int gi, int gj, double old_val, double new_val) {
    double dx2 = dx * dx;
    double delta_h = new_val - old_val;
    double delta_e = 0.0;

    /* Affected cells: (gi,gj) and its 4 periodic neighbors. */
    int affected_i[5] = {gi, (gi - 1 + n) % n, (gi + 1) % n, gi, gi};
    int affected_j[5] = {gj, gj, gj, (gj - 1 + n) % n, (gj + 1) % n};

    for (int k = 0; k < 5; k++) {
        int ai = affected_i[k];
        int aj = affected_j[k];
        double new_lap = lap_at(h, n, ai, aj, dx2);
        double shift;
        if (ai == gi && aj == gj)
            shift = -4.0 * delta_h / dx2;
        else
            shift = delta_h / dx2;
        double old_lap = new_lap - shift;
        delta_e += new_lap * new_lap - old_lap * old_lap;
    }
    return 0.5 * kappa * delta_e * dx2;
}
