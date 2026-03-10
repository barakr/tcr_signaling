#include "potentials.h"
#include <math.h>

double tcr_pmhc_potential(double h, double u_assoc, double sigma_bind) {
    return -u_assoc * exp(-(h * h) / (2.0 * sigma_bind * sigma_bind));
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
    for (int j = 0; j < n_mol; j++) {
        if (j == idx) continue;
        double dx = px - all_pos[2 * j];
        double dy = py - all_pos[2 * j + 1];
        if (dx > half_patch) dx -= patch_size;
        else if (dx < -half_patch) dx += patch_size;
        if (dy > half_patch) dy -= patch_size;
        else if (dy < -half_patch) dy += patch_size;
        double r = sqrt(dx * dx + dy * dy);
        if (r < r_cut) {
            double ratio = 1.0 - r / r_cut;
            total += eps * ratio * ratio;
        }
    }
    return total;
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
