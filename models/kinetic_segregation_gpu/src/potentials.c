#include "potentials.h"
#include <math.h>

double tcr_pmhc_potential(double h, double u_assoc, double sigma_bind) {
    return -u_assoc * exp(-(h * h) / (2.0 * sigma_bind * sigma_bind));
}

double cd45_repulsion(double h, double cd45_height) {
    if (h < cd45_height) {
        double diff = cd45_height - h;
        return 0.5 * 1.0 * diff * diff;  /* k_rep = 1.0 kT/nm^2 */
    }
    return 0.0;
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
