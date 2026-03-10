#include "rng.h"
#include <math.h>

void pcg64_seed(pcg64_t *rng, uint64_t seed) {
    rng->state = 0;
    rng->inc = (seed << 1u) | 1u;  /* must be odd */
    pcg64_random(rng);
    rng->state += seed;
    pcg64_random(rng);
}

uint32_t pcg64_random(pcg64_t *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

double pcg64_uniform(pcg64_t *rng) {
    return (double)pcg64_random(rng) / 4294967296.0;
}

float pcg64_uniform_f(pcg64_t *rng) {
    return (float)pcg64_random(rng) / 4294967296.0f;
}

double pcg64_normal(pcg64_t *rng, double sigma) {
    /* Box-Muller transform */
    double u1, u2;
    do {
        u1 = pcg64_uniform(rng);
    } while (u1 == 0.0);
    u2 = pcg64_uniform(rng);
    return sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
