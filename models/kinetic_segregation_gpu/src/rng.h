#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* PCG64 deterministic PRNG (PCG-XSH-RR variant). */
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg64_t;

/* Seed the generator. */
void pcg64_seed(pcg64_t *rng, uint64_t seed);

/* Uniform uint32 in [0, 2^32). */
uint32_t pcg64_random(pcg64_t *rng);

/* Uniform double in [0, 1). */
double pcg64_uniform(pcg64_t *rng);

/* Uniform float in [0, 1) — float32 division matching GPU precision. */
float pcg64_uniform_f(pcg64_t *rng);

/* Gaussian via Box-Muller (mean=0, stddev=sigma). */
double pcg64_normal(pcg64_t *rng, double sigma);

#endif /* RNG_H */
