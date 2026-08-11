/* ks_compat.h — platform shims for the CPU sources.
 *
 * Include this INSTEAD of <math.h> in any translation unit that uses M_PI.
 *
 * Why this file exists: M_PI is not standard C. glibc and libc++ expose it from
 * <math.h> unconditionally, but MSVC only does so when _USE_MATH_DEFINES is
 * defined *before* <math.h> is first included — so a file that includes
 * <math.h> directly compiles on macOS and Linux and fails on Windows with
 * "undeclared identifier 'M_PI'". Centralising the define means a new source
 * file cannot reintroduce that asymmetry by forgetting it.
 *
 * This header is CPU-only. `ks_physics.h` is the shared CPU+GPU float physics
 * header and is included by shaders.metal, which has no C standard library —
 * do not pull this in from there.
 */

#ifndef KS_COMPAT_H
#define KS_COMPAT_H

/* Must precede <math.h> for MSVC to declare M_PI and friends. */
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

/* Belt and braces: some MSVC configurations still hide it (e.g. /Za, or when
 * another header reached <math.h> first). Harmless where it is already set. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Monotonic clock, used only by the KS_PROFILE build ──────────────────────
 *
 * clock_gettime(CLOCK_MONOTONIC, ...) is POSIX and absent on MSVC. C11's
 * timespec_get() is portable but wall-clock, so it can step backwards under NTP
 * and produce negative phase timings. QueryPerformanceCounter is the monotonic
 * Windows equivalent, so each platform gets its real monotonic source.
 */
#ifdef KS_PROFILE

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static double ks_clock_ms(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

#else
#include <time.h>

static double ks_clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif /* _WIN32 */

#endif /* KS_PROFILE */

#endif /* KS_COMPAT_H */
