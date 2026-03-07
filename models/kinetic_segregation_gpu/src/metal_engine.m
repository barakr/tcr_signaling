#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_engine.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Parameters struct matching the shader's GridParams. */
typedef struct {
    int grid_size;
    float kappa;
    float dx;
    float step_size_h;
    float u_assoc;
    float sigma_bind;
    float cd45_height;
    int color;
    uint32_t rng_key0;
    uint32_t rng_key1;
    uint32_t rng_offset;
} GridParams;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;

    /* Buffers (sized for max grid_size). */
    id<MTLBuffer> h_buf;              /* float[n^2] — authoritative height field */
    id<MTLBuffer> tcr_count_buf;      /* int[n^2]   — molecule count per cell */
    id<MTLBuffer> cd45_count_buf;     /* int[n^2]   — molecule count per cell */
    id<MTLBuffer> params_buf;         /* GridParams — color 0 */
    id<MTLBuffer> params_buf2;        /* GridParams — color 1 */
    id<MTLBuffer> accept_buf;         /* atomic_int[1] */

    int grid_size;
    uint32_t rng_key0;   /* Philox key derived from CPU seed */
    uint32_t rng_key1;
    uint32_t rng_counter; /* Monotonically increasing offset for unique streams */
} MetalEngine;

void *metal_engine_create(int grid_size, uint64_t gpu_rng_key) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal: no GPU device available, using CPU fallback.\n");
            return NULL;
        }

        MetalEngine *eng = (MetalEngine *)calloc(1, sizeof(MetalEngine));
        eng->device = device;
        eng->grid_size = grid_size;
        eng->queue = [device newCommandQueue];

        /* Derive Philox key from CPU seed. */
        eng->rng_key0 = (uint32_t)(gpu_rng_key & 0xFFFFFFFFu);
        eng->rng_key1 = (uint32_t)(gpu_rng_key >> 32);
        eng->rng_counter = 0;

        /* Load shader source from file next to the binary. */
        NSString *shaderPath = nil;
        NSString *execPath = [[NSBundle mainBundle] executablePath];
        if (execPath) {
            NSString *dir = [execPath stringByDeletingLastPathComponent];
            shaderPath = [dir stringByAppendingPathComponent:@"shaders.metal"];
        }
        if (!shaderPath || ![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
            NSArray *candidates = @[
                @"src/shaders.metal",
                @"shaders.metal",
                @"models/kinetic_segregation_gpu/src/shaders.metal",
            ];
            for (NSString *c in candidates) {
                if ([[NSFileManager defaultManager] fileExistsAtPath:c]) {
                    shaderPath = c;
                    break;
                }
            }
        }

        if (!shaderPath || ![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
            fprintf(stderr, "Metal: shader source not found, using CPU fallback.\n");
            free(eng);
            return NULL;
        }

        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:shaderPath
                                                    encoding:NSUTF8StringEncoding
                                                       error:&error];
        if (!source) {
            fprintf(stderr, "Metal: failed to read shader: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(eng);
            return NULL;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:opts error:&error];
        if (!library) {
            fprintf(stderr, "Metal: shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(eng);
            return NULL;
        }

        id<MTLFunction> func = [library newFunctionWithName:@"grid_update_kernel"];
        if (!func) {
            fprintf(stderr, "Metal: kernel function not found.\n");
            free(eng);
            return NULL;
        }

        eng->pipeline = [device newComputePipelineStateWithFunction:func error:&error];
        if (!eng->pipeline) {
            fprintf(stderr, "Metal: pipeline creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(eng);
            return NULL;
        }

        /* Allocate buffers — no random buffers needed (GPU generates its own). */
        int n2 = grid_size * grid_size;
        eng->h_buf = [device newBufferWithLength:n2 * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        eng->tcr_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                 options:MTLResourceStorageModeShared];
        eng->cd45_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                  options:MTLResourceStorageModeShared];
        eng->params_buf = [device newBufferWithLength:sizeof(GridParams)
                                              options:MTLResourceStorageModeShared];
        eng->params_buf2 = [device newBufferWithLength:sizeof(GridParams)
                                               options:MTLResourceStorageModeShared];
        eng->accept_buf = [device newBufferWithLength:sizeof(int)
                                              options:MTLResourceStorageModeShared];

        /* Initialize h_buf to zero — will be populated on first grid_update call. */

        fprintf(stderr, "Metal: GPU engine initialized on %s (grid=%d).\n",
                [[device name] UTF8String], grid_size);
        return eng;
    }
}

void metal_engine_destroy(void *ctx) {
    if (!ctx) return;
    MetalEngine *eng = (MetalEngine *)ctx;
    /* ARC handles ObjC objects. */
    free(eng);
}

/* Bin molecule positions into a per-cell count grid on CPU. */
static void bin_molecules_f(const double *pos, int n_mol, int grid_size,
                            double dx, int *count_grid) {
    memset(count_grid, 0, grid_size * grid_size * sizeof(int));
    for (int m = 0; m < n_mol; m++) {
        int ix = (int)(pos[2 * m] / dx);
        int iy = (int)(pos[2 * m + 1] / dx);
        if (ix < 0) ix = 0; if (ix >= grid_size) ix = grid_size - 1;
        if (iy < 0) iy = 0; if (iy >= grid_size) iy = grid_size - 1;
        count_grid[ix * grid_size + iy]++;
    }
}

void metal_engine_grid_update(void *ctx, float *h, int grid_size,
                              double kappa, double dx, double step_size_h,
                              double u_assoc, double sigma_bind,
                              double cd45_height,
                              const double *tcr_pos, int n_tcr,
                              const double *cd45_pos, int n_cd45,
                              long *accepted, long *total_proposals) {
    @autoreleasepool {
        MetalEngine *eng = (MetalEngine *)ctx;
        int n2 = grid_size * grid_size;
        int half = n2 / 2;

        /* h_buf is shared memory — copy h into it (h is the authoritative float array). */
        float *h_gpu = (float *)[eng->h_buf contents];
        memcpy(h_gpu, h, n2 * sizeof(float));

        /* Bin molecules on CPU, write directly into shared GPU buffers. */
        bin_molecules_f(tcr_pos, n_tcr, grid_size, dx,
                        (int *)[eng->tcr_count_buf contents]);
        bin_molecules_f(cd45_pos, n_cd45, grid_size, dx,
                        (int *)[eng->cd45_count_buf contents]);

        /* Set params for color 0. */
        GridParams *p0 = (GridParams *)[eng->params_buf contents];
        p0->grid_size = grid_size;
        p0->kappa = (float)kappa;
        p0->dx = (float)dx;
        p0->step_size_h = (float)step_size_h;
        p0->u_assoc = (float)u_assoc;
        p0->sigma_bind = (float)sigma_bind;
        p0->cd45_height = (float)cd45_height;
        p0->color = 0;
        p0->rng_key0 = eng->rng_key0;
        p0->rng_key1 = eng->rng_key1;
        p0->rng_offset = eng->rng_counter++;

        /* Set params for color 1. */
        GridParams *p1 = (GridParams *)[eng->params_buf2 contents];
        *p1 = *p0;
        p1->color = 1;
        p1->rng_offset = eng->rng_counter++;

        /* Reset accept counter. */
        int *acc = (int *)[eng->accept_buf contents];
        *acc = 0;

        /* Single command buffer for both checkerboard colors. */
        MTLSize gridSize = MTLSizeMake(half, 1, 1);
        NSUInteger threadWidth = eng->pipeline.maxTotalThreadsPerThreadgroup;
        if (threadWidth > (NSUInteger)half)
            threadWidth = (NSUInteger)half;
        MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);

        id<MTLCommandBuffer> cmdBuf = [eng->queue commandBuffer];

        /* Color 0 (red). */
        id<MTLComputeCommandEncoder> enc0 = [cmdBuf computeCommandEncoder];
        [enc0 setComputePipelineState:eng->pipeline];
        [enc0 setBuffer:eng->h_buf offset:0 atIndex:0];
        [enc0 setBuffer:eng->tcr_count_buf offset:0 atIndex:1];
        [enc0 setBuffer:eng->cd45_count_buf offset:0 atIndex:2];
        [enc0 setBuffer:eng->params_buf offset:0 atIndex:3];
        [enc0 setBuffer:eng->accept_buf offset:0 atIndex:4];
        [enc0 dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc0 endEncoding];

        /* Color 1 (black). */
        id<MTLComputeCommandEncoder> enc1 = [cmdBuf computeCommandEncoder];
        [enc1 setComputePipelineState:eng->pipeline];
        [enc1 setBuffer:eng->h_buf offset:0 atIndex:0];
        [enc1 setBuffer:eng->tcr_count_buf offset:0 atIndex:1];
        [enc1 setBuffer:eng->cd45_count_buf offset:0 atIndex:2];
        [enc1 setBuffer:eng->params_buf2 offset:0 atIndex:3];
        [enc1 setBuffer:eng->accept_buf offset:0 atIndex:4];
        [enc1 dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc1 endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        *accepted += *acc;
        *total_proposals += n2;

        /* Copy results back from shared GPU buffer to h. */
        memcpy(h, h_gpu, n2 * sizeof(float));
    }
}
