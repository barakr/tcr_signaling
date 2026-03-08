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
    float k_rep;
    int color;
    uint32_t rng_key0;
    uint32_t rng_key1;
    uint32_t rng_offset;
} GridParams;

/* Per-cell proposal data matching the shader struct. */
typedef struct {
    float old_h;
    float u_accept;
    int accepted;
} CellProposal;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> propose_pipeline;
    id<MTLComputePipelineState> snapshot_pipeline;
    id<MTLComputePipelineState> evaluate_pipeline;
    id<MTLComputePipelineState> apply_pipeline;

    /* Buffers (sized for max grid_size). */
    id<MTLBuffer> h_buf;              /* float[n^2] — authoritative height field */
    id<MTLBuffer> h_snap_buf;         /* float[n^2] — frozen snapshot for evaluate */
    id<MTLBuffer> tcr_count_buf;      /* int[n^2]   — molecule count per cell */
    id<MTLBuffer> cd45_count_buf;     /* int[n^2]   — molecule count per cell */
    id<MTLBuffer> pmhc_count_buf;     /* int[n^2]   — pMHC count per cell */
    id<MTLBuffer> params_buf;         /* GridParams — color 0 */
    id<MTLBuffer> params_buf2;        /* GridParams — color 1 */
    id<MTLBuffer> accept_buf;         /* atomic_int[1] */
    id<MTLBuffer> proposals_buf;      /* CellProposal[n^2/2] */

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

        /* Load all four kernel functions. */
        NSArray *names = @[@"grid_propose_kernel", @"grid_snapshot_kernel",
                           @"grid_evaluate_kernel", @"grid_apply_kernel"];
        id<MTLFunction> funcs[4];
        for (int i = 0; i < 4; i++) {
            funcs[i] = [library newFunctionWithName:names[i]];
            if (!funcs[i]) {
                fprintf(stderr, "Metal: kernel '%s' not found.\n",
                        [names[i] UTF8String]);
                free(eng);
                return NULL;
            }
        }

        eng->propose_pipeline = [device newComputePipelineStateWithFunction:funcs[0] error:&error];
        eng->snapshot_pipeline = [device newComputePipelineStateWithFunction:funcs[1] error:&error];
        eng->evaluate_pipeline = [device newComputePipelineStateWithFunction:funcs[2] error:&error];
        eng->apply_pipeline = [device newComputePipelineStateWithFunction:funcs[3] error:&error];
        if (!eng->propose_pipeline || !eng->snapshot_pipeline ||
            !eng->evaluate_pipeline || !eng->apply_pipeline) {
            fprintf(stderr, "Metal: pipeline creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(eng);
            return NULL;
        }

        /* Allocate buffers. */
        int n2 = grid_size * grid_size;
        int half = n2 / 2;
        eng->h_buf = [device newBufferWithLength:n2 * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        eng->h_snap_buf = [device newBufferWithLength:n2 * sizeof(float)
                                              options:MTLResourceStorageModeShared];
        eng->tcr_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                 options:MTLResourceStorageModeShared];
        eng->cd45_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                  options:MTLResourceStorageModeShared];
        eng->pmhc_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                  options:MTLResourceStorageModeShared];
        eng->params_buf = [device newBufferWithLength:sizeof(GridParams)
                                              options:MTLResourceStorageModeShared];
        eng->params_buf2 = [device newBufferWithLength:sizeof(GridParams)
                                               options:MTLResourceStorageModeShared];
        eng->accept_buf = [device newBufferWithLength:sizeof(int)
                                              options:MTLResourceStorageModeShared];
        eng->proposals_buf = [device newBufferWithLength:half * sizeof(CellProposal)
                                                 options:MTLResourceStorageModeShared];

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
                              double cd45_height, double k_rep,
                              const double *tcr_pos, int n_tcr,
                              const double *cd45_pos, int n_cd45,
                              const int *pmhc_count,
                              long *accepted, long *total_proposals) {
    @autoreleasepool {
        MetalEngine *eng = (MetalEngine *)ctx;
        int n2 = grid_size * grid_size;
        int half = n2 / 2;

        /* Copy h into GPU shared buffer. */
        float *h_gpu = (float *)[eng->h_buf contents];
        memcpy(h_gpu, h, n2 * sizeof(float));

        /* Bin molecules on CPU. */
        bin_molecules_f(tcr_pos, n_tcr, grid_size, dx,
                        (int *)[eng->tcr_count_buf contents]);
        bin_molecules_f(cd45_pos, n_cd45, grid_size, dx,
                        (int *)[eng->cd45_count_buf contents]);

        /* Copy pMHC counts (or fill with 1s if NULL = all cells have pMHC). */
        int *pmhc_gpu = (int *)[eng->pmhc_count_buf contents];
        if (pmhc_count) {
            memcpy(pmhc_gpu, pmhc_count, n2 * sizeof(int));
        } else {
            for (int i = 0; i < n2; i++) pmhc_gpu[i] = 1;
        }

        /* Set params for color 0. */
        GridParams *p0 = (GridParams *)[eng->params_buf contents];
        p0->grid_size = grid_size;
        p0->kappa = (float)kappa;
        p0->dx = (float)dx;
        p0->step_size_h = (float)step_size_h;
        p0->u_assoc = (float)u_assoc;
        p0->sigma_bind = (float)sigma_bind;
        p0->cd45_height = (float)cd45_height;
        p0->k_rep = (float)k_rep;
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

        /* Thread group sizes. */
        MTLSize halfGrid = MTLSizeMake(half, 1, 1);
        MTLSize fullGrid = MTLSizeMake(n2, 1, 1);

        NSUInteger pw = eng->propose_pipeline.maxTotalThreadsPerThreadgroup;
        if (pw > (NSUInteger)half) pw = (NSUInteger)half;
        NSUInteger sw = eng->snapshot_pipeline.maxTotalThreadsPerThreadgroup;
        if (sw > (NSUInteger)n2) sw = (NSUInteger)n2;
        NSUInteger ew = eng->evaluate_pipeline.maxTotalThreadsPerThreadgroup;
        if (ew > (NSUInteger)half) ew = (NSUInteger)half;
        NSUInteger aw = eng->apply_pipeline.maxTotalThreadsPerThreadgroup;
        if (aw > (NSUInteger)half) aw = (NSUInteger)half;

        MTLSize ptg = MTLSizeMake(pw, 1, 1);
        MTLSize stg = MTLSizeMake(sw, 1, 1);
        MTLSize etg = MTLSizeMake(ew, 1, 1);
        MTLSize atg = MTLSizeMake(aw, 1, 1);

        id<MTLCommandBuffer> cmdBuf = [eng->queue commandBuffer];

        /* For each color: propose → snapshot → evaluate → apply.
           Each is a separate command encoder, providing barriers between phases. */
        id<MTLBuffer> paramsBufs[2] = {eng->params_buf, eng->params_buf2};
        for (int c = 0; c < 2; c++) {
            /* 1. Propose: write all proposals to h[], save old_h/u_accept. */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:eng->propose_pipeline];
                [enc setBuffer:eng->h_buf offset:0 atIndex:0];
                [enc setBuffer:paramsBufs[c] offset:0 atIndex:1];
                [enc setBuffer:eng->proposals_buf offset:0 atIndex:2];
                [enc dispatchThreads:halfGrid threadsPerThreadgroup:ptg];
                [enc endEncoding];
            }

            /* 2. Snapshot: copy h[] → h_snap[] for consistent reads. */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:eng->snapshot_pipeline];
                [enc setBuffer:eng->h_buf offset:0 atIndex:0];
                [enc setBuffer:eng->h_snap_buf offset:0 atIndex:1];
                [enc dispatchThreads:fullGrid threadsPerThreadgroup:stg];
                [enc endEncoding];
            }

            /* 3. Evaluate: compute bending_delta from snapshot, decide accept/reject. */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:eng->evaluate_pipeline];
                [enc setBuffer:eng->h_snap_buf offset:0 atIndex:0];
                [enc setBuffer:eng->tcr_count_buf offset:0 atIndex:1];
                [enc setBuffer:eng->cd45_count_buf offset:0 atIndex:2];
                [enc setBuffer:paramsBufs[c] offset:0 atIndex:3];
                [enc setBuffer:eng->accept_buf offset:0 atIndex:4];
                [enc setBuffer:eng->proposals_buf offset:0 atIndex:5];
                [enc setBuffer:eng->pmhc_count_buf offset:0 atIndex:6];
                [enc dispatchThreads:halfGrid threadsPerThreadgroup:etg];
                [enc endEncoding];
            }

            /* 4. Apply: restore rejected cells in h[]. */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:eng->apply_pipeline];
                [enc setBuffer:eng->h_buf offset:0 atIndex:0];
                [enc setBuffer:paramsBufs[c] offset:0 atIndex:1];
                [enc setBuffer:eng->proposals_buf offset:0 atIndex:2];
                [enc dispatchThreads:halfGrid threadsPerThreadgroup:atg];
                [enc endEncoding];
            }
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        *accepted += *acc;
        *total_proposals += n2;

        /* Copy results back. */
        memcpy(h, h_gpu, n2 * sizeof(float));
    }
}
