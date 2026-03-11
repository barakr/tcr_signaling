#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "gpu_engine.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef KS_PROFILE
extern double _profile_bin_ms;
static double _metal_clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

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
} CellProposal;

/* Parameters for GPU binning kernel. */
typedef struct {
    int grid_size;
    float inv_dx;
    int n_mol;
} BinParams;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> propose_pipeline;
    id<MTLComputePipelineState> evaluate_apply_pipeline;
    id<MTLComputePipelineState> bin_pipeline;

    /* Buffers (sized for max grid_size). */
    id<MTLBuffer> h_buf;              /* float[n^2] — authoritative height field */
    id<MTLBuffer> tcr_count_buf;      /* int[n^2]   — molecule count per cell */
    id<MTLBuffer> cd45_count_buf;     /* int[n^2]   — molecule count per cell */
    id<MTLBuffer> pmhc_count_buf;     /* int[n^2]   — pMHC count per cell */
    id<MTLBuffer> params_buf;         /* GridParams */
    id<MTLBuffer> accept_buf;         /* atomic_int[1] */
    id<MTLBuffer> proposals_buf;      /* CellProposal[n^2/2] */

    /* GPU binning: molecule positions as float (converted from double on CPU). */
    id<MTLBuffer> tcr_pos_buf;        /* float[n_tcr*2] */
    id<MTLBuffer> cd45_pos_buf;       /* float[n_cd45*2] */
    id<MTLBuffer> bin_params_buf;     /* BinParams */
    int max_tcr;
    int max_cd45;

    /* Async dispatch: previous command buffer (NULL if none pending). */
    id<MTLCommandBuffer> pending_cmd;

    int grid_size;
    uint32_t rng_key0;   /* Philox key derived from CPU seed */
    uint32_t rng_key1;
    uint32_t rng_counter; /* Monotonically increasing offset for unique streams */
} MetalEngine;

void *gpu_engine_create(int grid_size, uint64_t gpu_rng_key) {
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

        /* Try to load pre-compiled .metallib first, fall back to source. */
        NSError *error = nil;
        id<MTLLibrary> library = nil;

        NSArray *metallib_candidates = @[
            @"src/shaders.metallib",
            @"shaders.metallib",
            @"models/kinetic_segregation/src/shaders.metallib",
        ];
        NSString *execPath = [[NSBundle mainBundle] executablePath];
        if (execPath) {
            NSString *dir = [execPath stringByDeletingLastPathComponent];
            metallib_candidates = [@[[dir stringByAppendingPathComponent:@"shaders.metallib"]]
                                   arrayByAddingObjectsFromArray:metallib_candidates];
        }
        for (NSString *path in metallib_candidates) {
            NSURL *url = [NSURL fileURLWithPath:path];
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                library = [device newLibraryWithURL:url error:&error];
                if (library) {
                    fprintf(stderr, "Metal: loaded pre-compiled shader from %s\n",
                            [path UTF8String]);
                    break;
                }
            }
        }

        if (!library) {
            NSString *shaderPath = nil;
            if (execPath) {
                NSString *dir = [execPath stringByDeletingLastPathComponent];
                shaderPath = [dir stringByAppendingPathComponent:@"shaders.metal"];
            }
            if (!shaderPath || ![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
                NSArray *candidates = @[
                    @"src/shaders.metal",
                    @"shaders.metal",
                    @"models/kinetic_segregation/src/shaders.metal",
                ];
                for (NSString *c in candidates) {
                    if ([[NSFileManager defaultManager] fileExistsAtPath:c]) {
                        shaderPath = c;
                        break;
                    }
                }
            }

            if (!shaderPath || ![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
                fprintf(stderr, "Metal: shader not found, using CPU fallback.\n");
                free(eng);
                return NULL;
            }

            NSString *source = [NSString stringWithContentsOfFile:shaderPath
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
            if (!source) {
                fprintf(stderr, "Metal: failed to read shader: %s\n",
                        [[error localizedDescription] UTF8String]);
                free(eng);
                return NULL;
            }

            NSString *shaderDir = [shaderPath stringByDeletingLastPathComponent];
            NSString *physicsPath = [shaderDir stringByAppendingPathComponent:@"ks_physics.h"];
            NSString *physicsSource = [NSString stringWithContentsOfFile:physicsPath
                                                               encoding:NSUTF8StringEncoding
                                                                  error:&error];
            if (physicsSource) {
                source = [source stringByReplacingOccurrencesOfString:@"#include \"ks_physics.h\""
                                                          withString:physicsSource];
            } else {
                fprintf(stderr, "Metal: warning: could not read ks_physics.h for inline include: %s\n",
                        [[error localizedDescription] UTF8String]);
            }

            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            library = [device newLibraryWithSource:source options:opts error:&error];
            if (!library) {
                fprintf(stderr, "Metal: shader compilation failed: %s\n",
                        [[error localizedDescription] UTF8String]);
                free(eng);
                return NULL;
            }
            fprintf(stderr, "Metal: compiled shader from source (%s)\n",
                    [shaderPath UTF8String]);
        }

        /* Load kernel functions. */
        id<MTLFunction> propose_fn = [library newFunctionWithName:@"grid_propose_kernel"];
        id<MTLFunction> eval_apply_fn = [library newFunctionWithName:@"grid_evaluate_apply_kernel"];
        id<MTLFunction> bin_fn = [library newFunctionWithName:@"bin_molecules_kernel"];

        if (!propose_fn || !eval_apply_fn || !bin_fn) {
            fprintf(stderr, "Metal: kernel function not found.\n");
            free(eng);
            return NULL;
        }

        eng->propose_pipeline = [device newComputePipelineStateWithFunction:propose_fn error:&error];
        eng->evaluate_apply_pipeline = [device newComputePipelineStateWithFunction:eval_apply_fn error:&error];
        eng->bin_pipeline = [device newComputePipelineStateWithFunction:bin_fn error:&error];
        if (!eng->propose_pipeline || !eng->evaluate_apply_pipeline || !eng->bin_pipeline) {
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
        eng->tcr_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                 options:MTLResourceStorageModeShared];
        eng->cd45_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                  options:MTLResourceStorageModeShared];
        eng->pmhc_count_buf = [device newBufferWithLength:n2 * sizeof(int)
                                                  options:MTLResourceStorageModeShared];
        eng->params_buf = [device newBufferWithLength:sizeof(GridParams)
                                              options:MTLResourceStorageModeShared];
        eng->accept_buf = [device newBufferWithLength:sizeof(int)
                                              options:MTLResourceStorageModeShared];
        eng->proposals_buf = [device newBufferWithLength:half * sizeof(CellProposal)
                                                 options:MTLResourceStorageModeShared];
        eng->bin_params_buf = [device newBufferWithLength:sizeof(BinParams)
                                                  options:MTLResourceStorageModeShared];

        /* Pre-allocate molecule position buffers (will grow if needed). */
        eng->max_tcr = 256;
        eng->max_cd45 = 512;
        eng->tcr_pos_buf = [device newBufferWithLength:eng->max_tcr * 2 * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        eng->cd45_pos_buf = [device newBufferWithLength:eng->max_cd45 * 2 * sizeof(float)
                                                options:MTLResourceStorageModeShared];

        eng->pending_cmd = nil;

        fprintf(stderr, "Metal: GPU engine initialized on %s (grid=%d, 2-kernel pipeline).\n",
                [[device name] UTF8String], grid_size);
        return eng;
    }
}

void gpu_engine_destroy(void *ctx) {
    if (!ctx) return;
    MetalEngine *eng = (MetalEngine *)ctx;
    /* Wait for any pending GPU work before cleanup. */
    if (eng->pending_cmd) {
        [eng->pending_cmd waitUntilCompleted];
        eng->pending_cmd = nil;
    }
    free(eng);
}

float *gpu_engine_h_ptr(void *ctx) {
    if (!ctx) return NULL;
    MetalEngine *eng = (MetalEngine *)ctx;
    return (float *)[eng->h_buf contents];
}

/* Ensure molecule position buffer is large enough. */
static void ensure_tcr_pos_buf(MetalEngine *eng, int n_mol) {
    if (n_mol > eng->max_tcr) {
        eng->max_tcr = n_mol * 2;
        eng->tcr_pos_buf = [eng->device newBufferWithLength:eng->max_tcr * 2 * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    }
}
static void ensure_cd45_pos_buf(MetalEngine *eng, int n_mol) {
    if (n_mol > eng->max_cd45) {
        eng->max_cd45 = n_mol * 2;
        eng->cd45_pos_buf = [eng->device newBufferWithLength:eng->max_cd45 * 2 * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    }
}

/* Convert double positions to float and copy to GPU buffer. */
static void upload_positions(const double *pos, int n_mol, id<MTLBuffer> buf) {
    float *dst = (float *)[buf contents];
    for (int i = 0; i < n_mol * 2; i++) {
        dst[i] = (float)pos[i];
    }
}

void gpu_engine_grid_update(void *ctx, float *h, int grid_size,
                              double kappa, double dx, double step_size_h,
                              double u_assoc, double sigma_bind,
                              double cd45_height, double k_rep,
                              const double *tcr_pos, int n_tcr,
                              const double *cd45_pos, int n_cd45,
                              const int *pmhc_count,
                              long *accepted, long *total_proposals,
                              int n_substeps) {
    @autoreleasepool {
        MetalEngine *eng = (MetalEngine *)ctx;
        int n2 = grid_size * grid_size;
        int half = n2 / 2;
        if (n_substeps < 1) n_substeps = 1;

        /* Wait for previous async GPU work before touching shared buffers. */
        if (eng->pending_cmd) {
            [eng->pending_cmd waitUntilCompleted];
            eng->pending_cmd = nil;
        }

        (void)h;  /* h is the shared buffer directly */

#ifdef KS_PROFILE
        double _bt0 = _metal_clock_ms();
#endif
        /* Upload molecule positions to GPU (double→float conversion). */
        ensure_tcr_pos_buf(eng, n_tcr);
        ensure_cd45_pos_buf(eng, n_cd45);
        upload_positions(tcr_pos, n_tcr, eng->tcr_pos_buf);
        upload_positions(cd45_pos, n_cd45, eng->cd45_pos_buf);

        /* Copy pMHC counts (or fill with 1s if NULL = all cells have pMHC). */
        int *pmhc_gpu = (int *)[eng->pmhc_count_buf contents];
        if (pmhc_count) {
            memcpy(pmhc_gpu, pmhc_count, n2 * sizeof(int));
        } else {
            for (int i = 0; i < n2; i++) pmhc_gpu[i] = 1;
        }
#ifdef KS_PROFILE
        _profile_bin_ms += _metal_clock_ms() - _bt0;
#endif

        /* Build base params template. */
        GridParams base_params;
        base_params.grid_size = grid_size;
        base_params.kappa = (float)kappa;
        base_params.dx = (float)dx;
        base_params.step_size_h = (float)step_size_h;
        base_params.u_assoc = (float)u_assoc;
        base_params.sigma_bind = (float)sigma_bind;
        base_params.cd45_height = (float)cd45_height;
        base_params.k_rep = (float)k_rep;
        base_params.rng_key0 = eng->rng_key0;
        base_params.rng_key1 = eng->rng_key1;

        /* Thread group sizes. */
        MTLSize halfGrid = MTLSizeMake(half, 1, 1);
        NSUInteger pw = eng->propose_pipeline.maxTotalThreadsPerThreadgroup;
        if (pw > (NSUInteger)half) pw = (NSUInteger)half;
        NSUInteger ew = eng->evaluate_apply_pipeline.maxTotalThreadsPerThreadgroup;
        if (ew > (NSUInteger)half) ew = (NSUInteger)half;
        MTLSize ptg = MTLSizeMake(pw, 1, 1);
        MTLSize etg = MTLSizeMake(ew, 1, 1);

        /* Bin kernel thread setup. */
        NSUInteger bw = eng->bin_pipeline.maxTotalThreadsPerThreadgroup;
        int max_mol = n_tcr > n_cd45 ? n_tcr : n_cd45;
        if (bw > (NSUInteger)max_mol) bw = (NSUInteger)max_mol;
        if (bw < 1) bw = 1;
        MTLSize btg = MTLSizeMake(bw, 1, 1);

        float inv_dx = 1.0f / (float)dx;

        /* Encode ALL substeps × 2 colors into ONE command buffer.
           Also prepend GPU binning for each substep. */
        id<MTLCommandBuffer> cmdBuf = [eng->queue commandBuffer];

        /* Zero accept counter via blit to ensure GPU cache coherence. */
        {
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit fillBuffer:eng->accept_buf range:NSMakeRange(0, sizeof(int)) value:0];
            [blit endEncoding];
        }

        for (int sub = 0; sub < n_substeps; sub++) {
            /* GPU binning: zero counts and bin TCR + CD45 positions. */
            {
                /* Zero tcr_count and cd45_count via blit. */
                id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
                [blit fillBuffer:eng->tcr_count_buf range:NSMakeRange(0, n2 * sizeof(int)) value:0];
                [blit fillBuffer:eng->cd45_count_buf range:NSMakeRange(0, n2 * sizeof(int)) value:0];
                [blit endEncoding];
            }

            /* Bin TCR positions. */
            if (n_tcr > 0) {
                BinParams bp;
                bp.grid_size = grid_size;
                bp.inv_dx = inv_dx;
                bp.n_mol = n_tcr;
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:eng->bin_pipeline];
                [enc setBuffer:eng->tcr_pos_buf offset:0 atIndex:0];
                [enc setBuffer:eng->tcr_count_buf offset:0 atIndex:1];
                [enc setBytes:&bp length:sizeof(BinParams) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(n_tcr, 1, 1) threadsPerThreadgroup:btg];
                [enc endEncoding];
            }

            /* Bin CD45 positions. */
            if (n_cd45 > 0) {
                BinParams bp;
                bp.grid_size = grid_size;
                bp.inv_dx = inv_dx;
                bp.n_mol = n_cd45;
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:eng->bin_pipeline];
                [enc setBuffer:eng->cd45_pos_buf offset:0 atIndex:0];
                [enc setBuffer:eng->cd45_count_buf offset:0 atIndex:1];
                [enc setBytes:&bp length:sizeof(BinParams) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(n_cd45, 1, 1) threadsPerThreadgroup:btg];
                [enc endEncoding];
            }

            for (int c = 0; c < 2; c++) {
                GridParams p = base_params;
                p.color = c;
                p.rng_offset = eng->rng_counter++;

                /* 1. Propose */
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:eng->propose_pipeline];
                    [enc setBuffer:eng->h_buf offset:0 atIndex:0];
                    [enc setBytes:&p length:sizeof(GridParams) atIndex:1];
                    [enc setBuffer:eng->proposals_buf offset:0 atIndex:2];
                    [enc dispatchThreads:halfGrid threadsPerThreadgroup:ptg];
                    [enc endEncoding];
                }

                /* 2. Evaluate + Apply (fused — no snapshot needed) */
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:eng->evaluate_apply_pipeline];
                    [enc setBuffer:eng->h_buf offset:0 atIndex:0];
                    [enc setBuffer:eng->tcr_count_buf offset:0 atIndex:1];
                    [enc setBuffer:eng->cd45_count_buf offset:0 atIndex:2];
                    [enc setBytes:&p length:sizeof(GridParams) atIndex:3];
                    [enc setBuffer:eng->accept_buf offset:0 atIndex:4];
                    [enc setBuffer:eng->proposals_buf offset:0 atIndex:5];
                    [enc setBuffer:eng->pmhc_count_buf offset:0 atIndex:6];
                    [enc dispatchThreads:halfGrid threadsPerThreadgroup:etg];
                    [enc endEncoding];
                }
            }
        }

        [cmdBuf commit];

        /* Async: store pending command buffer. Caller must wait before
           reading h[] or shared buffers (done at top of next call). */
        eng->pending_cmd = cmdBuf;

        /* For accept counting, we still need to wait (acceptances are read
           by the caller immediately). Do a synchronous wait here. */
        [cmdBuf waitUntilCompleted];
        eng->pending_cmd = nil;

        int *acc = (int *)[eng->accept_buf contents];
        *accepted += *acc;
        *total_proposals += (long)n2 * n_substeps;
    }
}
