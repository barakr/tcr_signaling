#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "simulation.h"

/* MD5 hash for seed derivation, matching the Python __main__.py scheme. */
#import <CommonCrypto/CommonDigest.h>

static uint64_t derive_seed(uint64_t base_seed, double time_sec, double rigidity) {
    /* Pack two doubles and compute MD5, take first 8 hex chars -> uint32. */
    double vals[2] = {time_sec, rigidity};
    unsigned char digest[CC_MD5_DIGEST_LENGTH];
    CC_MD5(vals, sizeof(vals), digest);

    /* First 8 hex chars = first 4 bytes. */
    uint32_t hash_val = ((uint32_t)digest[0] << 24) |
                        ((uint32_t)digest[1] << 16) |
                        ((uint32_t)digest[2] << 8)  |
                        ((uint32_t)digest[3]);
    return base_seed + hash_val;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --time_sec FLOAT --rigidity_kT_nm2 FLOAT --run-dir PATH\n"
                    "       [--seed INT] [--n_tcr INT] [--n_cd45 INT] [--n_steps INT]\n"
                    "       [--grid_size INT] [--no-gpu] [--dump-frames] [--dump-interval INT]\n"
                    "       [--D_mol FLOAT] [--D_h FLOAT] [--dt FLOAT]\n", prog);
}

/* Write a single frame: height field (float[n^2]) + molecule positions. */
static void dump_frame(const SimState *sim, const char *frames_dir, int step) {
    char path[512];
    int n = sim->grid_size;
    int n2 = n * n;

    /* Write height field as raw float32 binary. */
    snprintf(path, sizeof(path), "%s/h_%05d.bin", frames_dir, step);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(sim->h, sizeof(float), n2, f);
        fclose(f);
    }

    /* Write molecule positions as raw float64 binary: [n_tcr*2 doubles, n_cd45*2 doubles]. */
    snprintf(path, sizeof(path), "%s/mol_%05d.bin", frames_dir, step);
    f = fopen(path, "wb");
    if (f) {
        fwrite(sim->tcr_pos, sizeof(double), sim->n_tcr * 2, f);
        fwrite(sim->cd45_pos, sizeof(double), sim->n_cd45 * 2, f);
        fclose(f);
    }
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        double time_sec = -1, rigidity = -1;
        int seed = 42, n_tcr = 50, n_cd45 = 100, grid_size = 64;
        int n_steps_arg = -1;
        int use_gpu = 1;
        int dump_frames = 0;
        int dump_interval = 1;
        double D_mol_arg = 0.0, D_h_arg = 0.0, dt_arg = -1.0;
        double cd45_height_arg = 0.0, cd45_k_rep_arg = 0.0;
        double mol_repulsion_eps_arg = 0.0, mol_repulsion_rcut_arg = 0.0;
        int n_pmhc_arg = 0;
        int pmhc_seed_arg = -1;
        const char *run_dir = NULL;

        /* Simple argument parsing. */
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--time_sec") == 0 && i + 1 < argc)
                time_sec = atof(argv[++i]);
            else if (strcmp(argv[i], "--rigidity_kT_nm2") == 0 && i + 1 < argc)
                rigidity = atof(argv[++i]);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
                seed = atoi(argv[++i]);
            else if (strcmp(argv[i], "--run-dir") == 0 && i + 1 < argc)
                run_dir = argv[++i];
            else if (strcmp(argv[i], "--n_tcr") == 0 && i + 1 < argc)
                n_tcr = (int)atof(argv[++i]);
            else if (strcmp(argv[i], "--n_cd45") == 0 && i + 1 < argc)
                n_cd45 = (int)atof(argv[++i]);
            else if (strcmp(argv[i], "--n_steps") == 0 && i + 1 < argc)
                n_steps_arg = (int)atof(argv[++i]);
            else if (strcmp(argv[i], "--grid_size") == 0 && i + 1 < argc)
                grid_size = (int)atof(argv[++i]);
            else if (strcmp(argv[i], "--no-gpu") == 0)
                use_gpu = 0;
            else if (strcmp(argv[i], "--dump-frames") == 0)
                dump_frames = 1;
            else if (strcmp(argv[i], "--dump-interval") == 0 && i + 1 < argc)
                dump_interval = (int)atof(argv[++i]);
            else if (strcmp(argv[i], "--D_mol") == 0 && i + 1 < argc)
                D_mol_arg = atof(argv[++i]);
            else if (strcmp(argv[i], "--D_h") == 0 && i + 1 < argc)
                D_h_arg = atof(argv[++i]);
            else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc)
                dt_arg = atof(argv[++i]);
            else if (strcmp(argv[i], "--cd45_height") == 0 && i + 1 < argc)
                cd45_height_arg = atof(argv[++i]);
            else if (strcmp(argv[i], "--cd45_k_rep") == 0 && i + 1 < argc)
                cd45_k_rep_arg = atof(argv[++i]);
            else if (strcmp(argv[i], "--mol_repulsion_eps") == 0 && i + 1 < argc)
                mol_repulsion_eps_arg = atof(argv[++i]);
            else if (strcmp(argv[i], "--mol_repulsion_rcut") == 0 && i + 1 < argc)
                mol_repulsion_rcut_arg = atof(argv[++i]);
            else if (strcmp(argv[i], "--n_pmhc") == 0 && i + 1 < argc)
                n_pmhc_arg = (int)atof(argv[++i]);
            else if (strcmp(argv[i], "--pmhc_seed") == 0 && i + 1 < argc)
                pmhc_seed_arg = atoi(argv[++i]);
            else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            }
        }

        if (time_sec < 0 || rigidity < 0 || !run_dir) {
            print_usage(argv[0]);
            return 1;
        }

        /* Derive seed (matching Python). */
        uint64_t point_seed = derive_seed((uint64_t)seed, time_sec, rigidity);

        /* Create output directory. */
        NSString *outDir = [NSString stringWithFormat:@"%s/out", run_dir];
        [[NSFileManager defaultManager] createDirectoryAtPath:outDir
                                  withIntermediateDirectories:YES
                                                  attributes:nil
                                                       error:nil];

        uint64_t pmhc_sd = (pmhc_seed_arg >= 0) ? (uint64_t)pmhc_seed_arg : (point_seed + 1);
        SimState *sim = sim_create(grid_size, n_tcr, n_cd45,
                                   rigidity, U_ASSOC_DEFAULT, point_seed,
                                   use_gpu, D_mol_arg, D_h_arg, dt_arg,
                                   cd45_height_arg, cd45_k_rep_arg,
                                   mol_repulsion_eps_arg, mol_repulsion_rcut_arg,
                                   n_pmhc_arg, pmhc_sd);

        /* Compute n_steps: explicit override or auto from time_sec / dt. */
        int n_steps;
        if (n_steps_arg > 0) {
            n_steps = n_steps_arg;
        } else {
            n_steps = (int)round(time_sec / sim->dt);
            if (n_steps < 50) n_steps = 50;
        }

        if (dump_frames) {
            NSString *framesDir = [NSString stringWithFormat:@"%s/frames", run_dir];
            [[NSFileManager defaultManager] createDirectoryAtPath:framesDir
                                      withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil];
            const char *fd = [framesDir UTF8String];

            if (dump_interval < 1) dump_interval = 1;
            int n_frames = n_steps / dump_interval;

            /* Write metadata. */
            char meta_path[512];
            snprintf(meta_path, sizeof(meta_path), "%s/meta.json", fd);
            FILE *mf = fopen(meta_path, "w");
            if (mf) {
                fprintf(mf, "{\"grid_size\":%d,\"n_tcr\":%d,\"n_cd45\":%d,"
                        "\"n_steps\":%d,\"dx\":%.6f,\"patch_nm\":%.1f,"
                        "\"dump_interval\":%d,\"n_frames\":%d,"
                        "\"dt\":%.10g,\"time_sec\":%.6f}\n",
                        grid_size, n_tcr, n_cd45, n_steps, sim->dx, PATCH_SIZE_NM,
                        dump_interval, n_frames, sim->dt, time_sec);
                fclose(mf);
            }

            /* Dump initial state as frame 0. */
            dump_frame(sim, fd, 0);
            sim->n_steps = n_steps;
            int frame_idx = 1;
            for (int step = 1; step <= n_steps; step++) {
                sim_step(sim);
                if (step % dump_interval == 0) {
                    dump_frame(sim, fd, frame_idx);
                    frame_idx++;
                }
            }
        } else {
            sim_run(sim, n_steps);
        }

        double depletion = sim_depletion_width(sim);
        double tcr_mean_r = sim_mean_r(sim->tcr_pos, sim->n_tcr);
        double cd45_mean_r = sim_mean_r(sim->cd45_pos, sim->n_cd45);
        double accept_rate = (sim->total_proposals > 0)
            ? (double)sim->accepted / sim->total_proposals : 0.0;

        /* JSON output. */
        NSString *json = [NSString stringWithFormat:
            @"{\n"
            @"  \"depletion_width_nm\": %.6f,\n"
            @"  \"diagnostics\": {\n"
            @"    \"D_h_nm2_per_s\": %.1f,\n"
            @"    \"D_mol_nm2_per_s\": %.1f,\n"
            @"    \"accept_rate\": %.6f,\n"
            @"    \"dt_seconds\": %.10g,\n"
            @"    \"final_cd45_mean_r_nm\": %.6f,\n"
            @"    \"final_tcr_mean_r_nm\": %.6f,\n"
            @"    \"n_steps_actual\": %d,\n"
            @"    \"step_size_h_nm\": %.6f,\n"
            @"    \"step_size_mol_nm\": %.6f\n"
            @"  },\n"
            @"  \"inputs\": {\n"
            @"    \"rigidity_kT_nm2\": %.1f,\n"
            @"    \"time_sec\": %.1f\n"
            @"  }\n"
            @"}",
            depletion,
            sim->D_h, sim->D_mol, accept_rate, sim->dt,
            cd45_mean_r, tcr_mean_r, n_steps,
            sim->step_size_h, sim->step_size_mol,
            rigidity, time_sec];

        /* Write to file. */
        NSString *outFile = [outDir stringByAppendingPathComponent:@"segregation.json"];
        [json writeToFile:outFile atomically:YES encoding:NSUTF8StringEncoding error:nil];

        /* Print to stdout. */
        printf("%s\n", [json UTF8String]);

        sim_destroy(sim);
        return 0;
    }
}
