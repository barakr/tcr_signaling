/*
 * main.cpp — CLI entry point for the kinetic segregation simulation.
 *
 * C++20.  Platform-independent (no ObjC/Foundation dependencies).
 * The simulation core (simulation.h) is pure C and linked via extern "C".
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "nlohmann/json.hpp"

extern "C" {
#include "simulation.h"
}

namespace fs = std::filesystem;
using json = nlohmann::json;

/* ---------- FNV-1a seed derivation (portable, deterministic) ---------- */

static uint64_t fnv1a_bytes(const void *data, size_t len) {
    auto p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xcbf29ce484222325ULL;  /* FNV offset basis */
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;  /* FNV prime */
    }
    return h;
}

static uint64_t derive_seed(uint64_t base_seed, double time_sec, double rigidity) {
    double vals[2] = {time_sec, rigidity};
    uint64_t hash_val = fnv1a_bytes(vals, sizeof(vals));
    return base_seed + (hash_val & 0xFFFFFFFFULL);
}

/* ---------- Usage ---------- */

static void print_usage(std::string_view prog) {
    std::cerr << "Usage: " << prog
              << " --time_sec FLOAT --rigidity_kT FLOAT --run-dir PATH\n"
              << "       [--seed INT] [--n_tcr INT] [--n_cd45 INT] [--n_steps INT]\n"
              << "       [--grid_size INT] [--no-gpu] [--dump-frames] [--dump-interval INT]\n"
              << "       [--monitor-binding FLOAT] [--monitor-interval INT]\n"
              << "       [--grid-substeps INT] [--D_mol FLOAT] [--D_h FLOAT]\n"
              << "       [--dt FLOAT] [--dt_factor FLOAT]\n"
              << "       [--params FILE] [--pmhc_mode MODE] [--pmhc_radius FLOAT]\n"
              << "       [--u_assoc FLOAT] [--sigma_bind FLOAT] [--sigma_r FLOAT]\n"
              << "       [--patch_size FLOAT]\n";
}

/* ---------- Frame dump (raw binary I/O) ---------- */

static void dump_frame(const SimState *sim, const fs::path &frames_dir, int step) {
    int n = sim->grid_size;
    int n2 = n * n;
    char name[32];

    /* Height field as raw float32 binary. */
    std::snprintf(name, sizeof(name), "h_%05d.bin", step);
    if (std::ofstream ofs(frames_dir / name, std::ios::binary); ofs) {
        ofs.write(reinterpret_cast<const char *>(sim->h), n2 * sizeof(float));
    }

    /* Molecule positions as raw float64 binary. */
    std::snprintf(name, sizeof(name), "mol_%05d.bin", step);
    if (std::ofstream ofs(frames_dir / name, std::ios::binary); ofs) {
        ofs.write(reinterpret_cast<const char *>(sim->tcr_pos),
                  sim->n_tcr * 2 * sizeof(double));
        ofs.write(reinterpret_cast<const char *>(sim->cd45_pos),
                  sim->n_cd45 * 2 * sizeof(double));
    }
}

/* ---------- Binding monitor (lightweight, no I/O) ---------- */

static double compute_bound_fraction(const SimState *sim, double threshold) {
    if (sim->n_pmhc <= 0 || sim->n_tcr <= 0 || !sim->pmhc_pos)
        return 0.0;
    double thr2 = threshold * threshold;
    double half = sim->patch_size / 2.0;
    double ps = sim->patch_size;
    int bound = 0;
    for (int t = 0; t < sim->n_tcr; t++) {
        double tx = sim->tcr_pos[t * 2];
        double ty = sim->tcr_pos[t * 2 + 1];
        for (int p = 0; p < sim->n_pmhc; p++) {
            double ddx = tx - sim->pmhc_pos[p * 2];
            double ddy = ty - sim->pmhc_pos[p * 2 + 1];
            if (ddx > half) ddx -= ps;
            else if (ddx < -half) ddx += ps;
            if (ddy > half) ddy -= ps;
            else if (ddy < -half) ddy += ps;
            if (ddx * ddx + ddy * ddy < thr2) { bound++; break; }
        }
    }
    return static_cast<double>(bound) / sim->n_tcr;
}

/* ---------- CLI argument helpers ---------- */

static bool match(const char *arg, std::string_view flag) {
    return flag == arg;
}

/* ---------- JSON param file loading ---------- */

static void load_params_file(const std::string &path,
                             double &time_sec, double &rigidity,
                             int &seed, int &n_tcr, int &n_cd45,
                             int &n_steps_arg, int &grid_size,
                             double &D_mol_arg, double &D_h_arg, double &dt_arg,
                             double &dt_factor_arg,
                             double &cd45_height_arg, double &cd45_k_rep_arg,
                             double &mol_repulsion_eps_arg, double &mol_repulsion_rcut_arg,
                             int &n_pmhc_arg, int &pmhc_seed_arg,
                             int &pmhc_mode_arg, double &pmhc_radius_arg,
                             int &binding_mode_arg, int &step_mode_arg,
                             double &h0_tcr_arg, double &init_height_arg,
                             double &u_assoc_arg, double &sigma_bind_arg,
                             double &sigma_r_arg, double &patch_size_arg) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Error: cannot read param file: " << path << "\n";
        std::exit(1);
    }
    json params;
    try {
        params = json::parse(ifs);
    } catch (const json::parse_error &) {
        std::cerr << "Error: invalid JSON in param file: " << path << "\n";
        std::exit(1);
    }

    auto get_d = [&](const char *key, double sentinel, double &val) {
        if (val == sentinel && params.contains(key))
            val = params[key].get<double>();
    };
    auto get_i = [&](const char *key, int sentinel, int &val) {
        if (val == sentinel && params.contains(key))
            val = params[key].get<int>();
    };

    get_d("time_sec", -1.0, time_sec);
    get_d("rigidity_kT", -1.0, rigidity);
    get_d("rigidity_kT_nm2", -1.0, rigidity);  /* deprecated alias */
    get_i("seed", 42, seed);
    get_i("n_tcr", 125, n_tcr);
    get_i("n_cd45", 500, n_cd45);
    get_i("n_steps", -1, n_steps_arg);
    get_i("grid_size", 64, grid_size);
    get_d("D_mol", 0.0, D_mol_arg);
    get_d("D_h", 0.0, D_h_arg);
    get_d("dt", -1.0, dt_arg);
    get_d("dt_factor", 0.0, dt_factor_arg);
    get_d("cd45_height", 0.0, cd45_height_arg);
    get_d("cd45_k_rep", 0.0, cd45_k_rep_arg);
    get_d("mol_repulsion_eps", 0.0, mol_repulsion_eps_arg);
    get_d("mol_repulsion_rcut", 0.0, mol_repulsion_rcut_arg);
    get_i("n_pmhc", 0, n_pmhc_arg);
    get_i("pmhc_seed", -1, pmhc_seed_arg);
    get_d("pmhc_radius", 0.0, pmhc_radius_arg);
    get_d("h0_tcr", 0.0, h0_tcr_arg);
    get_d("init_height", 0.0, init_height_arg);
    get_d("u_assoc", 0.0, u_assoc_arg);
    get_d("sigma_bind", 0.0, sigma_bind_arg);
    get_d("sigma_r", 0.0, sigma_r_arg);
    get_d("patch_size", 0.0, patch_size_arg);

    if (params.contains("pmhc_mode")) {
        auto mode = params["pmhc_mode"].get<std::string>();
        pmhc_mode_arg = (mode == "uniform") ? 0 : 1;
    }
    if (params.contains("binding_mode")) {
        auto mode = params["binding_mode"].get<std::string>();
        binding_mode_arg = (mode == "gaussian") ? BINDING_MODE_GAUSSIAN : BINDING_MODE_FORCED;
    }
    if (params.contains("step_mode")) {
        auto mode = params["step_mode"].get<std::string>();
        step_mode_arg = (mode == "brownian") ? STEP_MODE_BROWNIAN : STEP_MODE_PAPER;
    }
}

/* ---------- Main ---------- */

int main(int argc, const char *argv[]) {
    double time_sec = -1, rigidity = -1;
    int seed = 42, n_tcr = 125, n_cd45 = 500, grid_size = 64;
    int n_steps_arg = -1;
    int use_gpu = 1;
    bool dump_frames_flag = false;
    int dump_interval = 1;
    int grid_substeps = 1;
    double monitor_binding_threshold = 0.0;  /* 0 = disabled */
    int monitor_interval = 1;
    double D_mol_arg = 0.0, D_h_arg = 0.0, dt_arg = -1.0, dt_factor_arg = 0.0;
    double cd45_height_arg = 0.0, cd45_k_rep_arg = 0.0;
    double mol_repulsion_eps_arg = 0.0, mol_repulsion_rcut_arg = 0.0;
    int n_pmhc_arg = 0;
    int pmhc_seed_arg = -1;
    int pmhc_mode_arg = PMHC_MODE_UNIFORM;
    double pmhc_radius_arg = 0.0;
    int binding_mode_arg = BINDING_MODE_GAUSSIAN;
    int step_mode_arg = STEP_MODE_BROWNIAN;
    double h0_tcr_arg = 0.0;
    double init_height_arg = 0.0;
    double u_assoc_arg = 0.0;
    double sigma_bind_arg = 0.0;
    double sigma_r_arg = 0.0;
    double patch_size_arg = 0.0;
    std::string params_file;
    std::string run_dir;

    /* Argument parsing. */
    for (int i = 1; i < argc; i++) {
        if (match(argv[i], "--time_sec") && i + 1 < argc)
            time_sec = std::atof(argv[++i]);
        else if ((match(argv[i], "--rigidity_kT") ||
                  match(argv[i], "--rigidity_kT_nm2")) && i + 1 < argc)
            rigidity = std::atof(argv[++i]);
        else if (match(argv[i], "--seed") && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else if (match(argv[i], "--run-dir") && i + 1 < argc)
            run_dir = argv[++i];
        else if (match(argv[i], "--n_tcr") && i + 1 < argc)
            n_tcr = static_cast<int>(std::atof(argv[++i]));
        else if (match(argv[i], "--n_cd45") && i + 1 < argc)
            n_cd45 = static_cast<int>(std::atof(argv[++i]));
        else if (match(argv[i], "--n_steps") && i + 1 < argc)
            n_steps_arg = static_cast<int>(std::atof(argv[++i]));
        else if (match(argv[i], "--grid_size") && i + 1 < argc)
            grid_size = static_cast<int>(std::atof(argv[++i]));
        else if (match(argv[i], "--no-gpu"))
            use_gpu = 0;
        else if (match(argv[i], "--dump-frames"))
            dump_frames_flag = true;
        else if (match(argv[i], "--dump-interval") && i + 1 < argc)
            dump_interval = static_cast<int>(std::atof(argv[++i]));
        else if (match(argv[i], "--D_mol") && i + 1 < argc)
            D_mol_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--D_h") && i + 1 < argc)
            D_h_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--dt") && i + 1 < argc)
            dt_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--dt_factor") && i + 1 < argc)
            dt_factor_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--cd45_height") && i + 1 < argc)
            cd45_height_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--cd45_k_rep") && i + 1 < argc)
            cd45_k_rep_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--mol_repulsion_eps") && i + 1 < argc)
            mol_repulsion_eps_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--mol_repulsion_rcut") && i + 1 < argc)
            mol_repulsion_rcut_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--n_pmhc") && i + 1 < argc)
            n_pmhc_arg = static_cast<int>(std::atof(argv[++i]));
        else if (match(argv[i], "--pmhc_seed") && i + 1 < argc)
            pmhc_seed_arg = std::atoi(argv[++i]);
        else if (match(argv[i], "--pmhc_mode") && i + 1 < argc) {
            ++i;
            pmhc_mode_arg = match(argv[i], "uniform") ? 0 : 1;
        }
        else if (match(argv[i], "--pmhc_radius") && i + 1 < argc)
            pmhc_radius_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--params") && i + 1 < argc)
            params_file = argv[++i];
        else if (match(argv[i], "--binding_mode") && i + 1 < argc) {
            ++i;
            binding_mode_arg = match(argv[i], "gaussian") ? BINDING_MODE_GAUSSIAN : BINDING_MODE_FORCED;
        }
        else if (match(argv[i], "--step_mode") && i + 1 < argc) {
            ++i;
            step_mode_arg = match(argv[i], "brownian") ? STEP_MODE_BROWNIAN : STEP_MODE_PAPER;
        }
        else if (match(argv[i], "--h0_tcr") && i + 1 < argc)
            h0_tcr_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--init_height") && i + 1 < argc)
            init_height_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--u_assoc") && i + 1 < argc)
            u_assoc_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--sigma_bind") && i + 1 < argc)
            sigma_bind_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--sigma_r") && i + 1 < argc)
            sigma_r_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--patch_size") && i + 1 < argc)
            patch_size_arg = std::atof(argv[++i]);
        else if (match(argv[i], "--grid-substeps") && i + 1 < argc)
            grid_substeps = std::atoi(argv[++i]);
        else if (match(argv[i], "--monitor-binding") && i + 1 < argc)
            monitor_binding_threshold = std::atof(argv[++i]);
        else if (match(argv[i], "--monitor-interval") && i + 1 < argc)
            monitor_interval = static_cast<int>(std::atof(argv[++i]));
        else if (match(argv[i], "--help") || match(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Load JSON param file if given (CLI values already set override). */
    if (!params_file.empty()) {
        load_params_file(params_file,
                         time_sec, rigidity, seed, n_tcr, n_cd45,
                         n_steps_arg, grid_size, D_mol_arg, D_h_arg, dt_arg,
                         dt_factor_arg, cd45_height_arg, cd45_k_rep_arg,
                         mol_repulsion_eps_arg, mol_repulsion_rcut_arg,
                         n_pmhc_arg, pmhc_seed_arg,
                         pmhc_mode_arg, pmhc_radius_arg,
                         binding_mode_arg, step_mode_arg,
                         h0_tcr_arg, init_height_arg,
                         u_assoc_arg, sigma_bind_arg,
                         sigma_r_arg, patch_size_arg);
    }

    if (time_sec < 0 || rigidity < 0 || run_dir.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    /* --dt and --dt_factor are mutually exclusive. */
    if (dt_arg > 0.0 && dt_factor_arg > 0.0) {
        std::cerr << "Error: --dt and --dt_factor are mutually exclusive\n";
        return 1;
    }

    /* Derive seed. */
    auto point_seed = derive_seed(static_cast<uint64_t>(seed), time_sec, rigidity);

    /* Create output directory. */
    auto out_dir = fs::path(run_dir) / "out";
    fs::create_directories(out_dir);

    uint64_t pmhc_sd = (pmhc_seed_arg >= 0)
        ? static_cast<uint64_t>(pmhc_seed_arg)
        : (point_seed + 1);
    double u_assoc_val = (u_assoc_arg > 0.0) ? u_assoc_arg : U_ASSOC_DEFAULT;
    auto *sim = sim_create(grid_size, n_tcr, n_cd45,
                           rigidity, u_assoc_val, point_seed,
                           use_gpu, D_mol_arg, D_h_arg, dt_arg,
                           dt_factor_arg, cd45_height_arg, cd45_k_rep_arg,
                           mol_repulsion_eps_arg, mol_repulsion_rcut_arg,
                           n_pmhc_arg, pmhc_sd,
                           pmhc_mode_arg, pmhc_radius_arg,
                           binding_mode_arg, step_mode_arg,
                           h0_tcr_arg, init_height_arg,
                           sigma_r_arg, sigma_bind_arg, patch_size_arg);
    if (grid_substeps > 1) sim->grid_substeps = grid_substeps;
    sim->bind_threshold = monitor_binding_threshold;

    /* Compute n_steps: explicit override or auto from time_sec / dt. */
    int n_steps;
    if (n_steps_arg > 0) {
        n_steps = n_steps_arg;
    } else {
        n_steps = static_cast<int>(std::round(time_sec / sim->dt));
        if (n_steps < MIN_N_STEPS) n_steps = MIN_N_STEPS;
    }

    /* Binding monitor time series (populated if --monitor-binding is set). */
    std::vector<double> binding_timeseries;

    if (dump_frames_flag) {
        auto frames_dir = fs::path(run_dir) / "frames";
        fs::create_directories(frames_dir);

        if (dump_interval < 1) dump_interval = 1;
        int n_frames = n_steps / dump_interval;

        /* Write metadata. */
        json meta = {
            {"grid_size", grid_size}, {"n_tcr", n_tcr}, {"n_cd45", n_cd45},
            {"n_steps", n_steps}, {"dx", sim->dx}, {"patch_nm", sim->patch_size},
            {"dump_interval", dump_interval}, {"n_frames", n_frames},
            {"dt", sim->dt}, {"time_sec", time_sec},
            {"rigidity_kT", rigidity}, {"n_pmhc", sim->n_pmhc},
            {"pmhc_mode", (pmhc_mode_arg == PMHC_MODE_UNIFORM) ? "uniform" : "inner_circle"},
            {"pmhc_radius", sim->pmhc_radius}
        };
        if (std::ofstream ofs(frames_dir / "meta.json"); ofs)
            ofs << meta.dump() << "\n";

        /* Dump pMHC positions once (static). */
        if (sim->n_pmhc > 0 && sim->pmhc_pos) {
            if (std::ofstream ofs(frames_dir / "pmhc.bin", std::ios::binary); ofs)
                ofs.write(reinterpret_cast<const char *>(sim->pmhc_pos),
                          sim->n_pmhc * 2 * sizeof(double));
        }

        /* Dump initial state as frame 0. */
        dump_frame(sim, frames_dir, 0);
        sim->n_steps = n_steps;
        int frame_idx = 1;
        for (int step = 1; step <= n_steps; step++) {
            sim_step(sim);
            if (step % dump_interval == 0) {
                dump_frame(sim, frames_dir, frame_idx);
                frame_idx++;
            }
        }
    } else if (monitor_binding_threshold > 0.0) {
        /* Lightweight binding monitor: compute bound fraction at intervals,
         * no file I/O. Much faster than --dump-frames. */
        if (monitor_interval < 1) monitor_interval = 1;
        binding_timeseries.reserve(n_steps / monitor_interval + 1);
        /* Record initial state. */
        binding_timeseries.push_back(
            compute_bound_fraction(sim, monitor_binding_threshold));
        sim->n_steps = n_steps;
        for (int step = 1; step <= n_steps; step++) {
            sim_step(sim);
            if (step % monitor_interval == 0) {
                binding_timeseries.push_back(
                    compute_bound_fraction(sim, monitor_binding_threshold));
            }
        }
    } else {
        sim_run(sim, n_steps);
    }

#ifdef KS_PROFILE
    sim_profile_report(n_steps);
#endif
    double depletion = sim_depletion_width(sim);
    DepletionMetrics dm = sim_depletion_metrics(sim);
    double tcr_mean_r = sim_mean_r(sim, sim->tcr_pos, sim->n_tcr);
    double cd45_mean_r = sim_mean_r(sim, sim->cd45_pos, sim->n_cd45);
    double accept_rate = (sim->total_proposals > 0)
        ? static_cast<double>(sim->accepted) / sim->total_proposals : 0.0;

    /* Build JSON output. */
    json output = {
        {"depletion_width_nm", depletion},
        {"diagnostics", {
            {"D_h_nm2_per_s", sim->D_h},
            {"D_mol_nm2_per_s", sim->D_mol},
            {"accept_rate", accept_rate},
            {"depletion_bound_tcr_cd45_nn_p10_nm",
             dm.bound_tcr_cd45_nn_p10 < 0 ? nlohmann::json(nullptr)
                                           : nlohmann::json(dm.bound_tcr_cd45_nn_p10)},
            {"depletion_cd45_bound_tcr_nn_p10_nm",
             dm.cd45_bound_tcr_nn_p10 < 0 ? nlohmann::json(nullptr)
                                           : nlohmann::json(dm.cd45_bound_tcr_nn_p10)},
            {"depletion_cross_nn_median_nm", dm.cross_nn_median},
            {"depletion_frontier_nn_gap_nm", dm.frontier_nn_gap},
            {"depletion_ks_statistic", dm.ks_statistic},
            {"depletion_overlap_coeff", dm.overlap_coeff},
            {"depletion_percentile_gap_nm", dm.percentile_gap},
            {"dt_seconds", sim->dt},
            {"dt_auto_seconds", sim->dt_auto},
            {"dt_factor", sim->dt_factor},
            {"final_cd45_mean_r_nm", cd45_mean_r},
            {"final_tcr_mean_r_nm", tcr_mean_r},
            {"n_steps_actual", n_steps},
            {"step_size_h_nm", sim->step_size_h},
            {"step_size_mol_nm", sim->step_size_mol}
        }},
        {"inputs", {
            {"rigidity_kT", rigidity},
            {"time_sec", time_sec},
            {"patch_size_nm", sim->patch_size},
            {"sigma_bind_nm", sim->sigma_bind},
            {"sigma_r_nm", sim->sigma_r},
            {"u_assoc", sim->u_assoc}
        }}
    };

    /* Add binding time series if monitoring was active. */
    if (!binding_timeseries.empty()) {
        output["binding_timeseries"] = binding_timeseries;
        output["binding_monitor_interval"] = monitor_interval;
        output["binding_threshold_nm"] = monitor_binding_threshold;
    }

    auto json_str = output.dump(2);

    /* Write to file. */
    if (std::ofstream ofs(out_dir / "segregation.json"); ofs)
        ofs << json_str << "\n";

    /* Print to stdout. */
    std::cout << json_str << std::endl;

    sim_destroy(sim);
    return 0;
}
