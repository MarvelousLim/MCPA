#include "../lib/baxterwu_lib.h"
#include "../lib/checkpoint.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <time.h>
#include <unistd.h>   /* ftruncate (POSIX / Linux cluster) */

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

#define FREE_HOST_DEVICE(host_ptr, dev_ptr) do { \
    free(host_ptr); \
    CUDA_CHECK(cudaFree(dev_ptr)); \
} while(0)


/* ── Output-file helpers (same pattern as main_bc.cpp) ──────────────────── */

static bool ensure_directory(const char* path, const char* purpose) {
    std::error_code error;
    const std::filesystem::path directory(path);
    std::filesystem::create_directories(directory, error);
    if (!error) {
        const bool is_directory = std::filesystem::is_directory(directory, error);
        if (!error && is_directory) return true;
    }
    fprintf(stderr, "ERROR: Cannot create %s directory %s: %s\n",
            purpose, path, error ? error.message().c_str() : "path is not a directory");
    return false;
}

static void truncate_output_file(FILE* f, int64_t pos) {
    if (!f || pos < 0) return;
    fflush(f);
    if (ftruncate(fileno(f), (off_t)pos) != 0)
        perror("[resume] ftruncate");
}

static void repair_last_line(const char* path) {
    FILE* f = fopen(path, "r+b");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz == 0) { fclose(f); return; }
    fseek(f, sz - 1, SEEK_SET);
    char c;
    if (fread(&c, 1, 1, f) != 1) { fclose(f); return; }
    if (c == '\n') { fclose(f); return; }
    long pos = sz - 2;
    while (pos >= 0) {
        fseek(f, pos, SEEK_SET);
        if (fread(&c, 1, 1, f) != 1) { fclose(f); return; }
        if (c == '\n') break;
        pos--;
    }
    long trunc_at = (pos >= 0) ? pos + 1 : 0;
    if (ftruncate(fileno(f), trunc_at) != 0) {
        perror("[resume] ftruncate");
        fclose(f);
        return;
    }
    fclose(f);
    printf("[resume] Repaired broken last line in %s  (%ld→%ld bytes)\n",
           path, sz, trunc_at);
}

static FILE* open_output_file(const char* path, bool resuming) {
    if (resuming) {
        repair_last_line(path);
        FILE* f = fopen(path, "a");
        if (!f) fprintf(stderr, "ERROR: Cannot append to %s\n", path);
        else    printf("APPEND: %s\n", path);
        return f;
    }
    FILE* f = fopen(path, "w");
    if (!f) fprintf(stderr, "ERROR: Cannot open %s\n", path);
    else    printf("SUCCESS: Opened %s\n", path);
    return f;
}

static Files open_output_files(const Params& params, const char* outdir, bool resuming) {
    char path[256];
    const char* heating = params.heat ? "Heating" : "";
    Files files{};
    snprintf(path, sizeof(path),
             "%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_main.txt",
             outdir, heating, params.N, params.R, params.nSteps, params.seed);
    files.main_file = open_output_file(path, resuming);
    snprintf(path, sizeof(path),
             "%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_agg_stats.txt",
             outdir, heating, params.N, params.R, params.nSteps, params.seed);
    files.agg_stats_file = open_output_file(path, resuming);
    snprintf(path, sizeof(path),
             "%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_detailed_stats.txt",
             outdir, heating, params.N, params.R, params.nSteps, params.seed);
    files.detailed_stats_file = open_output_file(path, resuming);
    return files;
}

static bool output_files_ready(const Files& files) {
    return files.main_file && files.agg_stats_file && files.detailed_stats_file;
}

static void close_output_files(Files& files) {
    if (files.main_file) fclose(files.main_file);
    if (files.agg_stats_file) fclose(files.agg_stats_file);
    if (files.detailed_stats_file) fclose(files.detailed_stats_file);
    files = {};
}

static void flush_output_files(const Files& files) {
    fflush(files.main_file);
    fflush(files.agg_stats_file);
    fflush(files.detailed_stats_file);
}

static RunGpuMetadata collect_gpu_metadata_before_setup() {
    RunGpuMetadata metadata{};
    int device_index = 0;
    CUDA_CHECK(cudaGetDevice(&device_index));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
    snprintf(metadata.name, sizeof(metadata.name), "%s", properties.name);
    metadata.compute_capability = properties.major + properties.minor / 10.0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    metadata.total_memory_bytes = static_cast<uint64_t>(total_bytes);
    metadata.free_memory_before_setup_bytes = static_cast<uint64_t>(free_bytes);
    CUDA_CHECK(cudaDriverGetVersion(&metadata.cuda_driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&metadata.cuda_runtime_version));
    return metadata;
}

static void collect_gpu_metadata_after_setup(RunGpuMetadata& metadata) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;
    metadata.free_memory_after_setup_bytes = static_cast<uint64_t>(free_bytes);
}

static bool parse_detailed_limit(const char* text, int& value) {
    if (!text || !text[0]) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0'
        || parsed < -1 || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_checkpoint_hours(const char* text, double& value) {
    if (!text || !text[0]) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)
        || parsed < 0.0 || parsed > static_cast<double>(INT_MAX) / 3600.0)
        return false;
    value = parsed;
    return true;
}

static bool load_frozen_nsteps_schedule(
    const char* path, std::map<int, int>& schedule, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = std::string("cannot open schedule: ") + path;
        return false;
    }
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int ceiling = 0;
        int nsteps = 0;
        if (!(row >> ceiling >> nsteps)) {
            if (line_number == 1 && line.find("nSteps") != std::string::npos) continue;
            error = "invalid schedule row " + std::to_string(line_number);
            return false;
        }
        std::string trailing;
        if (row >> trailing || nsteps <= 0) {
            error = "invalid schedule row " + std::to_string(line_number);
            return false;
        }
        if (!schedule.emplace(ceiling, nsteps).second) {
            error = "duplicate ceiling " + std::to_string(ceiling);
            return false;
        }
    }
    if (schedule.empty()) {
        error = "schedule contains no data rows";
        return false;
    }
    return true;
}


/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr,
                "Usage: %s seed L blocks threads nSteps heat"
                " [checkpoint_dir|none] [checkpoint_hours] [detailed_limit]"
                " [frozen_nsteps_schedule|none]\n"
                "  detailed_limit: 0=header only, -1=all matching replicas,"
                " positive=deterministic hash sample (default 1000)\n"
                "  frozen_nsteps_schedule: two-column 'ceiling nSteps' table;"
                " checkpoints are disabled for scheduled experiments\n",
                argv[0]);
        return 2;
    }

    using SteadyClock = std::chrono::steady_clock;
    const auto global_start = SteadyClock::now();
    double equilibrate_total_seconds = 0.0;
    double prepare_resample_seconds = 0.0;
    double family_avg_seconds = 0.0;
    double replica_stats_seconds = 0.0;
    double update_replicas_seconds = 0.0;
    double copy_d2h_seconds = 0.0;
    double copy_h2d_seconds = 0.0;

    statisticsMode statistics_mode = detailed;
    equlibrateMode equlibrate_mode = normal;
    initializePopulationMode initialize_population_mode = random_pop;

    Params params;
    params.seed = atoi(argv[1]);
    params.L = atoi(argv[2]);
    params.N = params.L * params.L;
    params.blocks = atoi(argv[3]);
    params.threads = atoi(argv[4]);
    params.R = params.blocks * params.threads;
    params.nSteps = atoi(argv[5]);
    params.fullLatticeByteSize = (size_t)params.R * (size_t)params.N * sizeof(int);
    params.singleIntRowByteSize = (size_t)params.R * sizeof(int);
    params.replicaStatisticsByteSize = (size_t)params.R * sizeof(replicaStatistics);
    params.heat = (bool)atoi(argv[6]);

    int detailed_limit = 1000;
    if (argc >= 10 && !parse_detailed_limit(argv[9], detailed_limit)) {
        fprintf(stderr,
                "ERROR: detailed_limit must be -1, 0, or a positive integer; got '%s'\n",
                argv[9]);
        return 2;
    }

    std::map<int, int> frozen_nsteps_schedule;
    const bool use_frozen_nsteps_schedule =
        argc >= 11 && strcmp(argv[10], "none") != 0;
    if (use_frozen_nsteps_schedule) {
        std::string schedule_error;
        if (!load_frozen_nsteps_schedule(
                argv[10], frozen_nsteps_schedule, schedule_error)) {
            fprintf(stderr, "ERROR: %s\n", schedule_error.c_str());
            return 2;
        }
        const int sentinel = params.heat ? -2 * params.N - 2 : 2 * params.N + 2;
        if (frozen_nsteps_schedule.count(sentinel) == 0) {
            fprintf(stderr,
                    "ERROR: frozen nSteps schedule has no initial ceiling %d\n",
                    sentinel);
            return 2;
        }
        for (int ceiling = -2 * params.N; ceiling <= 2 * params.N; ceiling += 4) {
            if (frozen_nsteps_schedule.count(ceiling) == 0) {
                fprintf(stderr,
                        "ERROR: frozen nSteps schedule has no physical ceiling %d\n",
                        ceiling);
                return 2;
            }
        }
    }

    // ── 1. Checkpoint manager ─────────────────────────────────────────────────
    const char* chk_dir = (argc >= 8 && strcmp(argv[7], "none") != 0) ? argv[7] : "checkpoints";
    bool chk_enabled    = !(argc >= 8 && strcmp(argv[7], "none") == 0);
    if (use_frozen_nsteps_schedule && chk_enabled) {
        fprintf(stderr,
                "ERROR: frozen nSteps schedules are experimental and currently"
                " require checkpoint_dir=none\n");
        return 2;
    }
    double chk_hours = 0.25;
    if (argc >= 9 && !parse_checkpoint_hours(argv[8], chk_hours)) {
        fprintf(stderr,
                "ERROR: checkpoint_hours must be a finite nonnegative decimal; got '%s'\n",
                argv[8]);
        return 2;
    }
    int chk_interval = static_cast<int>(chk_hours * 3600.0);

    if (chk_enabled && !ensure_directory(chk_dir, "checkpoint")) return 2;

    char bw_chk_base[512];
    snprintf(bw_chk_base, sizeof(bw_chk_base),
             "2DBaxterWu%s_N%d_R%d_nSteps%d_run%d",
             params.heat ? "Heating" : "", params.N, params.R, params.nSteps, params.seed);

    CheckpointManager chk_mgr;
    checkpoint_init(chk_mgr,
                    params.L, params.N, params.R, params.nSteps, params.seed,
                    /*D=*/0.0f, (int)params.heat,
                    chk_dir, chk_enabled, chk_interval,
                    bw_chk_base);

    if (checkpoint_is_done(chk_mgr)) {
        printf("[chk] Run is already complete; leaving outputs unchanged.\n");
        return 0;
    }

    RunGpuMetadata run_gpu_metadata = collect_gpu_metadata_before_setup();
    initialize_resampling_rng(params.seed);

    // Check resume BEFORE opening files — "w" truncates immediately
    const bool will_resume = checkpoint_exists(chk_mgr);

    // ── 2. Output files ───────────────────────────────────────────────────────
    const char* outdir = "./datasets/2DBaxterWu";
    if (!ensure_directory(outdir, "output")) return 2;
    Files files = open_output_files(params, outdir, will_resume);
    if (!output_files_ready(files)) {
        close_output_files(files);
        return 2;
    }
    if (!will_resume) initialize_print(files);

    // ── 3. Allocate memory ────────────────────────────────────────────────────
    mainMemoryPointers host{}, device{};
    host.spin = (int*)malloc(params.fullLatticeByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.spin, params.fullLatticeByteSize));
    host.E = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.E, params.singleIntRowByteSize));
    host.replica_statistics = (replicaStatistics*)malloc(params.replicaStatisticsByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.replica_statistics, params.replicaStatisticsByteSize));
    host.O = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.O, params.singleIntRowByteSize));
    host.update = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.update, params.singleIntRowByteSize));
    host.replica_family = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.replica_family, params.singleIntRowByteSize));
    int* measured_replica_family = (int*)malloc(params.singleIntRowByteSize);

    const size_t fourier_phase_bytes = static_cast<size_t>(3) * params.N * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&device.fourier_phase_cos, fourier_phase_bytes));
    CUDA_CHECK(cudaMalloc((void**)&device.fourier_phase_sin, fourier_phase_bytes));
    initialize_fourier_phases(device, params);

    void* curand_states = setup_curand_states(params);
    const size_t rng_state_bytes = curand_states_byte_size(params);
    void* host_rng_state = chk_enabled ? malloc(rng_state_bytes) : nullptr;
    if (chk_enabled && host_rng_state == nullptr) {
        fprintf(stderr, "ERROR: Cannot allocate %zu bytes for checkpoint RNG state\n",
                rng_state_bytes);
        return 2;
    }
    collect_gpu_metadata_after_setup(run_gpu_metadata);

    // ── 4. Initialise / resume ────────────────────────────────────────────────
    initialize_update_arrays(host, params);

    int upper_energy = 2 * params.N + 2;
    int lower_energy = -2 * params.N - 2;
    int U_int        = params.heat ? lower_energy : upper_energy;
    int64_t chk_step_count = 0;
    int64_t out_pos[3] = {-1, -1, -1};
    uint64_t resumed_resampling_state = 0;
    uint64_t resumed_resampling_stream = 1;

    bool resumed = checkpoint_load(chk_mgr,
                                   params.L, params.N, params.R, params.nSteps,
                                   params.seed, 0.0f, (int)params.heat,
                                   host.spin, host.E, host.replica_family, host.O,
                                   U_int, chk_step_count,
                                   host_rng_state, rng_state_bytes,
                                   resumed_resampling_state,
                                   resumed_resampling_stream,
                                   out_pos);
    chk_mgr.step_count = chk_step_count;

    if (will_resume && !resumed) {
        // Candidate files existed, but current/previous/tmp all failed model,
        // CRC or semantic validation.  Preserve outputs for diagnosis.
        fprintf(stderr,
                "ERROR: checkpoint candidates exist but none validate; outputs were not changed\n");
        close_output_files(files);
        FREE_HOST_DEVICE(host.spin, device.spin);
        FREE_HOST_DEVICE(host.E, device.E);
        FREE_HOST_DEVICE(host.replica_statistics, device.replica_statistics);
        FREE_HOST_DEVICE(host.O, device.O);
        FREE_HOST_DEVICE(host.update, device.update);
        FREE_HOST_DEVICE(host.replica_family, device.replica_family);
        free(measured_replica_family);
        CUDA_CHECK(cudaFree(device.fourier_phase_cos));
        CUDA_CHECK(cudaFree(device.fourier_phase_sin));
        free(host_rng_state);
        CUDA_CHECK(cudaFree(curand_states));
        return 3;
    }

    if (resumed) {
        // Trim output files back to the checkpoint position — removes duplicates
        truncate_output_file(files.main_file,              out_pos[0]);
        truncate_output_file(files.agg_stats_file,         out_pos[1]);
        truncate_output_file(files.detailed_stats_file,    out_pos[2]);
        printf("[chk] Truncated output files to checkpoint positions\n");
        copyHostToDevice(device.spin, host.spin, params.fullLatticeByteSize);
        copyHostToDevice(device.E,    host.E,    params.singleIntRowByteSize);
        copyHostToDevice(curand_states, host_rng_state, rng_state_bytes);
        set_resampling_rng_state(
            ResamplingRngState{resumed_resampling_state,
                               resumed_resampling_stream});
        printf("[chk] Resuming from U=%d  (step %lld)\n", U_int, (long long)chk_step_count);
    } else {
        initialize_population(curand_states, device, params, initialize_population_mode);
        calc_device_energy(device, params);
    }

    int U = U_int;
    int no_replicas_try_again_counter = 0;
    int break_flg = 0;
    bool write_run_gpu_metadata = !resumed;

    // ── 5. Main MCPA loop ─────────────────────────────────────────────────────
    while ((U >= lower_energy && !params.heat) || (U <= upper_energy && params.heat)) {
        const int equilibrate_ceiling = U;
        Params equilibrate_params = params;
        if (use_frozen_nsteps_schedule) {
            const auto scheduled = frozen_nsteps_schedule.find(equilibrate_ceiling);
            if (scheduled == frozen_nsteps_schedule.end()) {
                fprintf(stderr,
                        "ERROR: frozen nSteps schedule has no entry for ceiling %d\n",
                        equilibrate_ceiling);
                return 4;
            }
            equilibrate_params.nSteps = scheduled->second;
        }
        printf("U:\t%f between %d and %d; nSteps: %d; policy: %s\n",
               1.0 * U, upper_energy, lower_energy,
               equilibrate_params.nSteps,
               use_frozen_nsteps_schedule ? "frozen_1_over_acceptance" : "fixed");

        const auto equilibrate_wall_start = SteadyClock::now();
        equilibrate(curand_states, device, equilibrate_params, U);
        const double equilibrate_seconds = std::chrono::duration<double>(
            SteadyClock::now() - equilibrate_wall_start).count();
        equilibrate_total_seconds += equilibrate_seconds;

        const auto copy_energy_start = SteadyClock::now();
        copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);
        copy_d2h_seconds += std::chrono::duration<double>(
            SteadyClock::now() - copy_energy_start).count();

        // The configurations reported at this energy still belong to these
        // families. prepare_resample_arrays relabels culled destination slots
        // with their future parents before the physical replica update.
        std::memcpy(measured_replica_family, host.replica_family,
                    params.singleIntRowByteSize);

        const auto prepare_resample_start = SteadyClock::now();
        int culled_replica_number = 0;
        double X = prepare_resample_arrays(host, params, &U, &culled_replica_number);
        prepare_resample_seconds += std::chrono::duration<double>(
            SteadyClock::now() - prepare_resample_start).count();

        if (X == 1) {
            if (no_replicas_try_again_counter < 10) {
                printf("try again number %d at U=%d\n", no_replicas_try_again_counter, U);
                no_replicas_try_again_counter++;
                continue;
            } else {
                printf("ended with no replicas\n");
                break_flg = 1;
            }
        }

        const auto family_avg_start = SteadyClock::now();
        const FamilyMetrics family_metrics = calc_family_metrics(
            host.replica_family, params.R);
        const double rho_t = family_metrics.simpson_concentration;
        printf("RhoT:\t%f\n", rho_t);
        family_avg_seconds += std::chrono::duration<double>(
            SteadyClock::now() - family_avg_start).count();

        const auto replica_stats_start = SteadyClock::now();
        calc_replica_statistics(device, params, U);
        replica_stats_seconds += std::chrono::duration<double>(
            SteadyClock::now() - replica_stats_start).count();

        const auto copy_stats_start = SteadyClock::now();
        copyDeviceToHost(host.replica_statistics, device.replica_statistics,
                         params.replicaStatisticsByteSize);
        copy_d2h_seconds += std::chrono::duration<double>(
            SteadyClock::now() - copy_stats_start).count();

        int64_t accepted_attempts = 0;
        for (int replica = 0; replica < params.R; ++replica)
            accepted_attempts += host.replica_statistics[replica].flip_count;
        const double attempted_updates = static_cast<double>(params.R)
                                       * static_cast<double>(params.N)
                                       * equilibrate_params.nSteps;
        const double population_acceptance_ratio =
            attempted_updates > 0.0 ? accepted_attempts / attempted_updates : NAN;

        print_main_data(files, U, X, rho_t, culled_replica_number,
                        equilibrate_seconds, family_metrics,
                        write_run_gpu_metadata ? &run_gpu_metadata : nullptr,
                        equilibrate_ceiling, equilibrate_params.nSteps,
                        population_acceptance_ratio,
                        use_frozen_nsteps_schedule
                            ? "frozen_1_over_acceptance" : "fixed");
        write_run_gpu_metadata = false;

        print_agg_stats(host, params, files, U, measured_replica_family);
        print_detailed_stats(host, params, files, U, detailed_limit,
                             measured_replica_family);

        chk_mgr.step_count++;
        const bool save_checkpoint = checkpoint_should_save(chk_mgr);

        if (break_flg) break;

        const auto copy_update_start = SteadyClock::now();
        copyHostToDevice(device.update, host.update, params.singleIntRowByteSize);
        copy_h2d_seconds += std::chrono::duration<double>(
            SteadyClock::now() - copy_update_start).count();

        const auto update_replicas_start = SteadyClock::now();
        update_replicas(device, params);
        update_replicas_seconds += std::chrono::duration<double>(
            SteadyClock::now() - update_replicas_start).count();

        if (save_checkpoint) {
            // Save only after physical replica copying.  At this boundary the
            // device configurations/energies and host family labels all refer
            // to the same offspring population that starts the next shell.
            flush_output_files(files);
            int64_t positions[3] = {
                (int64_t)ftell(files.main_file),
                (int64_t)ftell(files.agg_stats_file),
                (int64_t)ftell(files.detailed_stats_file),
            };
            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);
            copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);
            copyDeviceToHost(host_rng_state, curand_states, rng_state_bytes);
            const ResamplingRngState resampling_rng = get_resampling_rng_state();
            checkpoint_save(chk_mgr,
                            params.L, params.N, params.R, params.nSteps,
                            params.seed, 0.0f, (int)params.heat,
                            host.spin, host.E, host.replica_family, host.O,
                            U, host_rng_state, rng_state_bytes,
                            resampling_rng.state, resampling_rng.stream,
                            positions);
        }
    }

    // ── 6. Cleanup ────────────────────────────────────────────────────────────
    flush_output_files(files);
    close_output_files(files);
    checkpoint_mark_done(chk_mgr);

    FREE_HOST_DEVICE(host.spin, device.spin);
    FREE_HOST_DEVICE(host.E, device.E);
    FREE_HOST_DEVICE(host.replica_statistics, device.replica_statistics);
    FREE_HOST_DEVICE(host.O, device.O);
    FREE_HOST_DEVICE(host.update, device.update);
    FREE_HOST_DEVICE(host.replica_family, device.replica_family);
    free(measured_replica_family);
    free(host_rng_state);
    CUDA_CHECK(cudaFree(device.fourier_phase_cos));
    CUDA_CHECK(cudaFree(device.fourier_phase_sin));
    CUDA_CHECK(cudaFree(curand_states));

    const double total = std::chrono::duration<double>(
        SteadyClock::now() - global_start).count();
    const double measured = equilibrate_total_seconds + prepare_resample_seconds
                          + family_avg_seconds + replica_stats_seconds
                          + update_replicas_seconds + copy_d2h_seconds
                          + copy_h2d_seconds;
    const double other_seconds = total > measured ? total - measured : 0.0;
    const auto percent = [total](double seconds) {
        return total > 0.0 ? 100.0 * seconds / total : 0.0;
    };
    printf("\n=== TIMING SUMMARY ===\n");
    printf("Total time:        %.2fs\n", total);
    printf("Equilibrate:       %.2fs (%.1f%%)\n", equilibrate_total_seconds,
           percent(equilibrate_total_seconds));
    printf("Prepare resample:  %.2fs (%.1f%%)\n", prepare_resample_seconds,
           percent(prepare_resample_seconds));
    printf("Family avg:        %.2fs (%.1f%%)\n", family_avg_seconds,
           percent(family_avg_seconds));
    printf("Replica stats:     %.2fs (%.1f%%)\n", replica_stats_seconds,
           percent(replica_stats_seconds));
    printf("Update replicas:   %.2fs (%.1f%%)\n", update_replicas_seconds,
           percent(update_replicas_seconds));
    printf("Copy D→H:          %.2fs (%.1f%%)\n", copy_d2h_seconds,
           percent(copy_d2h_seconds));
    printf("Copy H→D:          %.2fs (%.1f%%)\n", copy_h2d_seconds,
           percent(copy_h2d_seconds));
    printf("Other/output:      %.2fs (%.1f%%)\n", other_seconds,
           percent(other_seconds));
    printf("====================\n");
    return 0;
}
