/*
 * Usage:
 *   main_bc seed L blocks threads nSteps heat D [chk_dir] [chk_interval_h] [detail_cap]
 *
 *   seed           : integer, seeds RNG and labels output files
 *   L              : linear lattice size (N = L*L spins)
 *   blocks         : CUDA grid width
 *   threads        : CUDA block width (population R = blocks * threads)
 *   nSteps         : MC sweeps per energy level
 *   heat           : 0 = cooling (ceiling), 1 = heating (floor)
 *   D              : crystal-field coupling Δ  (decimal string, e.g. 1.96)
 *   chk_dir        : (optional) checkpoint directory, or "none"  (default: ./checkpoints)
 *   chk_interval_h : (optional) decimal hours between checkpoints; default 0.25
 *   detail_cap     : (optional) 100 default, 0 none, -1 every matching replica
 */

#include "../lib/blumeCapel_lib.h"
#include "../lib/checkpoint.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <filesystem>
#include <system_error>
#include <cerrno>
#include <sys/stat.h>
#include <chrono>
#include <cmath>
#include <vector>

#define FREE_HOST_DEVICE(host_ptr, dev_ptr) do { \
    free(host_ptr); \
    CUDA_CHECK(cudaFree(dev_ptr)); \
} while(0)


/* ── Output-file helpers ──────────────────────────────────────────────────── */

/* Truncate an already-open file to `pos` bytes to strip output written after
 * the last checkpoint.  Works even in append mode (POSIX). */
static bool truncate_output_file(FILE* f, int64_t pos) {
    if (!f || pos < 0 || fflush(f) != 0) return false;
    struct stat status{};
    if (fstat(fileno(f), &status) != 0 || status.st_size < pos) return false;
    if (ftruncate(fileno(f), (off_t)pos) != 0) return false;
    return fseek(f, 0, SEEK_END) == 0;
}

/* If the file exists and its last byte is not '\n', walk back to the previous
 * '\n' and truncate there.  Removes a partially-written line from a crash. */
static void repair_last_line(const char* path) {
    FILE* f = fopen(path, "r+b");
    if (!f) return;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return; }

    if (fseek(f, sz - 1, SEEK_SET) != 0) { fclose(f); return; }
    char c;
    if (fread(&c, 1, 1, f) != 1) { fclose(f); return; }
    if (c == '\n') { fclose(f); return; }

    long pos = sz - 2;
    while (pos >= 0) {
        if (fseek(f, pos, SEEK_SET) != 0 || fread(&c, 1, 1, f) != 1) {
            fclose(f);
            return;
        }
        if (c == '\n') break;
        pos--;
    }
    long trunc_at = (pos >= 0) ? pos + 1 : 0;
    if (ftruncate(fileno(f), trunc_at) != 0) {
        perror("[resume] ftruncate broken line");
        fclose(f);
        return;
    }
    fclose(f);
    printf("[resume] Repaired broken last line in %s  (truncated %ld→%ld bytes)\n",
           path, sz, trunc_at);
}

static FILE* open_output_file(const char* path, bool resuming) {
    if (resuming) {
        if (access(path, F_OK) != 0) {
            fprintf(stderr, "ERROR: Resume output is missing: %s\n", path);
            return nullptr;
        }
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

static Files open_output_files(struct Params params, const char* outdir, int resuming) {
    char path[256];
    char d_str[64];
    const char* heating = params.heat ? "Heating" : "";
    format_D_for_path(params.D_num, params.D_denum, d_str, sizeof(d_str));
    Files files;

    snprintf(path, sizeof(path),
             "%s/2DBlume%s_q3_D%s_N%d_R%d_nSteps%d_run%d_main.txt",
             outdir, heating, d_str, params.N, params.R, params.nSteps, params.seed);
    files.main_file = open_output_file(path, resuming);

    snprintf(path, sizeof(path),
             "%s/2DBlume%s_q3_D%s_N%d_R%d_nSteps%d_run%d_agg_stats.txt",
             outdir, heating, d_str, params.N, params.R, params.nSteps, params.seed);
    files.agg_stats_file = open_output_file(path, resuming);

    snprintf(path, sizeof(path),
             "%s/2DBlume%s_q3_D%s_N%d_R%d_nSteps%d_run%d_detailed_stats.txt",
             outdir, heating, d_str, params.N, params.R, params.nSteps, params.seed);
    files.detailed_stats_file = open_output_file(path, resuming);

    return files;
}

static bool ensure_output_directory(const char* outdir) {
    std::error_code error;
    std::filesystem::create_directories(outdir, error);
    if (error) {
        fprintf(stderr, "ERROR: Cannot create output directory %s: %s\n",
                outdir, error.message().c_str());
        return false;
    }
    return true;
}

static void flush_output_files(struct Files files) {
    fflush(files.main_file);
    fflush(files.agg_stats_file);
    fflush(files.detailed_stats_file);
}

static bool parse_int_argument(const char* text, const char* name,
                               long minimum, long maximum, int* value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
        || parsed < minimum || parsed > maximum) {
        fprintf(stderr, "ERROR: %s must be an integer in [%ld, %ld], got '%s'\n",
                name, minimum, maximum, text);
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

static bool parse_checkpoint_hours(const char* text, double* value) {
    errno = 0;
    char* end = nullptr;
    const double parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(parsed)
        || parsed < 0.0 || parsed > static_cast<double>(INT_MAX) / 3600.0) {
        fprintf(stderr,
                "ERROR: chk_hours must be a finite nonnegative decimal, got '%s'\n",
                text);
        return false;
    }
    *value = parsed;
    return true;
}


/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    if (argc < 8 || argc > 11) {
        fprintf(stderr,
                "Usage: %s seed L blocks threads nSteps heat D [chk_dir] [chk_hours] [detail_cap]\n"
                "  chk_dir   : checkpoint directory, or 'none' to disable  (default: ./checkpoints)\n"
                "  chk_hours : decimal hours between checkpoints; 0 every shell (default: 0.25)\n"
                "  detail_cap: 100 default, 0 none, -1 all, positive means deterministic prefix\n",
                argv[0]);
        return 1;
    }

    clock_t global_start = clock();

    struct Params params{};
    int heat_value = 0;
    if (!parse_int_argument(argv[1], "seed", 0, INT_MAX, &params.seed)
        || !parse_int_argument(argv[2], "L", 3, 46340, &params.L)
        || !parse_int_argument(argv[3], "blocks", 1, INT_MAX, &params.blocks)
        || !parse_int_argument(argv[4], "threads", 1, 1024, &params.threads)
        || !parse_int_argument(argv[5], "nSteps", 1, INT_MAX, &params.nSteps)
        || !parse_int_argument(argv[6], "heat", 0, 1, &heat_value)) {
        return 1;
    }
    params.heat = heat_value != 0;

    const long long site_count = static_cast<long long>(params.L) * params.L;
    const long long replica_count = static_cast<long long>(params.blocks) * params.threads;
    if (site_count > INT_MAX || replica_count > INT_MAX) {
        fprintf(stderr, "ERROR: L^2 and blocks*threads must fit signed 32-bit indexing\n");
        return 1;
    }
    params.N = static_cast<int>(site_count);
    params.R = static_cast<int>(replica_count);

    if (parse_D_from_string(argv[7], &params.D_num, &params.D_denum) != 0) {
        fprintf(stderr, "ERROR: invalid D value '%s'\n", argv[7]);
        return 1;
    }
    if (check_bc_energy_overflow(&params) != 0) return 1;
    compute_U_stop(&params);

    params.fullLatticeByteSize       = (size_t)params.R * (size_t)params.N * sizeof(int);
    params.singleIntRowByteSize      = (size_t)params.R * sizeof(int);
    params.replicaStatisticsByteSize = (size_t)params.R * sizeof(replicaStatistics);

    initialize_resampling_rng(params.seed);

    /* Checkpoint manager — init before opening output files */
    const char* chk_dir = (argc >= 9 && strcmp(argv[8], "none") != 0) ? argv[8] : "checkpoints";
    bool chk_enabled     = !(argc >= 9 && strcmp(argv[8], "none") == 0);
    double checkpoint_hours = 0.25;
    if (argc >= 10 && !parse_checkpoint_hours(argv[9], &checkpoint_hours)) {
        return 1;
    }
    int detail_cap = 100;
    if (argc >= 11
        && !parse_int_argument(argv[10], "detail_cap", -1, INT_MAX,
                               &detail_cap)) {
        return 1;
    }
    int chk_interval = static_cast<int>(checkpoint_hours * 3600.0);

    CheckpointManager chk_mgr;
    if (!checkpoint_init_bc(chk_mgr,
                            params.L, params.N, params.R, params.nSteps,
                            params.seed, params.D_num, params.D_denum, params.heat,
                            detail_cap, chk_dir, chk_enabled, chk_interval)) return 1;
    if (checkpoint_is_done(chk_mgr)) {
        printf("[chk] Run is already complete; leaving outputs unchanged.\n");
        return 0;
    }

    GpuMetadata gpu_metadata{};
    int gpu_device = 0;
    cudaDeviceProp gpu_properties{};
    size_t free_memory_before_setup = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaGetDevice(&gpu_device));
    CUDA_CHECK(cudaGetDeviceProperties(&gpu_properties, gpu_device));
    CUDA_CHECK(cudaRuntimeGetVersion(&gpu_metadata.cuda_runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&gpu_metadata.cuda_driver_version));
    CUDA_CHECK(cudaMemGetInfo(&free_memory_before_setup, &total_memory));
    gpu_metadata.compute_major = gpu_properties.major;
    gpu_metadata.compute_minor = gpu_properties.minor;
    gpu_metadata.total_memory_bytes = static_cast<uint64_t>(total_memory);
    gpu_metadata.free_memory_before_setup_bytes =
        static_cast<uint64_t>(free_memory_before_setup);
    snprintf(gpu_metadata.name, sizeof(gpu_metadata.name), "%s", gpu_properties.name);
    for (char& character : gpu_metadata.name)
        if (character == '\t' || character == '\n' || character == '\r') character = ' ';

    struct mainMemoryPointers host, device;
    host.spin               = (int*)malloc(params.fullLatticeByteSize);
    host.e_j                = (int*)malloc(params.singleIntRowByteSize);
    host.e_delta            = (int*)malloc(params.singleIntRowByteSize);
    host.replica_statistics = (struct replicaStatistics*)malloc(params.replicaStatisticsByteSize);
    host.O                  = (int*)malloc(params.singleIntRowByteSize);
    host.update             = (int*)malloc(params.singleIntRowByteSize);
    host.replica_family     = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.spin,               params.fullLatticeByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.e_j,                params.singleIntRowByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.e_delta,            params.singleIntRowByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.replica_statistics, params.replicaStatisticsByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.O,                  params.singleIntRowByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.update,             params.singleIntRowByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.replica_family,     params.singleIntRowByteSize));

    void* curand_states = setup_curand_states(params);
    size_t free_memory_after_setup = 0;
    size_t total_memory_after_setup = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory_after_setup, &total_memory_after_setup));
    gpu_metadata.free_memory_after_setup_bytes =
        static_cast<uint64_t>(free_memory_after_setup);
    const size_t rng_state_bytes = curand_states_byte_size(params);
    void* host_rng_state = chk_enabled ? malloc(rng_state_bytes) : nullptr;
    if (chk_enabled && !host_rng_state) {
        fprintf(stderr, "ERROR: Cannot allocate checkpoint RNG buffer\n");
        return 1;
    }
    initialize_update_arrays(host, params);

    int U = params.heat ? INT32_MIN : INT32_MAX;
    int64_t chk_step_count = 0;

    int64_t out_pos[3] = {-1, -1, -1};
    uint64_t resumed_pcg_state = 0;
    uint64_t resumed_pcg_stream = 1;
    const bool checkpoint_candidates_present = checkpoint_exists(chk_mgr);
    bool resumed = checkpoint_load_bc(chk_mgr,
                                     params.L, params.N, params.R, params.nSteps,
                                     params.seed, params.D_num, params.D_denum,
                                     params.heat, detail_cap,
                                     host.spin, host.e_j, host.e_delta,
                                     host.replica_family, host.O,
                                     U, chk_step_count,
                                     host_rng_state, rng_state_bytes,
                                     resumed_pcg_state, resumed_pcg_stream,
                                     out_pos);
    if (checkpoint_candidates_present && !resumed) {
        fprintf(stderr,
                "ERROR: checkpoint candidates exist but none validate; outputs were not opened\n");
        FREE_HOST_DEVICE(host.spin,               device.spin);
        FREE_HOST_DEVICE(host.e_j,                device.e_j);
        FREE_HOST_DEVICE(host.e_delta,            device.e_delta);
        FREE_HOST_DEVICE(host.replica_statistics, device.replica_statistics);
        FREE_HOST_DEVICE(host.O,                  device.O);
        FREE_HOST_DEVICE(host.update,             device.update);
        FREE_HOST_DEVICE(host.replica_family,     device.replica_family);
        free(host_rng_state);
        CUDA_CHECK(cudaFree(curand_states));
        return 1;
    }
    chk_mgr.step_count = chk_step_count;

    /* Validation is complete before append/truncate versus fresh output is chosen. */
    const char* outdir = "./datasets/2DBlumeEnergyParts";
    if (!ensure_output_directory(outdir)) return 1;
    struct Files files = open_output_files(params, outdir, resumed);
    if (!files.main_file || !files.agg_stats_file || !files.detailed_stats_file) {
        if (files.main_file) fclose(files.main_file);
        if (files.agg_stats_file) fclose(files.agg_stats_file);
        if (files.detailed_stats_file) fclose(files.detailed_stats_file);
        return 1;
    }
    if (!resumed) initialize_print(files);

    if (resumed) {
        if (!truncate_output_file(files.main_file, out_pos[0])
            || !truncate_output_file(files.agg_stats_file, out_pos[1])
            || !truncate_output_file(files.detailed_stats_file, out_pos[2])) {
            fprintf(stderr, "ERROR: Resume output is shorter than checkpoint offset\n");
            fclose(files.main_file);
            fclose(files.agg_stats_file);
            fclose(files.detailed_stats_file);
            return 1;
        }
        printf("[chk] Truncated output files to checkpoint positions\n");
        copyHostToDevice(device.spin,    host.spin,    params.fullLatticeByteSize);
        copyHostToDevice(device.e_j,     host.e_j,     params.singleIntRowByteSize);
        copyHostToDevice(device.e_delta, host.e_delta, params.singleIntRowByteSize);
        copyHostToDevice(curand_states, host_rng_state, rng_state_bytes);
        set_resampling_rng_state(
            ResamplingRngState{resumed_pcg_state, resumed_pcg_stream});
        printf("[chk] Resuming from E=%.6f  (step %lld)\n",
               energy_physical_at_U(host, params, U), (long long)chk_step_count);
    } else {
        initialize_population(curand_states, device, params);
        calc_device_energy(device, params);
        /* No host copy here — first equilibrate runs on GPU state only */
    }

    bool metadata_pending = !resumed;
    std::vector<int> measured_family(static_cast<size_t>(params.R));

    /* ── Main MCPA loop ───────────────────────────────────────────────────── */
    int retry_counter = 0;

    while ((U >= params.U_stop_cool && !params.heat) ||
           (U <= params.U_stop_heat &&  params.heat)) {

        if (U == INT32_MAX || U == INT32_MIN)
            printf("U:\t(initial %s)\n", params.heat ? "floor" : "ceiling");
        else
            printf("U:\t%.6f\n", energy_physical(0, 1, U, params.D_denum));
        fflush(stdout);

        const auto equilibrate_start = std::chrono::steady_clock::now();
        equilibrate(curand_states, device, params, U);
        const double equilibrate_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - equilibrate_start).count();

        copyDeviceToHost(host.e_j,     device.e_j,     params.singleIntRowByteSize);
        copyDeviceToHost(host.e_delta, device.e_delta, params.singleIntRowByteSize);

        memcpy(measured_family.data(), host.replica_family,
               params.singleIntRowByteSize);
        const double pre_family_concentration = calc_family_concentration(
            measured_family.data(), params.R);

        int n_culled_exact = 0;
        double X = prepare_resample_arrays(host, params, &U, &n_culled_exact);

        if (X >= 1.0) {
            if (retry_counter < 10) {
                retry_counter++;
                continue;
            }
            const double rho = calc_family_avg_sq_size(host, params);
            print_main_data(files, energy_physical_at_U(host, params, U), X, rho,
                            U, params.D_num, params.D_denum, n_culled_exact,
                            equilibrate_seconds, pre_family_concentration, rho,
                            metadata_pending ? &gpu_metadata : nullptr);
            metadata_pending = false;
            break;
        }
        retry_counter = 0;

        double rho = calc_family_avg_sq_size(host, params);
        print_main_data(files, energy_physical_at_U(host, params, U), X, rho,
                        U, params.D_num, params.D_denum, n_culled_exact,
                        equilibrate_seconds, pre_family_concentration, rho,
                        metadata_pending ? &gpu_metadata : nullptr);
        metadata_pending = false;

        calc_replica_statistics(device, params, U);
        copyDeviceToHost(host.replica_statistics, device.replica_statistics,
                         params.replicaStatisticsByteSize);

        print_agg_stats(host, params, files, U, measured_family.data());
        print_detailed_stats(host, params, files, U, measured_family.data(),
                             detail_cap);

        chk_mgr.step_count++;
        const bool save_checkpoint = checkpoint_should_save(chk_mgr);

        copyHostToDevice(device.update, host.update, params.singleIntRowByteSize);
        update_replicas(device, params);
        copyDeviceToHost(host.e_j,     device.e_j,     params.singleIntRowByteSize);
        copyDeviceToHost(host.e_delta, device.e_delta, params.singleIntRowByteSize);

        if (save_checkpoint) {
            /* Clean boundary: device population and host genealogy are offspring-aligned. */
            flush_output_files(files);
            int64_t save_pos[3] = {
                (int64_t)ftell(files.main_file),
                (int64_t)ftell(files.agg_stats_file),
                (int64_t)ftell(files.detailed_stats_file),
            };
            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);
            copyDeviceToHost(host_rng_state, curand_states, rng_state_bytes);
            const ResamplingRngState pcg = get_resampling_rng_state();
            if (!checkpoint_save_bc(
                    chk_mgr, params.L, params.N, params.R, params.nSteps,
                    params.seed, params.D_num, params.D_denum, params.heat,
                    detail_cap,
                    host.spin, host.e_j, host.e_delta,
                    host.replica_family, host.O, U,
                    host_rng_state, rng_state_bytes, pcg.state, pcg.stream,
                    save_pos)) {
                fprintf(stderr, "ERROR: Checkpoint save failed\n");
                return 1;
            }
        }
    }

    const bool checkpoint_completed = checkpoint_mark_done(chk_mgr);

    fclose(files.main_file);
    fclose(files.agg_stats_file);
    fclose(files.detailed_stats_file);

    FREE_HOST_DEVICE(host.spin,               device.spin);
    FREE_HOST_DEVICE(host.e_j,                device.e_j);
    FREE_HOST_DEVICE(host.e_delta,            device.e_delta);
    FREE_HOST_DEVICE(host.replica_statistics, device.replica_statistics);
    FREE_HOST_DEVICE(host.O,                  device.O);
    FREE_HOST_DEVICE(host.update,             device.update);
    FREE_HOST_DEVICE(host.replica_family,     device.replica_family);
    free(host_rng_state);
    CUDA_CHECK(cudaFree(curand_states));

    char d_str[64];
    format_D_for_path(params.D_num, params.D_denum, d_str, sizeof(d_str));
    clock_t global_end = clock();
    printf("Total time: %.2fs\n", (double)(global_end - global_start) / CLOCKS_PER_SEC);
    printf("D=%s  L=%d  N=%d  R=%d  heat=%d  steps=%lld\n",
           d_str, params.L, params.N, params.R, params.heat,
           (long long)chk_mgr.step_count);
    if (!checkpoint_completed) {
        fprintf(stderr, "ERROR: could not mark the completed checkpoint\n");
        return 1;
    }
    return 0;
}
