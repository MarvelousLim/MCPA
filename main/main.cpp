#include "../lib/potts_lib.h"
#include "../lib/potts_checkpoint.h"
#include "../lib/potts_output.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <array>
#include <chrono>
#include <cctype>
#include <errno.h>
#include <filesystem>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <system_error>
#include <unistd.h>

#define CUDA_CHECK(ans) { gpu_assert((int)(ans), __FILE__, __LINE__); }

static FILE* open_output(const char* path, bool resume = false,
                         int64_t offset = 0) {
    FILE* file = fopen(path, resume ? "r+b" : "w");
    if (!file) fprintf(stderr, "Could not open output file: %s\n", path);
    if (file && resume) {
        if (offset < 0 || ftruncate(fileno(file), static_cast<off_t>(offset)) != 0
                || fseek(file, static_cast<long>(offset), SEEK_SET) != 0) {
            fprintf(stderr, "Could not restore output position: %s\n", path);
            fclose(file);
            return nullptr;
        }
    }
    return file;
}

static bool capture_output_offsets(
        const std::array<FILE*, kPottsLiveOutputCount>& files,
        std::array<int64_t, kPottsLiveOutputCount>* offsets) {
    for (size_t i = 0; i < files.size(); ++i) {
        if (fflush(files[i]) != 0) return false;
        const long position = ftell(files[i]);
        if (position < 0) return false;
        (*offsets)[i] = static_cast<int64_t>(position);
    }
    return true;
}

static bool ensure_output_directory(const char* path) {
    std::error_code error;
    const std::filesystem::path directory(path);
    std::filesystem::create_directories(directory, error);
    if (!error && std::filesystem::is_directory(directory, error) && !error)
        return true;
    fprintf(stderr, "Could not create output directory %s: %s\n", path,
            error ? error.message().c_str() : "path is not a directory");
    return false;
}

static bool parse_int_arg(const char* text, const char* name, int* value) {
    if (!text || !name || !value) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || parsed < INT_MIN || parsed > INT_MAX) {
        fprintf(stderr, "Invalid integer for %s: %s\n", name, text);
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

static std::string rectangular_token(std::string value) {
    for (char& character : value) {
        if (std::isspace(static_cast<unsigned char>(character))) character = '_';
    }
    return value.empty() ? "unknown" : value;
}

int main(int argc, char* argv[]) {
    if (argc < 8 || argc > 10) {
        fprintf(stderr,
                "Usage: %s seed L blocks threads nSteps q heat "
                "[checkpoint_dir|none [checkpoint_shell_interval]]\n"
                "  heat: 0=cooling, 1=heating\n",
                argv[0]);
        return 1;
    }

    struct PottsParams params = {};
    int heat_value = 0;
    if (!parse_int_arg(argv[1], "seed", &params.seed)
            || !parse_int_arg(argv[2], "L", &params.L)
            || !parse_int_arg(argv[3], "blocks", &params.blocks)
            || !parse_int_arg(argv[4], "threads", &params.threads)
            || !parse_int_arg(argv[5], "nSteps", &params.nSteps)
            || !parse_int_arg(argv[6], "q", &params.q)
            || !parse_int_arg(argv[7], "heat", &heat_value)) {
        return 2;
    }

    if (params.L < 2 || params.blocks <= 0 || params.threads <= 0
            || params.nSteps <= 0 || !potts_supported_q(params.q)
            || (heat_value != 0 && heat_value != 1)) {
        fprintf(stderr,
                "L must be at least 2; blocks, threads and nSteps must be positive; "
                "q must be one of 2, 3, 4; heat must be 0 or 1\n");
        return 2;
    }

    const long long N_wide = static_cast<long long>(params.L) * params.L;
    const long long R_wide = static_cast<long long>(params.blocks) * params.threads;
    if (N_wide > (INT_MAX - 1LL) / 2LL || R_wide > INT_MAX
            || static_cast<unsigned long long>(N_wide)
                   > static_cast<unsigned long long>(SIZE_MAX) /
                     static_cast<unsigned long long>(R_wide)) {
        fprintf(stderr, "L*L, blocks*threads, or R*N exceeds the supported integer range\n");
        return 2;
    }
    params.N = static_cast<int>(N_wide);
    params.R = static_cast<int>(R_wide);
    params.heat = heat_value != 0;
    params.fullLatticeByteSize =
        (size_t)params.R * (size_t)params.N * sizeof(char);
    params.singleIntRowByteSize = (size_t)params.R * sizeof(int);

    int detailed_cap = 100;
    if (const char* cap_text = getenv("MCPA_DETAILED_CAP")) {
        if (!parse_int_arg(cap_text, "MCPA_DETAILED_CAP", &detailed_cap)
                || detailed_cap < -1) {
            fprintf(stderr, "MCPA_DETAILED_CAP must be -1 or nonnegative\n");
            return 2;
        }
    }

    const bool checkpoint_enabled = argc >= 9 && strcmp(argv[8], "none") != 0;
    int checkpoint_shell_interval = 1;
    if (argc == 10
            && (!parse_int_arg(argv[9], "checkpoint_shell_interval",
                               &checkpoint_shell_interval)
                || checkpoint_shell_interval <= 0)) {
        fprintf(stderr, "checkpoint_shell_interval must be positive\n");
        return 2;
    }

    const char* heating = params.heat ? "Heating" : "";
    printf("Start\n");
    printf("n_s (number of flips): %lld\n",
           (long long)params.N * params.nSteps);
    printf("running 2DPotts%s_q%d_N%d_R%d_nSteps%d_run%de.txt\n",
           heating, params.q, params.N, params.R, params.nSteps, params.seed);
    fflush(stdout);

    const char* prefix = getenv("MCPA_OUTPUT_ROOT");
    if (!prefix || !prefix[0]) prefix = "./datasets";
    char output_directory[512];
    if (snprintf(output_directory, sizeof(output_directory),
                 "%s/2DPotts", prefix) >= static_cast<int>(sizeof(output_directory))) {
        fprintf(stderr, "Output root is too long\n");
        return 2;
    }
    if (!ensure_output_directory(output_directory)) {
        return 2;
    }
    if (checkpoint_enabled && !ensure_output_directory(argv[8])) return 2;

    char run_name_buffer[256];
    if (snprintf(run_name_buffer, sizeof(run_name_buffer),
                 "2DPotts_v%d%s_q%d_N%d_R%d_nSteps%d_run%d",
                 kPottsOutputVersion,
                 heating, params.q, params.N, params.R, params.nSteps, params.seed)
            >= static_cast<int>(sizeof(run_name_buffer))) {
        fprintf(stderr, "Run identity is too long\n");
        return 2;
    }
    const std::string run_name = run_name_buffer;
    const std::string checkpoint_base = checkpoint_enabled
        ? (std::filesystem::path(argv[8]) / run_name).string() : std::string{};
    const PottsCheckpointIdentity checkpoint_identity{
        params.L, params.N, params.R, params.nSteps, params.seed, params.q,
        params.heat, detailed_cap,
        static_cast<size_t>(params.R) * sizeof(curandStatePhilox4_32_10_t)};
    PottsCheckpointState checkpoint_state;
    std::string checkpoint_source;
    std::string checkpoint_error;
    PottsCheckpointLoadStatus checkpoint_status = PottsCheckpointLoadStatus::not_found;
    if (checkpoint_enabled) {
        checkpoint_status = load_potts_checkpoint(
            checkpoint_base, checkpoint_identity, &checkpoint_state,
            &checkpoint_source, &checkpoint_error);
        if (checkpoint_status == PottsCheckpointLoadStatus::invalid) {
            fprintf(stderr, "No valid Potts checkpoint generation: %s\n",
                    checkpoint_error.c_str());
            return 3;
        }
        if (checkpoint_status == PottsCheckpointLoadStatus::done) {
            printf("[chk] Run is already complete; leaving outputs unchanged.\n");
            return 0;
        }
    }
    const bool resumed = checkpoint_status == PottsCheckpointLoadStatus::loaded;

    struct MemEstimate mem = estimate_potts_setup_memory(params.N, params.R);
    PottsGpuMetadata gpu_metadata;
    cudaDeviceProp device_properties{};
    size_t free_before_setup = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));
    CUDA_CHECK(cudaMemGetInfo(&free_before_setup, &total_memory));
    gpu_metadata.name = rectangular_token(device_properties.name);
    gpu_metadata.compute_major = device_properties.major;
    gpu_metadata.compute_minor = device_properties.minor;
    gpu_metadata.total_memory_bytes = total_memory;
    gpu_metadata.free_memory_before_setup_bytes = free_before_setup;
    CUDA_CHECK(cudaDriverGetVersion(&gpu_metadata.driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&gpu_metadata.runtime_version));
    if (report_setup_memory("2DPotts", params.L, params.N, params.R,
                            sizeof(char), &mem)) {
        fprintf(stderr, "Aborting: not enough free GPU memory for this (L,R)\n");
        fflush(stderr);
        return 1;
    }

    fflush(stdout);
    fflush(stderr);

    const std::array<const char*, kPottsLiveOutputCount> output_suffixes{{
        "_main.txt", "_agg_stats.txt", "_detailed_stats.txt"}};
    std::array<std::string, kPottsLiveOutputCount> output_paths;
    std::array<FILE*, kPottsLiveOutputCount> output_files{};
    for (size_t i = 0; i < output_paths.size(); ++i) {
        output_paths[i] = (std::filesystem::path(output_directory)
                           / (run_name + output_suffixes[i])).string();
    }
    if (resumed) {
        for (size_t i = 0; i < output_paths.size(); ++i) {
            std::error_code error;
            const uintmax_t size = std::filesystem::file_size(output_paths[i], error);
            if (error || checkpoint_state.output_offsets[i] < 0
                    || size < static_cast<uintmax_t>(
                        checkpoint_state.output_offsets[i])) {
                fprintf(stderr, "Checkpoint output is missing or shorter than its saved offset: %s\n",
                        output_paths[i].c_str());
                return 3;
            }
        }
    }
    for (size_t i = 0; i < output_paths.size(); ++i) {
        output_files[i] = open_output(
            output_paths[i].c_str(), resumed,
            resumed ? checkpoint_state.output_offsets[i] : 0);
    }
    if (!output_files[0] || !output_files[1] || !output_files[2]) {
        for (FILE* file : output_files) if (file) fclose(file);
        return 1;
    }
    FILE* main_file = output_files[0];
    FILE* aggregate_file = output_files[1];
    FILE* detailed_file = output_files[2];
    if (!resumed && !write_potts_output_headers(
            main_file, aggregate_file, detailed_file)) {
        fprintf(stderr, "Could not write Potts output headers\n");
        for (FILE* file : output_files) fclose(file);
        return 1;
    }

    struct PottsMemoryPointers host = {};
    struct PottsMemoryPointers device = {};
    host.spin = (char*)malloc(params.fullLatticeByteSize);
    host.E = (int*)malloc(params.singleIntRowByteSize);
    host.update = (int*)malloc(params.singleIntRowByteSize);
    host.accepted_flips = (uint64_t*)malloc(
        static_cast<size_t>(params.R) * sizeof(uint64_t));
    int* replica_family = (int*)malloc(params.singleIntRowByteSize);
    int* energy_order = (int*)malloc(params.singleIntRowByteSize);

    if (!host.spin || !host.E || !host.update || !host.accepted_flips
            || !replica_family || !energy_order) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    CUDA_CHECK(cudaMalloc((void**)&device.spin, params.fullLatticeByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.E, params.singleIntRowByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.update, params.singleIntRowByteSize));
    CUDA_CHECK(cudaMalloc((void**)&device.accepted_flips,
                          static_cast<size_t>(params.R) * sizeof(uint64_t)));

    void* curand_states = setup_curand_states(params);
    size_t free_after_setup = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_setup, &total_memory));
    gpu_metadata.free_memory_after_setup_bytes = free_after_setup;
    int upper_energy = potts_cooling_start_U();
    int lower_energy = potts_heating_start_U(params.N);
    int U = params.heat ? lower_energy : upper_energy;
    uint64_t completed_shells = 0;
    if (resumed) {
        U = checkpoint_state.U;
        completed_shells = checkpoint_state.completed_shells;
        memcpy(host.E, checkpoint_state.energies.data(), params.singleIntRowByteSize);
        memcpy(replica_family, checkpoint_state.families.data(),
               params.singleIntRowByteSize);
        memcpy(energy_order, checkpoint_state.order.data(),
               params.singleIntRowByteSize);
        set_resampling_rng_state(checkpoint_state.resampling_rng);
        CUDA_CHECK(cudaMemcpy(device.spin, checkpoint_state.spins.data(),
                              params.fullLatticeByteSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device.E, checkpoint_state.energies.data(),
                              params.singleIntRowByteSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(curand_states, checkpoint_state.philox.data(),
                              checkpoint_identity.philox_bytes,
                              cudaMemcpyHostToDevice));
        printf("[chk] Resumed %s at U=%d after %llu copied shells\n",
               checkpoint_source.c_str(), U,
               static_cast<unsigned long long>(completed_shells));
    } else {
        for (int i = 0; i < params.R; i++) {
            energy_order[i] = i;
            replica_family[i] = i;
        }
        initialize_resampling_rng(params.seed);
        initialize_population(curand_states, device, params);
        calc_device_energy(device, params);
    }

    bool checkpoint_failure = false;
    const bool deterministic_timings = getenv("MCPA_DETERMINISTIC_TIMINGS") != nullptr;
    if (deterministic_timings) {
        gpu_metadata.free_memory_before_setup_bytes = 0;
        gpu_metadata.free_memory_after_setup_bytes = 0;
    }
    int checkpoint_pause_ms = 0;
    if (const char* pause_text = getenv("MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS")) {
        if (!parse_int_arg(pause_text, "MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS",
                           &checkpoint_pause_ms) || checkpoint_pause_ms < 0) {
            fprintf(stderr, "MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS must be nonnegative\n");
            return 2;
        }
    }
    while ((U >= lower_energy && !params.heat)
            || (U <= upper_energy && params.heat)) {
        printf("U: %d out of %d; nSteps: %d;\n",
               U, -3 * params.N / 2, params.nSteps);

        const auto equilibrate_start = std::chrono::steady_clock::now();
        equilibrate(curand_states, device, params, U);
        double equilibrate_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - equilibrate_start).count();
        if (deterministic_timings) equilibrate_seconds = 0.0;
        CUDA_CHECK(cudaMemcpy(host.E, device.E, params.singleIntRowByteSize,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.spin, device.spin, params.fullLatticeByteSize,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.accepted_flips, device.accepted_flips,
                              static_cast<size_t>(params.R) * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));

        const std::vector<int> measured_families(
            replica_family, replica_family + params.R);
        const PottsResampleResult resample_result = resample(
            host.E, energy_order, host.update, replica_family,
            params.R, &U, params.heat);
        if (resample_result.status == POTTS_RESAMPLE_NO_NEXT_SHELL) {
            printf("Process ended: no next strict energy shell\n");
            break;
        }
        const PottsShellStatistics shell_stats = potts_shell_statistics(
            host.spin, host.E, host.accepted_flips, measured_families.data(),
            params.R, params.N, params.q, resample_result.new_U);
        if (shell_stats.replicas.size()
                != static_cast<size_t>(resample_result.n_cull)) {
            fprintf(stderr,
                    "Strict-shell population does not equal exact nCull\n");
            checkpoint_failure = true;
            break;
        }
        const PottsFamilyStatistics population_family_stats =
            potts_family_statistics(replica_family, params.R, params.R);
        const PottsGpuMetadata* row_metadata = completed_shells == 0
            ? &gpu_metadata : nullptr;
        if (!write_potts_main_row(main_file, resample_result, params.nSteps,
                                  equilibrate_seconds, population_family_stats,
                                  row_metadata)
                || !write_potts_aggregate_row(
                    aggregate_file, resample_result.new_U, params.N,
                    params.nSteps, shell_stats)
                || !write_potts_detailed_rows(
                    detailed_file, resample_result.new_U, shell_stats,
                    detailed_cap)) {
            fprintf(stderr, "Potts output write failed\n");
            checkpoint_failure = true;
            break;
        }

        if (resample_result.status == POTTS_RESAMPLE_TERMINAL_FULL_CULL) {
            printf("Process ended at terminal shell with culling fraction 1\n");
            break;
        }

        CUDA_CHECK(cudaMemcpy(device.update, host.update,
                              params.singleIntRowByteSize,
                              cudaMemcpyHostToDevice));
        update_replicas(device, params);
        ++completed_shells;
        if (checkpoint_enabled
                && completed_shells
                       % static_cast<uint64_t>(checkpoint_shell_interval) == 0) {
            /* update_replicas synchronizes: all serialized fields now describe
               the copied population entering the next strict shell. */
            checkpoint_state.U = U;
            checkpoint_state.completed_shells = completed_shells;
            checkpoint_state.resampling_rng = get_resampling_rng_state();
            checkpoint_state.spins.resize(params.fullLatticeByteSize);
            checkpoint_state.energies.resize(params.R);
            checkpoint_state.families.assign(
                replica_family, replica_family + params.R);
            checkpoint_state.order.assign(energy_order, energy_order + params.R);
            checkpoint_state.philox.resize(checkpoint_identity.philox_bytes);
            CUDA_CHECK(cudaMemcpy(checkpoint_state.spins.data(), device.spin,
                                  params.fullLatticeByteSize,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(checkpoint_state.energies.data(), device.E,
                                  params.singleIntRowByteSize,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(checkpoint_state.philox.data(), curand_states,
                                  checkpoint_identity.philox_bytes,
                                  cudaMemcpyDeviceToHost));
            if (!capture_output_offsets(output_files,
                                        &checkpoint_state.output_offsets)
                    || !save_potts_checkpoint(
                        checkpoint_base, checkpoint_identity, checkpoint_state,
                        &checkpoint_error)) {
                fprintf(stderr, "Potts checkpoint save failed: %s\n",
                        checkpoint_error.c_str());
                checkpoint_failure = true;
                break;
            }
            printf("[chk] Saved shell %llu at U=%d\n",
                   static_cast<unsigned long long>(completed_shells), U);
            if (checkpoint_pause_ms > 0)
                usleep(static_cast<useconds_t>(checkpoint_pause_ms) * 1000U);
        }
    }

    CUDA_CHECK(cudaFree(curand_states));
    CUDA_CHECK(cudaFree(device.spin));
    CUDA_CHECK(cudaFree(device.E));
    CUDA_CHECK(cudaFree(device.update));
    CUDA_CHECK(cudaFree(device.accepted_flips));
    free(host.spin);
    free(host.E);
    free(host.update);
    free(host.accepted_flips);
    free(replica_family);
    free(energy_order);

    fclose(main_file);
    fclose(aggregate_file);
    fclose(detailed_file);
    if (checkpoint_failure) return 3;

    if (checkpoint_enabled
            && !mark_potts_checkpoint_done(checkpoint_base, &checkpoint_error)) {
        fprintf(stderr, "Could not mark Potts checkpoint complete: %s\n",
                checkpoint_error.c_str());
        return 3;
    }
    return 0;
}
