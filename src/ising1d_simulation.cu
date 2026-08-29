#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <cuda.h>
#include <curand_kernel.h>
#include <unistd.h>

#include "checkpoint_1d.h"
#include "ising1d_output.h"
#include "ising1d_runtime.h"

namespace {

ResamplingRngState resampling_rng{0, 1};

uint32_t next_resampling_random() {
    const uint64_t old_state = resampling_rng.state;
    resampling_rng.state = old_state * 6364136223846793005ULL
        + (resampling_rng.stream | 1ULL);
    const uint32_t xorshifted = static_cast<uint32_t>(
        ((old_state >> 18U) ^ old_state) >> 27U);
    const uint32_t rotation = static_cast<uint32_t>(old_state >> 59U);
    return (xorshifted >> rotation)
        | (xorshifted << ((-rotation) & 31U));
}

} // namespace

void initializeResamplingRng(int seed) {
    const uint64_t unsigned_seed = static_cast<uint64_t>(static_cast<uint32_t>(seed));
    resampling_rng.state = 0;
    resampling_rng.stream = (unsigned_seed << 1U) | 1U;
    (void)next_resampling_random();
    resampling_rng.state += unsigned_seed ^ 0x9e3779b97f4a7c15ULL;
    (void)next_resampling_random();
}

ResamplingRngState getResamplingRngState() { return resampling_rng; }

void setResamplingRngState(ResamplingRngState state) {
    state.stream |= 1ULL;
    resampling_rng = state;
}

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
    if (code == cudaSuccess) return;
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

struct neibors_indexes { int right; int left; };

__device__ neibors_indexes SLF(int j, int L, int N) {
    (void)L;
    return {(j + 1) % N, (j - 1 + N) % N};
}

struct neibors { char left; char right; };

__device__ neibors get_neibors_values(const char* spins,
                                      neibors_indexes indexes,
                                      int replica_shift) {
    return {spins[indexes.left + replica_shift],
            spins[indexes.right + replica_shift]};
}

__device__ int LocalE(char spin, neibors neighbors) {
    return -(spin * neighbors.left) - (spin * neighbors.right);
}

__device__ int DeltaE(char current, char suggested, neibors neighbors) {
    return LocalE(suggested, neighbors) - LocalE(current, neighbors);
}

__global__ void deviceEnergy(char* spins, int* energies, int L, int N) {
    const int replica = threadIdx.x + blockIdx.x * blockDim.x;
    const int shift = replica * N;
    int sum = 0;
    for (int site = 0; site < N; ++site) {
        const neibors_indexes indexes = SLF(site, L, N);
        sum += LocalE(spins[shift + site],
                      get_neibors_values(spins, indexes, shift));
    }
    energies[replica] = sum / 2;
}

__device__ char suggestSpin(curandStatePhilox4_32_10_t* state, int replica) {
    return static_cast<char>(2 * (curand(&state[replica]) % 2) - 1);
}

__device__ char suggestSpinSwap(curandStatePhilox4_32_10_t*, int,
                                char current_spin) {
    return -current_spin;
}

__global__ void initializePopulation(curandStatePhilox4_32_10_t* state,
                                     char* spins, int N, int q) {
    (void)q;
    const int replica = threadIdx.x + blockIdx.x * blockDim.x;
    const int shift = replica * N;
    for (int site = 0; site < N; ++site)
        spins[shift + site] = suggestSpin(state, replica);
}

__global__ void equilibrate(curandStatePhilox4_32_10_t* state, char* spins,
                            int* energies, int L, int N, int R, int q,
                            int nSteps, int U, int* flip_counts) {
    (void)R;
    (void)q;
    const int replica = threadIdx.x + blockIdx.x * blockDim.x;
    const int shift = replica * N;
    flip_counts[replica] = 0;
    for (int attempt = 0; attempt < N * nSteps; ++attempt) {
        const int site = curand(&state[replica]) % N;
        const char current = spins[shift + site];
        const char suggested = suggestSpinSwap(state, replica, current);
        const neibors_indexes indexes = SLF(site, L, N);
        const int delta = DeltaE(
            current, suggested, get_neibors_values(spins, indexes, shift));
        const int suggested_energy = energies[replica] + delta;
        if (suggested_energy < U) {
            energies[replica] = suggested_energy;
            spins[shift + site] = suggested;
            ++flip_counts[replica];
        }
    }
}

void Swap(int* values, int i, int j) {
    const int temporary = values[i];
    values[i] = values[j];
    values[j] = temporary;
}

void quicksort(int* energies, int* order, int left, int right, int direction) {
    const int middle = (left + right) / 2;
    int i = left;
    int j = right;
    const int pivot = direction * energies[order[middle]];
    while (left < j || i < right) {
        while (direction * energies[order[i]] > pivot) ++i;
        while (direction * energies[order[j]] < pivot) --j;
        if (i <= j) {
            Swap(order, i, j);
            ++i;
            --j;
        } else {
            if (left < j) quicksort(energies, order, left, j, direction);
            if (i < right) quicksort(energies, order, i, right, direction);
            return;
        }
    }
}

IsingResampleResult resample(int* energies, int* order, int* update,
                             int* families, int R, int* U, bool heat) {
    quicksort(energies, order, 0, R - 1, 1 - 2 * heat);
    const int old_U = *U;
    int new_U = old_U;
    for (int i = 0; i < R; ++i) {
        const int candidate = energies[order[i]];
        if ((!heat && candidate < old_U) || (heat && candidate > old_U)) {
            new_U = candidate;
            break;
        }
    }
    if (new_U == old_U)
        return {IsingResampleStatus::no_next_shell, old_U, old_U, R, 1.0};

    *U = new_U;
    int n_cull = 0;
    while (n_cull < R
           && ((!heat && energies[order[n_cull]] >= new_U)
               || (heat && energies[order[n_cull]] <= new_U)))
        ++n_cull;
    const double fraction = static_cast<double>(n_cull) / R;
    if (n_cull == R)
        return {IsingResampleStatus::terminal_full_cull,
                old_U, new_U, n_cull, fraction};

    for (int i = 0; i < R; ++i) update[i] = i;
    for (int i = 0; i < n_cull; ++i) {
        const int source = static_cast<int>(next_resampling_random()
            % static_cast<uint32_t>(R - n_cull)) + n_cull;
        update[order[i]] = order[source];
        families[order[i]] = families[order[source]];
    }
    return {IsingResampleStatus::ok, old_U, new_U, n_cull, fraction};
}

__global__ void updateReplicas(char* spins, int* energies, int* update, int N) {
    const int replica = threadIdx.x + blockIdx.x * blockDim.x;
    const int source = update[replica];
    if (source == replica) return;
    const int shift = replica * N;
    const int source_shift = source * N;
    for (int site = 0; site < N; ++site)
        spins[shift + site] = spins[source_shift + site];
    energies[replica] = energies[source];
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, int seed) {
    const int replica = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, replica, 0, state + replica);
}

#ifndef MCPA_1D_ISING_NO_MAIN
namespace {

struct RunGpuMetadata {
    char name[256]{};
    double compute_capability = 0.0;
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t free_memory_before_setup_bytes = 0;
    std::uint64_t free_memory_after_setup_bytes = 0;
    int cuda_driver_version = 0;
    int cuda_runtime_version = 0;
};

bool parse_int_argument(const char* text, const char* name, int minimum,
                        int* value) {
    if (text == nullptr || value == nullptr || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
        || parsed < minimum || parsed > INT_MAX) {
        fprintf(stderr, "ERROR: %s must be an integer >= %d (got '%s')\n",
                name, minimum, text);
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool parse_detailed_cap(int* cap) {
    const char* text = std::getenv("MCPA_DETAILED_CAP");
    if (text == nullptr || *text == '\0') {
        *cap = 100;
        return true;
    }
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
        || parsed < -1 || parsed > INT_MAX) {
        fprintf(stderr,
                "ERROR: MCPA_DETAILED_CAP must be -1 or an integer >= 0 (got '%s')\n",
                text);
        return false;
    }
    *cap = static_cast<int>(parsed);
    return true;
}

bool checked_int_product(int a, int b, const char* name, int* product) {
    const long long result = static_cast<long long>(a) * b;
    if (result > INT_MAX) {
        fprintf(stderr, "ERROR: %s exceeds INT_MAX\n", name);
        return false;
    }
    *product = static_cast<int>(result);
    return true;
}

bool ensure_output_directory(const char* path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (!error && std::filesystem::is_directory(path)) return true;
    fprintf(stderr, "ERROR: cannot create output directory %s: %s\n",
            path, error.message().c_str());
    return false;
}

FILE* open_output_file(const char* path, bool resume, std::int64_t offset) {
    FILE* file = fopen(path, resume ? "r+b" : "w");
    if (file == nullptr) {
        fprintf(stderr, "ERROR: cannot open output file %s: %s\n",
                path, strerror(errno));
        return nullptr;
    }
    if (resume
        && (ftruncate(fileno(file), static_cast<off_t>(offset)) != 0
            || fseek(file, static_cast<long>(offset), SEEK_SET) != 0)) {
        fprintf(stderr, "ERROR: cannot restore output offset for %s: %s\n",
                path, strerror(errno));
        fclose(file);
        return nullptr;
    }
    return file;
}

bool capture_output_offsets(
    const std::array<FILE*, kIsing1DOutputFileCount>& files,
    std::array<std::int64_t, kIsing1DOutputFileCount>* offsets) {
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (fflush(files[i]) != 0) return false;
        const long position = ftell(files[i]);
        if (position < 0) return false;
        (*offsets)[i] = static_cast<std::int64_t>(position);
    }
    return true;
}

RunGpuMetadata collect_gpu_metadata_before_setup() {
    RunGpuMetadata metadata;
    int device = 0;
    cudaCheckError(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    cudaCheckError(cudaGetDeviceProperties(&properties, device));
    std::snprintf(metadata.name, sizeof(metadata.name), "%s", properties.name);
    metadata.compute_capability = properties.major + properties.minor / 10.0;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaCheckError(cudaMemGetInfo(&free_bytes, &total_bytes));
    metadata.total_memory_bytes = total_bytes;
    metadata.free_memory_before_setup_bytes = free_bytes;
    cudaCheckError(cudaDriverGetVersion(&metadata.cuda_driver_version));
    cudaCheckError(cudaRuntimeGetVersion(&metadata.cuda_runtime_version));
    return metadata;
}

void collect_gpu_metadata_after_setup(RunGpuMetadata* metadata) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaCheckError(cudaMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;
    metadata->free_memory_after_setup_bytes = free_bytes;
}

void write_headers(const std::array<FILE*, kIsing1DOutputFileCount>& files) {
    fprintf(files[0],
        "E\tculling_factor\treplica_family_avg_sq\tnCull"
        "\tculling_factor_full_precision\tstatus\tnSteps"
        "\tequilibrate_seconds\tfamily_count\tfamily_max_size"
        "\tfamily_max_fraction\tfamily_shannon_entropy"
        "\tfamily_effective_shannon\tgpu_name\tgpu_compute_capability"
        "\tgpu_total_memory_bytes\tgpu_free_memory_before_setup_bytes"
        "\tgpu_free_memory_after_setup_bytes\tcuda_driver_version"
        "\tcuda_runtime_version\n");
    fprintf(files[1],
        "shell\tE\tshell_replica_count\taccepted_flip_sum"
        "\taccepted_flip_mean\taccepted_flip_rate\tM_sum\tM_mean"
        "\tabs_M_sum\tabs_M_mean\tM2_sum\tM2_mean"
        "\tpre_family_count\tpre_family_max_size"
        "\tpre_family_max_fraction\tpre_family_shannon_entropy"
        "\tpre_family_effective_shannon"
        "\tpre_family_simpson_concentration\n");
    fprintf(files[2],
        "shell\tE\treplica\tpre_resampling_family\tflips\tM\tabs_M\tM2"
        "\ttotal_matching_replicas\tdetailed_cap\tsampling_policy\n");
    for (FILE* file : files) fflush(file);
}

const char* status_name(IsingResampleStatus status) {
    switch (status) {
        case IsingResampleStatus::ok: return "ok";
        case IsingResampleStatus::no_next_shell: return "no_next_shell";
        case IsingResampleStatus::terminal_full_cull: return "terminal_full_cull";
    }
    return "invalid";
}

void write_main_row(FILE* file, const IsingResampleResult& result,
                    int n_steps, double equilibrate_seconds,
                    const IsingFamilyMetrics& family,
                    const RunGpuMetadata* metadata) {
    fprintf(file,
        "%d\t%.6f\t%.6f\t%d\t%.17g\t%s\t%d\t%.9g"
        "\t%d\t%d\t%.17g\t%.17g\t%.17g",
        result.new_U, result.culling_fraction,
        family.replica_family_avg_sq, result.n_cull,
        result.culling_fraction, status_name(result.status), n_steps,
        equilibrate_seconds, family.count, family.max_size,
        family.max_fraction, family.shannon_entropy,
        family.effective_shannon);
    if (metadata != nullptr) {
        fprintf(file, "\t%s\t%.1f\t%llu\t%llu\t%llu\t%d\t%d\n",
            metadata->name, metadata->compute_capability,
            static_cast<unsigned long long>(metadata->total_memory_bytes),
            static_cast<unsigned long long>(metadata->free_memory_before_setup_bytes),
            static_cast<unsigned long long>(metadata->free_memory_after_setup_bytes),
            metadata->cuda_driver_version, metadata->cuda_runtime_version);
    } else {
        fprintf(file, "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n");
    }
    fflush(file);
}

void write_aggregate_row(FILE* file, std::uint64_t shell, int energy,
                         const IsingShellAggregate& aggregate) {
    const IsingFamilyMetrics& family = aggregate.pre_resampling_family;
    fprintf(file,
        "%llu\t%d\t%d\t%lld\t%.17g\t%.17g\t%lld\t%.17g"
        "\t%lld\t%.17g\t%lld\t%.17g\t%d\t%d\t%.17g\t%.17g"
        "\t%.17g\t%.17g\n",
        static_cast<unsigned long long>(shell), energy,
        aggregate.shell_replica_count,
        static_cast<long long>(aggregate.accepted_flip_sum),
        aggregate.accepted_flip_mean, aggregate.accepted_flip_rate,
        static_cast<long long>(aggregate.magnetization_sum),
        aggregate.magnetization_mean,
        static_cast<long long>(aggregate.absolute_magnetization_sum),
        aggregate.absolute_magnetization_mean,
        static_cast<long long>(aggregate.magnetization_squared_sum),
        aggregate.magnetization_squared_mean, family.count, family.max_size,
        family.max_fraction, family.shannon_entropy,
        family.effective_shannon, family.simpson_concentration);
    fflush(file);
}

void write_detailed_rows(FILE* file, std::uint64_t shell, int energy,
                         const int* energies, const int* pre_families,
                         const std::vector<IsingReplicaMeasurement>& measurements,
                         int R, int detailed_cap) {
    int total_matching = 0;
    for (int replica = 0; replica < R; ++replica)
        if (energies[replica] == energy) ++total_matching;
    int printed = 0;
    for (int replica = 0; replica < R; ++replica) {
        if (energies[replica] != energy) continue;
        if (detailed_cap >= 0 && printed >= detailed_cap) break;
        const IsingReplicaMeasurement& measured
            = measurements[static_cast<std::size_t>(replica)];
        fprintf(file, "%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%lld\t%d\t%d"
            "\tfirst_matching_replica_indices\n",
            static_cast<unsigned long long>(shell), energy, replica,
            pre_families[replica], measured.flips, measured.magnetization,
            measured.absolute_magnetization,
            static_cast<long long>(measured.magnetization_squared),
            total_matching, detailed_cap);
        ++printed;
    }
    fflush(file);
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 8) {
#ifdef MCPA_1D_ISING_LEGACY_L_SQUARED_CLI
        fprintf(stderr, "Usage: %s run_number L blocks threads nSteps [checkpoint_dir|none [checkpoint_shell_interval]]\n", argv[0]);
#else
        fprintf(stderr, "Usage: %s run_number N blocks threads nSteps [checkpoint_dir|none [checkpoint_shell_interval]]\n", argv[0]);
#endif
        return 2;
    }

    int detailed_cap = 100;
    if (!parse_detailed_cap(&detailed_cap)) return 2;
    int run_number = 0;
    int site_argument = 0;
    int blocks = 0;
    int threads = 0;
    int n_steps = 0;
    if (!parse_int_argument(argv[1], "run_number", 0, &run_number)
        || !parse_int_argument(argv[2],
#ifdef MCPA_1D_ISING_LEGACY_L_SQUARED_CLI
            "L",
#else
            "N",
#endif
            3, &site_argument)
        || !parse_int_argument(argv[3], "blocks", 1, &blocks)
        || !parse_int_argument(argv[4], "threads", 1, &threads)
        || !parse_int_argument(argv[5], "nSteps", 1, &n_steps)) return 2;
    if (threads > 1024) {
        fprintf(stderr, "ERROR: threads must be <= 1024\n");
        return 2;
    }

    int L = site_argument;
    int N = site_argument;
#ifdef MCPA_1D_ISING_LEGACY_L_SQUARED_CLI
    if (!checked_int_product(L, L, "L*L", &N)) return 2;
#else
    L = N;
#endif
    if (N > INT_MAX - 2) {
        fprintf(stderr, "ERROR: N is too large for N+2\n");
        return 2;
    }
    int R = 0;
    int lattice_entries = 0;
    if (!checked_int_product(blocks, threads, "blocks*threads", &R)
        || !checked_int_product(N, R, "N*R", &lattice_entries)) return 2;

    const bool checkpoint_enabled
        = argc >= 7 && std::strcmp(argv[6], "none") != 0;
    int checkpoint_interval = 1;
    if (argc == 8
        && !parse_int_argument(argv[7], "checkpoint_shell_interval", 1,
                               &checkpoint_interval)) return 2;
    if (!ensure_output_directory("datasets/test")) return 2;
    if (checkpoint_enabled && !ensure_output_directory(argv[6])) return 2;

    char name_buffer[256];
    std::snprintf(name_buffer, sizeof(name_buffer),
                  "1DIsing_N%d_R%d_nSteps%d_run%d", N, R, n_steps, run_number);
    const std::string run_base = name_buffer;
    const std::string checkpoint_base = checkpoint_enabled
        ? (std::filesystem::path(argv[6]) / run_base).string() : std::string{};
    const Ising1DCheckpointIdentity checkpoint_identity{
        N, R, n_steps, run_number, detailed_cap,
        static_cast<std::size_t>(R) * sizeof(curandStatePhilox4_32_10_t)};

    Ising1DCheckpointState checkpoint_state;
    std::string checkpoint_source;
    std::string checkpoint_error;
    Ising1DCheckpointLoadStatus checkpoint_status
        = Ising1DCheckpointLoadStatus::not_found;
    if (checkpoint_enabled) {
        checkpoint_status = load_ising1d_checkpoint(
            checkpoint_base, checkpoint_identity, &checkpoint_state,
            &checkpoint_source, &checkpoint_error);
        if (checkpoint_status == Ising1DCheckpointLoadStatus::invalid) {
            fprintf(stderr, "ERROR: no valid checkpoint generation: %s\n",
                    checkpoint_error.c_str());
            return 3;
        }
        if (checkpoint_status == Ising1DCheckpointLoadStatus::done) {
            printf("[chk] Run is already complete; outputs are unchanged.\n");
            return 0;
        }
    }
    const bool resumed = checkpoint_status == Ising1DCheckpointLoadStatus::loaded;

    const std::array<const char*, kIsing1DOutputFileCount> suffixes{{
        "_main.txt", "_agg_stats.txt", "_detailed_stats.txt"}};
    std::array<std::string, kIsing1DOutputFileCount> paths;
    for (std::size_t i = 0; i < paths.size(); ++i)
        paths[i] = std::string("datasets/test/") + run_base + suffixes[i];
    if (resumed) {
        for (std::size_t i = 0; i < paths.size(); ++i) {
            std::error_code error;
            const std::uintmax_t size = std::filesystem::file_size(paths[i], error);
            if (error || checkpoint_state.output_offsets[i] < 0
                || size < static_cast<std::uintmax_t>(checkpoint_state.output_offsets[i])) {
                fprintf(stderr, "ERROR: checkpoint output missing/short: %s\n",
                        paths[i].c_str());
                return 3;
            }
        }
    }

    RunGpuMetadata gpu_metadata = collect_gpu_metadata_before_setup();
    const std::size_t spin_bytes = static_cast<std::size_t>(lattice_entries);
    const std::size_t int_row_bytes = static_cast<std::size_t>(R) * sizeof(int);
    std::vector<char> host_spins(spin_bytes);
    std::vector<int> host_energies(static_cast<std::size_t>(R));
    std::vector<int> host_updates(static_cast<std::size_t>(R));
    std::vector<int> host_flips(static_cast<std::size_t>(R));
    std::vector<int> families(static_cast<std::size_t>(R));
    std::vector<int> pre_families(static_cast<std::size_t>(R));
    std::vector<int> order(static_cast<std::size_t>(R));

    char* device_spins = nullptr;
    int* device_energies = nullptr;
    int* device_updates = nullptr;
    int* device_flips = nullptr;
    curandStatePhilox4_32_10_t* philox = nullptr;
    cudaCheckError(cudaMalloc(&device_spins, spin_bytes));
    cudaCheckError(cudaMalloc(&device_energies, int_row_bytes));
    cudaCheckError(cudaMalloc(&device_updates, int_row_bytes));
    cudaCheckError(cudaMalloc(&device_flips, int_row_bytes));
    cudaCheckError(cudaMalloc(&philox, checkpoint_identity.philox_bytes));

    int U = N + 2;
    std::uint64_t completed_shells = 0;
    if (resumed) {
        U = checkpoint_state.U;
        completed_shells = checkpoint_state.completed_shells;
        families = checkpoint_state.families;
        order = checkpoint_state.order;
        setResamplingRngState(checkpoint_state.resampling_rng);
        cudaCheckError(cudaMemcpy(device_spins, checkpoint_state.spins.data(),
                                  spin_bytes, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(device_energies, checkpoint_state.energies.data(),
                                  int_row_bytes, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(philox, checkpoint_state.philox.data(),
                                  checkpoint_identity.philox_bytes,
                                  cudaMemcpyHostToDevice));
        printf("[chk] Resumed %s at U=%d after %llu copied shells\n",
               checkpoint_source.c_str(), U,
               static_cast<unsigned long long>(completed_shells));
    } else {
        for (int replica = 0; replica < R; ++replica)
            families[replica] = order[replica] = replica;
        setup_kernel<<<blocks, threads>>>(philox, run_number);
        cudaCheckError(cudaPeekAtLastError());
        initializeResamplingRng(run_number);
        initializePopulation<<<blocks, threads>>>(philox, device_spins, N, 2);
        cudaCheckError(cudaPeekAtLastError());
        deviceEnergy<<<blocks, threads>>>(device_spins, device_energies, L, N);
        cudaCheckError(cudaPeekAtLastError());
    }
    cudaCheckError(cudaDeviceSynchronize());
    collect_gpu_metadata_after_setup(&gpu_metadata);

    std::array<FILE*, kIsing1DOutputFileCount> files{};
    for (std::size_t i = 0; i < files.size(); ++i)
        files[i] = open_output_file(paths[i].c_str(), resumed,
            resumed ? checkpoint_state.output_offsets[i] : 0);
    if (std::any_of(files.begin(), files.end(),
                    [](FILE* file) { return file == nullptr; })) {
        for (FILE* file : files) if (file) fclose(file);
        cudaFree(philox); cudaFree(device_flips); cudaFree(device_updates);
        cudaFree(device_energies); cudaFree(device_spins);
        return 2;
    }
    if (!resumed) write_headers(files);

    bool checkpoint_failure = false;
    bool write_gpu_metadata = !resumed;
    while (U >= -N - 2) {
        const auto equilibrate_start = std::chrono::steady_clock::now();
        equilibrate<<<blocks, threads>>>(
            philox, device_spins, device_energies, L, N, R, 2,
            n_steps, U, device_flips);
        cudaCheckError(cudaPeekAtLastError());
        cudaCheckError(cudaDeviceSynchronize());
        const double equilibrate_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - equilibrate_start).count();

        cudaCheckError(cudaMemcpy(host_energies.data(), device_energies,
                                  int_row_bytes, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(host_flips.data(), device_flips,
                                  int_row_bytes, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(host_spins.data(), device_spins,
                                  spin_bytes, cudaMemcpyDeviceToHost));
        pre_families = families;
        const std::vector<IsingReplicaMeasurement> measurements
            = measure_ising_replicas(host_spins.data(), host_flips.data(), N, R);

        const IsingResampleResult result = resample(
            host_energies.data(), order.data(), host_updates.data(),
            families.data(), R, &U, false);
        const IsingFamilyMetrics post_family
            = calculate_ising_family_metrics(families.data(), R);
        const IsingShellAggregate aggregate = aggregate_ising_shell(
            measurements, host_energies.data(), pre_families.data(),
            R, result.new_U, N, n_steps);
        write_main_row(files[0], result, n_steps, equilibrate_seconds,
                       post_family, write_gpu_metadata ? &gpu_metadata : nullptr);
        write_gpu_metadata = false;
        write_aggregate_row(files[1], completed_shells, result.new_U, aggregate);
        write_detailed_rows(files[2], completed_shells, result.new_U,
                            host_energies.data(), pre_families.data(),
                            measurements, R, detailed_cap);
        printf("U: %d; culling fraction: %.17g; status: %s\n",
               result.new_U, result.culling_fraction, status_name(result.status));

        if (result.status != IsingResampleStatus::ok) break;
        cudaCheckError(cudaMemcpy(device_updates, host_updates.data(),
                                  int_row_bytes, cudaMemcpyHostToDevice));
        updateReplicas<<<blocks, threads>>>(
            device_spins, device_energies, device_updates, N);
        cudaCheckError(cudaPeekAtLastError());
        cudaCheckError(cudaDeviceSynchronize());
        ++completed_shells;

        if (checkpoint_enabled
            && completed_shells % static_cast<std::uint64_t>(checkpoint_interval) == 0) {
            checkpoint_state.U = U;
            checkpoint_state.completed_shells = completed_shells;
            checkpoint_state.resampling_rng = getResamplingRngState();
            checkpoint_state.spins.resize(spin_bytes);
            checkpoint_state.energies.resize(R);
            checkpoint_state.families = families;
            checkpoint_state.order = order;
            checkpoint_state.philox.resize(checkpoint_identity.philox_bytes);
            cudaCheckError(cudaMemcpy(checkpoint_state.spins.data(), device_spins,
                                      spin_bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(checkpoint_state.energies.data(), device_energies,
                                      int_row_bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(checkpoint_state.philox.data(), philox,
                                      checkpoint_identity.philox_bytes,
                                      cudaMemcpyDeviceToHost));
            if (!capture_output_offsets(files, &checkpoint_state.output_offsets)
                || !save_ising1d_checkpoint(
                    checkpoint_base, checkpoint_identity, checkpoint_state,
                    &checkpoint_error)) {
                fprintf(stderr, "ERROR: checkpoint save failed: %s\n",
                        checkpoint_error.c_str());
                checkpoint_failure = true;
                break;
            }
        }
    }

    for (FILE* file : files) fclose(file);
    cudaFree(philox); cudaFree(device_flips); cudaFree(device_updates);
    cudaFree(device_energies); cudaFree(device_spins);
    if (checkpoint_failure) return 3;
    if (checkpoint_enabled
        && !mark_ising1d_checkpoint_done(checkpoint_base, &checkpoint_error)) {
        fprintf(stderr, "ERROR: could not mark checkpoint complete: %s\n",
                checkpoint_error.c_str());
        return 3;
    }
    return 0;
}
#endif
