#include "baxterwu_lib.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using RngState = curandStatePhilox4_32_10_t;

struct Config {
    std::string preset = "smoke";
    std::string variant = "both";
    std::string regime = "open";
    int L = 36;
    int R = 8192;
    int n_steps = 1;
    int threads = 256;
    int warmups = 2;
    int repeats = 7;
    int seed = 173;
    bool heat = false;
    bool json = false;
    bool check = true;
};

struct Sample {
    double kernel_ms;
    double wall_ms;
    double acceptance;
};

struct Summary {
    double median;
    double mad;
    double p10;
    double p90;
};

struct Population {
    mainMemoryPointers memory{};
    RngState* states = nullptr;
    size_t spin_bytes = 0;
    size_t row_bytes = 0;
    size_t stats_bytes = 0;

    explicit Population(const Params& params)
        : spin_bytes(params.fullLatticeByteSize),
          row_bytes(params.singleIntRowByteSize),
          stats_bytes(params.replicaStatisticsByteSize) {
        if (cudaMalloc(&memory.spin, spin_bytes) != cudaSuccess
            || cudaMalloc(&memory.E, row_bytes) != cudaSuccess
            || cudaMalloc(&memory.replica_statistics, stats_bytes) != cudaSuccess
            || cudaMalloc(&states, static_cast<size_t>(params.R) * sizeof(RngState))
                   != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed; reduce R or choose a smaller preset");
        }
    }

    ~Population() {
        cudaFree(memory.spin);
        cudaFree(memory.E);
        cudaFree(memory.replica_statistics);
        cudaFree(states);
    }

    Population(const Population&) = delete;
    Population& operator=(const Population&) = delete;
};

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void reset_population_kernel(RngState* states,
                                        mainMemoryPointers device,
                                        Params params,
                                        bool mixed) {
    const int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    const long long shift = static_cast<long long>(r) * params.N;
    for (int j = 0; j < params.N; ++j) {
        const int x = j % params.L;
        const int y = j / params.L;
        const int sublattice = (x + y) % 3;
        int spin = sublattice == 0 ? 1 : -1;
        // Spacing-three defects are isolated: no elementary triangle contains
        // two of them.  The exact fixture energy is therefore -2N/3.
        if (mixed && params.heat && (r % 2 != 0)) {
            // Global inversion maps the -2N ground branch to the +2N ceiling.
            spin = -spin;
        } else if (mixed && !params.heat
                   && (x % 3 == 0) && (y % 3 == 0)) {
            spin = -spin;
        }
        device.spin[shift + j] = spin;
    }
    device.E[r] = -2 * params.N;
    device.replica_statistics[r] = {};
    curand_init(params.seed, r, 0, &states[r]);
}

__device__ neiborsIndexes reference_neighbors(int j, const Params& params) {
    const int x = j % params.L;
    const int y = j / params.L;
    return {
        (x - 1 + params.L) % params.L + y * params.L,
        (x + 1) % params.L + y * params.L,
        x + ((y - 1 + params.L) % params.L) * params.L,
        x + ((y + 1) % params.L) * params.L,
        (x - 1 + params.L) % params.L + ((y - 1 + params.L) % params.L) * params.L,
        (x + 1) % params.L + ((y + 1) % params.L) * params.L,
    };
}

__device__ int reference_local_energy(const int* spins, long long shift, int j,
                                      const Params& params) {
    const neiborsIndexes n = reference_neighbors(j, params);
    const int s = spins[shift + j];
    return -s * (
        spins[shift + n.diag_left] * spins[shift + n.up]
      + spins[shift + n.diag_left] * spins[shift + n.left]
      + spins[shift + n.diag_right] * spins[shift + n.down]
      + spins[shift + n.diag_right] * spins[shift + n.right]
      + spins[shift + n.down] * spins[shift + n.left]
      + spins[shift + n.up] * spins[shift + n.right]);
}

__global__ void reference_equilibrate_kernel(RngState* states,
                                             mainMemoryPointers device,
                                             Params params,
                                             int U) {
    const int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    const long long shift = static_cast<long long>(r) * params.N;
    int flip_count = 0;
    uint4 randoms{};
    auto* values = reinterpret_cast<uint32_t*>(&randoms);

    for (int k = 0; k < params.N * params.nSteps; ++k) {
        if ((k & 3) == 0) randoms = curand4(&states[r]);
        const int j = static_cast<int>(values[k & 3] % static_cast<uint32_t>(params.N));
        const int local = reference_local_energy(device.spin, shift, j, params);
        const int suggested_energy = device.E[r] - 2 * local;
        if ((!params.heat && suggested_energy < U)
            || (params.heat && suggested_energy > U)) {
            device.E[r] = suggested_energy;
            device.spin[shift + j] = -device.spin[shift + j];
            ++flip_count;
        }
        device.replica_statistics[r].flip_count = flip_count;
    }
}

__global__ void compare_bytes_kernel(const unsigned char* lhs,
                                     const unsigned char* rhs,
                                     size_t count,
                                     unsigned long long* mismatches) {
    size_t i = static_cast<size_t>(threadIdx.x)
             + static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    unsigned long long local = 0;
    for (; i < count; i += stride) local += lhs[i] != rhs[i];
    if (local) atomicAdd(mismatches, local);
}

void reset_population(Population& population, const Params& params, bool mixed) {
    reset_population_kernel<<<params.blocks, params.threads>>>(
        population.states, population.memory, params, mixed);
    cuda_check(cudaPeekAtLastError(), "reset kernel launch");
    if (mixed) {
        // Recompute rather than trusting the analytic fixture energy.  Reset
        // work is outside the timed CUDA-event interval.
        calc_device_energy(population.memory, params);
    } else {
        cuda_check(cudaDeviceSynchronize(), "reset kernel execution");
    }
}

void launch_reference(Population& population, const Params& params, int U) {
    reference_equilibrate_kernel<<<params.blocks, params.threads>>>(
        population.states, population.memory, params, U);
    cuda_check(cudaPeekAtLastError(), "reference kernel launch");
    cuda_check(cudaDeviceSynchronize(), "reference kernel execution");
}

void launch_candidate(Population& population, const Params& params, int U) {
    equilibrate(population.states, population.memory, params, U);
}

int ceiling_for(const Config& config, const Params& params) {
    if (config.regime == "mixed") {
        const int fixture_energy = -2 * params.N / 3;
        // A single heating trajectory rapidly moves above a fixed lower
        // boundary and becomes almost fully open.  For heating, alternate
        // ground-state (frozen) and ceiling-state (open) replicas to retain a stable
        // accepted/rejected population mix.  Cooling uses the defect fixture
        // within every replica and remains intrinsically mixed.
        return config.heat ? 0 : fixture_energy + 1;
    }
    if (config.regime == "open") {
        return config.heat ? -2 * params.N - 2 : 2 * params.N + 2;
    }
    return config.heat ? 2 * params.N - 2 : -2 * params.N + 8;
}

void validate_mixed_fixture(Population& population, const Params& params) {
    reset_population(population, params, true);
    std::vector<int> energies(params.R);
    cuda_check(cudaMemcpy(energies.data(), population.memory.E,
                          params.singleIntRowByteSize, cudaMemcpyDeviceToHost),
               "mixed fixture energy copy");
    const int expected = -2 * params.N / 3;
    bool valid = true;
    for (int r = 0; r < params.R; ++r) {
        const int wanted = params.heat
            ? (r % 2 == 0 ? -2 * params.N : 2 * params.N)
            : expected;
        valid = valid && energies[r] == wanted;
    }
    if (!valid) {
        throw std::runtime_error("mixed fixture energy does not match its deterministic pattern");
    }
}

unsigned long long compare_device_buffers(const void* lhs, const void* rhs, size_t bytes) {
    unsigned long long* device_mismatches = nullptr;
    unsigned long long mismatches = 0;
    cuda_check(cudaMalloc(&device_mismatches, sizeof(mismatches)), "comparison allocation");
    cuda_check(cudaMemset(device_mismatches, 0, sizeof(mismatches)), "comparison reset");
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>(4096, (bytes + threads - 1) / threads));
    compare_bytes_kernel<<<std::max(1, blocks), threads>>>(
        static_cast<const unsigned char*>(lhs),
        static_cast<const unsigned char*>(rhs), bytes, device_mismatches);
    cuda_check(cudaPeekAtLastError(), "comparison kernel launch");
    cuda_check(cudaMemcpy(&mismatches, device_mismatches, sizeof(mismatches),
                          cudaMemcpyDeviceToHost), "comparison result copy");
    cudaFree(device_mismatches);
    return mismatches;
}

void verify_equivalence(Population& working, const Params& params,
                        const Config& config, int U) {
    int* candidate_spins = nullptr;
    int* candidate_energy = nullptr;
    replicaStatistics* candidate_stats = nullptr;
    RngState* candidate_states = nullptr;
    cuda_check(cudaMalloc(&candidate_spins, params.fullLatticeByteSize), "spin snapshot allocation");
    cuda_check(cudaMalloc(&candidate_energy, params.singleIntRowByteSize), "energy snapshot allocation");
    cuda_check(cudaMalloc(&candidate_stats, params.replicaStatisticsByteSize), "stats snapshot allocation");
    cuda_check(cudaMalloc(&candidate_states, static_cast<size_t>(params.R) * sizeof(RngState)),
               "RNG snapshot allocation");

    reset_population(working, params, config.regime == "mixed");
    launch_candidate(working, params, U);
    cuda_check(cudaMemcpy(candidate_spins, working.memory.spin, params.fullLatticeByteSize,
                          cudaMemcpyDeviceToDevice), "spin snapshot");
    cuda_check(cudaMemcpy(candidate_energy, working.memory.E, params.singleIntRowByteSize,
                          cudaMemcpyDeviceToDevice), "energy snapshot");
    cuda_check(cudaMemcpy(candidate_stats, working.memory.replica_statistics,
                          params.replicaStatisticsByteSize, cudaMemcpyDeviceToDevice),
               "statistics snapshot");
    cuda_check(cudaMemcpy(candidate_states, working.states,
                          static_cast<size_t>(params.R) * sizeof(RngState),
                          cudaMemcpyDeviceToDevice), "RNG snapshot");

    reset_population(working, params, config.regime == "mixed");
    launch_reference(working, params, U);

    const unsigned long long spin_diff = compare_device_buffers(
        candidate_spins, working.memory.spin, params.fullLatticeByteSize);
    const unsigned long long energy_diff = compare_device_buffers(
        candidate_energy, working.memory.E, params.singleIntRowByteSize);
    const unsigned long long stats_diff = compare_device_buffers(
        candidate_stats, working.memory.replica_statistics, params.replicaStatisticsByteSize);
    const unsigned long long rng_diff = compare_device_buffers(
        candidate_states, working.states, static_cast<size_t>(params.R) * sizeof(RngState));

    cudaFree(candidate_spins);
    cudaFree(candidate_energy);
    cudaFree(candidate_stats);
    cudaFree(candidate_states);

    if (spin_diff || energy_diff || stats_diff || rng_diff) {
        throw std::runtime_error(
            "bitwise check failed (spin bytes=" + std::to_string(spin_diff)
            + ", energy bytes=" + std::to_string(energy_diff)
            + ", statistics bytes=" + std::to_string(stats_diff)
            + ", RNG bytes=" + std::to_string(rng_diff) + ")");
    }
}

double acceptance_rate(const Population& population, const Params& params) {
    std::vector<replicaStatistics> stats(params.R);
    cuda_check(cudaMemcpy(stats.data(), population.memory.replica_statistics,
                          params.replicaStatisticsByteSize, cudaMemcpyDeviceToHost),
               "statistics copy");
    const long long accepted = std::accumulate(
        stats.begin(), stats.end(), 0LL,
        [](long long sum, const replicaStatistics& value) {
            return sum + value.flip_count;
        });
    const long long attempted = static_cast<long long>(params.R) * params.N * params.nSteps;
    return static_cast<double>(accepted) / static_cast<double>(attempted);
}

Sample measure_once(const std::string& variant,
                    Population& population,
                    const Params& params,
                    const Config& config,
                    int U) {
    reset_population(population, params, config.regime == "mixed");
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cuda_check(cudaEventCreate(&start), "event creation");
    cuda_check(cudaEventCreate(&stop), "event creation");
    cuda_check(cudaEventRecord(start), "start event");
    const auto wall_start = Clock::now();
    if (variant == "candidate") launch_candidate(population, params, U);
    else launch_reference(population, params, U);
    const auto wall_stop = Clock::now();
    cuda_check(cudaEventRecord(stop), "stop event");
    cuda_check(cudaEventSynchronize(stop), "stop event wait");
    float kernel_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&kernel_ms, start, stop), "event elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {
        kernel_ms,
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count(),
        acceptance_rate(population, params),
    };
}

Summary summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto quantile = [&values](double q) {
        const double position = q * static_cast<double>(values.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(values.size() - 1, lower + 1);
        const double fraction = position - static_cast<double>(lower);
        return values[lower] * (1.0 - fraction) + values[upper] * fraction;
    };
    const double median = quantile(0.5);
    std::vector<double> deviations(values.size());
    std::transform(values.begin(), values.end(), deviations.begin(),
                   [median](double value) { return std::abs(value - median); });
    std::sort(deviations.begin(), deviations.end());
    const double mad = deviations[deviations.size() / 2];
    return {median, mad, quantile(0.1), quantile(0.9)};
}

void apply_preset(Config& config) {
    if (config.preset == "smoke") {
        config.L = 36; config.R = 8192; config.n_steps = 1;
        config.warmups = 2; config.repeats = 7;
    } else if (config.preset == "production") {
        config.L = 36; config.R = 131072; config.n_steps = 2;
        config.warmups = 3; config.repeats = 10;
    } else if (config.preset == "working-set") {
        config.L = 162; config.R = 8192; config.n_steps = 1;
        config.warmups = 2; config.repeats = 7;
    } else {
        throw std::runtime_error("unknown preset: " + config.preset);
    }
}

int parse_int(const char* value, const char* option) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0) {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    return static_cast<int>(parsed);
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto value = [&](const char* option) {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + option);
            return argv[i];
        };
        if (argument == "--preset") config.preset = value("--preset");
        else if (argument == "--variant") config.variant = value("--variant");
        else if (argument == "--regime") config.regime = value("--regime");
        else if (argument == "--threads") config.threads = parse_int(value("--threads"), "--threads");
        else if (argument == "--repeats") config.repeats = parse_int(value("--repeats"), "--repeats");
        else if (argument == "--warmups") config.warmups = parse_int(value("--warmups"), "--warmups");
        else if (argument == "--seed") config.seed = parse_int(value("--seed"), "--seed");
        else if (argument == "--heat") config.heat = true;
        else if (argument == "--json") config.json = true;
        else if (argument == "--skip-check") config.check = false;
        else if (argument == "--help") {
            std::cout
                << "Usage: mcpa_equilibrate_bench [options]\n"
                << "  --preset smoke|production|working-set\n"
                << "  --variant both|reference|candidate\n"
                << "  --regime open|frozen|mixed\n"
                << "  --threads 128|256|512 --warmups N --repeats N\n"
                << "  --heat --json --skip-check\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    const int custom_threads = config.threads;
    const int custom_warmups = config.warmups;
    const int custom_repeats = config.repeats;
    apply_preset(config);
    if (custom_threads != 256) config.threads = custom_threads;
    if (custom_warmups != 2) config.warmups = custom_warmups;
    if (custom_repeats != 7) config.repeats = custom_repeats;
    if (config.variant != "both" && config.variant != "reference"
        && config.variant != "candidate") {
        throw std::runtime_error("variant must be both, reference, or candidate");
    }
    if (config.regime != "open" && config.regime != "frozen"
        && config.regime != "mixed") {
        throw std::runtime_error("regime must be open, frozen, or mixed");
    }
    if (config.L % 3 != 0) throw std::runtime_error("L must be divisible by 3");
    return config;
}

void print_result(const Config& config,
                  const cudaDeviceProp& device,
                  const std::string& variant,
                  const std::vector<Sample>& samples,
                  const Params& params) {
    std::vector<double> kernel;
    std::vector<double> wall;
    std::vector<double> acceptance;
    for (const Sample& sample : samples) {
        kernel.push_back(sample.kernel_ms);
        wall.push_back(sample.wall_ms);
        acceptance.push_back(sample.acceptance);
    }
    const Summary kernel_summary = summarize(kernel);
    const Summary wall_summary = summarize(wall);
    const double acceptance_mean = std::accumulate(acceptance.begin(), acceptance.end(), 0.0)
                                 / static_cast<double>(acceptance.size());
    const double attempts = static_cast<double>(params.R) * params.N * params.nSteps;
    const double updates_per_second = attempts / (kernel_summary.median / 1000.0);

    if (config.json) {
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"benchmark\":\"equilibrate\",\"variant\":\"" << variant
                  << "\",\"preset\":\"" << config.preset
                  << "\",\"regime\":\"" << config.regime
                  << "\",\"direction\":\"" << (config.heat ? "heating" : "cooling")
                  << "\",\"gpu\":\"" << device.name
                  << "\",\"L\":" << params.L << ",\"R\":" << params.R
                  << ",\"n_steps\":" << params.nSteps
                  << ",\"threads\":" << params.threads
                  << ",\"repeats\":" << config.repeats
                  << ",\"kernel_ms_median\":" << kernel_summary.median
                  << ",\"kernel_ms_mad\":" << kernel_summary.mad
                  << ",\"kernel_ms_p10\":" << kernel_summary.p10
                  << ",\"kernel_ms_p90\":" << kernel_summary.p90
                  << ",\"wall_ms_median\":" << wall_summary.median
                  << ",\"attempts_per_second\":" << updates_per_second
                  << ",\"acceptance\":" << acceptance_mean << "}\n";
    } else {
        std::cout << std::fixed << std::setprecision(3)
                  << std::left << std::setw(10) << variant
                  << " kernel " << kernel_summary.median << " ms"
                  << "  MAD " << kernel_summary.mad << " ms"
                  << "  p10-p90 [" << kernel_summary.p10 << ", " << kernel_summary.p90 << "]"
                  << "  wall " << wall_summary.median << " ms"
                  << "  " << std::setprecision(2) << updates_per_second / 1.0e9 << " Gattempt/s"
                  << "  acceptance " << 100.0 * acceptance_mean << "%\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config config = parse_args(argc, argv);
        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            std::cout << "SKIP: benchmark needs an accessible NVIDIA GPU";
            if (device_status != cudaSuccess) {
                std::cout << " (" << cudaGetErrorString(device_status) << ')';
            }
            std::cout << '\n';
            return 77;
        }
        cudaDeviceProp device{};
        cuda_check(cudaGetDeviceProperties(&device, 0), "CUDA device properties");

        Params params{};
        params.L = config.L;
        params.N = config.L * config.L;
        params.R = config.R;
        params.seed = config.seed;
        params.threads = config.threads;
        params.blocks = (config.R + config.threads - 1) / config.threads;
        params.nSteps = config.n_steps;
        params.heat = config.heat;
        params.fullLatticeByteSize = static_cast<size_t>(params.R) * params.N * sizeof(int);
        params.singleIntRowByteSize = static_cast<size_t>(params.R) * sizeof(int);
        params.replicaStatisticsByteSize = static_cast<size_t>(params.R) * sizeof(replicaStatistics);
        const int U = ceiling_for(config, params);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "CUDA memory query");
        const size_t check_peak = 2 * params.fullLatticeByteSize
                                + 2 * params.singleIntRowByteSize
                                + 2 * params.replicaStatisticsByteSize
                                + 2 * static_cast<size_t>(params.R) * sizeof(RngState);
        if (config.check && check_peak > free_bytes * 9 / 10) {
            throw std::runtime_error(
                "bitwise check needs about " + std::to_string(check_peak / 1048576)
                + " MiB but only " + std::to_string(free_bytes / 1048576)
                + " MiB is free; use a smaller preset/R");
        }

        if (!config.json) {
            std::cout << "GPU: " << device.name << "  compute " << device.major << '.' << device.minor
                      << "  free " << free_bytes / 1048576 << " MiB / "
                      << total_bytes / 1048576 << " MiB\n"
                      << "Preset: " << config.preset << "  L=" << params.L
                      << " R=" << params.R << " attempts/replica=" << params.N * params.nSteps
                      << " threads=" << params.threads << " regime=" << config.regime
                      << (params.heat ? " heating" : " cooling") << '\n';
        }

        Population population(params);
        if (config.regime == "mixed") validate_mixed_fixture(population, params);
        if (config.check) {
            verify_equivalence(population, params, config, U);
            if (!config.json) std::cout << "Correctness: PASS (spins, energies, statistics, Philox state bitwise equal)\n";
        }

        const std::vector<std::string> variants = config.variant == "both"
            ? std::vector<std::string>{"reference", "candidate"}
            : std::vector<std::string>{config.variant};
        for (const std::string& variant : variants) {
            for (int i = 0; i < config.warmups; ++i) {
                (void)measure_once(variant, population, params, config, U);
            }
            std::vector<Sample> samples;
            samples.reserve(config.repeats);
            for (int i = 0; i < config.repeats; ++i) {
                const Sample sample = measure_once(variant, population, params, config, U);
                if (config.regime == "mixed"
                    && !(sample.acceptance >= 0.05 && sample.acceptance <= 0.95)) {
                    throw std::runtime_error(
                        "mixed fixture acceptance is outside the required 5--95% range");
                }
                samples.push_back(sample);
            }
            print_result(config, device, variant, samples, params);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
