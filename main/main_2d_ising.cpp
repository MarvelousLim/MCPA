#include "mcpa/ising2d_cuda.hpp"
#include "mcpa/ising2d_checkpoint.hpp"
#include "mcpa/ising2d_model.hpp"
#include "mcpa/ising2d_resampling.hpp"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using namespace mcpa::ising2d;
using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_cuda(cudaError_t status, const char* operation) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw_cuda(status, operation);
}

std::uint64_t parse_u64(const char* text, const char* name) {
    if (text == nullptr || *text == '\0' || *text == '-')
        throw std::invalid_argument(std::string("invalid ") + name);
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0')
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    return static_cast<std::uint64_t>(value);
}

int parse_positive_int(const char* text, const char* name) {
    const std::uint64_t value = parse_u64(text, name);
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " must be in [1, INT_MAX]");
    return static_cast<int>(value);
}

int parse_nonnegative_int(const char* text, const char* name) {
    const std::uint64_t value = parse_u64(text, name);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " exceeds INT_MAX");
    return static_cast<int>(value);
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename T>
T* allocate_device(std::size_t count, const char* description) {
    T* pointer = nullptr;
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)),
                 description);
    return pointer;
}

struct DeviceRun {
    DeviceEngineView engine{};
    DeviceReplicaCopyWorkspace workspace{};
    ReplicaIndex* parents = nullptr;

    explicit DeviceRun(EngineShape shape) {
        engine.population.spins = allocate_device<Spin>(
            static_cast<std::size_t>(shape.site_count) * shape.replicas,
            "allocate device spins");
        engine.population.energies = allocate_device<Energy>(shape.replicas,
                                                              "allocate energies");
        engine.population.flip_counts = allocate_device<FlipCount>(
            shape.replicas, "allocate flip counts");
        engine.philox = allocate_device<PhiloxState>(shape.replicas,
                                                     "allocate Philox states");
        workspace.spins = allocate_device<Spin>(
            static_cast<std::size_t>(shape.site_count) * shape.replicas,
            "allocate replica-copy spin scratch");
        workspace.energies = allocate_device<Energy>(
            shape.replicas, "allocate replica-copy energy scratch");
        parents = allocate_device<ReplicaIndex>(shape.replicas,
                                                "allocate parent indices");
    }

    ~DeviceRun() {
        cudaFree(parents);
        cudaFree(workspace.energies);
        cudaFree(workspace.spins);
        cudaFree(engine.philox);
        cudaFree(engine.population.flip_counts);
        cudaFree(engine.population.energies);
        cudaFree(engine.population.spins);
    }

    DeviceRun(const DeviceRun&) = delete;
    DeviceRun& operator=(const DeviceRun&) = delete;
};

struct RunGpuMetadata {
    std::string name;
    std::string compute_capability;
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t free_memory_before_setup_bytes = 0;
    std::uint64_t free_memory_after_setup_bytes = 0;
    int cuda_driver_version = 0;
    int cuda_runtime_version = 0;
};

std::string tsv_safe(std::string value) {
    for (char& character : value) {
        if (character == '\t' || character == '\n' || character == '\r')
            character = ' ';
    }
    return value;
}

RunGpuMetadata collect_gpu_metadata_before_setup(
    const cudaDeviceProp& properties) {
    RunGpuMetadata metadata{};
    metadata.name = tsv_safe(properties.name);
    metadata.compute_capability = std::to_string(properties.major) + "."
                                + std::to_string(properties.minor);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
                 "query CUDA memory before setup");
    metadata.total_memory_bytes = static_cast<std::uint64_t>(total_bytes);
    metadata.free_memory_before_setup_bytes = static_cast<std::uint64_t>(free_bytes);
    require_cuda(cudaDriverGetVersion(&metadata.cuda_driver_version),
                 "query CUDA driver version");
    require_cuda(cudaRuntimeGetVersion(&metadata.cuda_runtime_version),
                 "query CUDA runtime version");
    return metadata;
}

void collect_gpu_metadata_after_setup(RunGpuMetadata& metadata) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
                 "query CUDA memory after setup");
    metadata.free_memory_after_setup_bytes = static_cast<std::uint64_t>(free_bytes);
}

struct OutputFiles {
    std::ofstream main;
    std::ofstream aggregate;
    std::ofstream detailed;
};

std::array<std::filesystem::path, 3> output_paths(
    const std::filesystem::path& directory, const std::string& basename) {
    return {{directory / (basename + "_main.txt"),
             directory / (basename + "_agg_stats.txt"),
             directory / (basename + "_detailed_stats.txt")}};
}

OutputFiles open_outputs(const std::filesystem::path& directory,
                         const std::string& basename,
                         const std::array<std::uint64_t, 3>* resume_offsets) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error || !std::filesystem::is_directory(directory, error) || error)
        throw std::runtime_error("cannot create output directory "
                                 + directory.string() + ": " + error.message());

    OutputFiles files;
    files.main.exceptions(std::ios::failbit | std::ios::badbit);
    files.aggregate.exceptions(std::ios::failbit | std::ios::badbit);
    files.detailed.exceptions(std::ios::failbit | std::ios::badbit);
    const auto paths = output_paths(directory, basename);
    if (resume_offsets != nullptr) {
        for (std::size_t index = 0; index < paths.size(); ++index) {
            if (!std::filesystem::is_regular_file(paths[index]))
                throw std::runtime_error("checkpoint output is missing: "
                                         + paths[index].string());
            const std::uintmax_t size = std::filesystem::file_size(paths[index]);
            if ((*resume_offsets)[index] > size)
                throw std::runtime_error("checkpoint output offset exceeds file size: "
                                         + paths[index].string());
            std::filesystem::resize_file(paths[index], (*resume_offsets)[index]);
        }
        files.main.open(paths[0], std::ios::out | std::ios::app);
        files.aggregate.open(paths[1], std::ios::out | std::ios::app);
        files.detailed.open(paths[2], std::ios::out | std::ios::app);
        return files;
    }
    files.main.open(paths[0], std::ios::out | std::ios::trunc);
    files.aggregate.open(paths[1], std::ios::out | std::ios::trunc);
    files.detailed.open(paths[2], std::ios::out | std::ios::trunc);
    files.main
        << "E\tculling_factor\treplica_family_avg_sq\tnCull"
           "\tculling_factor_full_precision\tstatus"
           "\tequilibrate_seconds\tcopy_d2h_seconds\tresample_seconds"
           "\treplica_copy_seconds\tshell_seconds"
           "\tfamily_count\tfamily_max_size\tfamily_max_fraction"
           "\tfamily_shannon_entropy\tfamily_effective_shannon"
           "\tgpu_name\tgpu_compute_capability\tgpu_total_memory_bytes"
           "\tgpu_free_memory_before_setup_bytes"
           "\tgpu_free_memory_after_setup_bytes"
           "\tcuda_driver_version\tcuda_runtime_version\n";
    files.aggregate
        << "E\tn_rep\tflip_count_sum\tflip_count_mean"
           "\tflip_count_min\tflip_count_max\tflip_rate"
           "\tabs_m_mean\tm2_mean\tfamily_count_at_E"
           "\tfamily_max_size_at_E\tfamily_max_fraction_at_E"
           "\tfamily_simpson_at_E\n";
    files.detailed
        << "E\treplica\tfamily\tflip_count\tM\tabs_M\tM2"
           "\tdetailed_cap\tselection_policy\n";
    return files;
}

std::array<std::uint64_t, 3> output_offsets(OutputFiles& files) {
    const std::streampos positions[] = {
        files.main.tellp(), files.aggregate.tellp(), files.detailed.tellp()};
    std::array<std::uint64_t, 3> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (positions[index] < 0)
            throw std::runtime_error("cannot query checkpoint output offset");
        result[index] = static_cast<std::uint64_t>(positions[index]);
    }
    return result;
}

void sync_outputs(OutputFiles& files,
                  const std::array<std::filesystem::path, 3>& paths) {
    files.main.flush();
    files.aggregate.flush();
    files.detailed.flush();
    for (const auto& path : paths) {
        const int descriptor = open(path.c_str(), O_RDONLY);
        if (descriptor < 0)
            throw std::runtime_error("cannot open output for fsync " + path.string()
                                     + ": " + std::strerror(errno));
        if (fsync(descriptor) != 0) {
            const std::string message = std::strerror(errno);
            close(descriptor);
            throw std::runtime_error("cannot fsync output " + path.string()
                                     + ": " + message);
        }
        if (close(descriptor) != 0)
            throw std::runtime_error("cannot close synced output " + path.string());
    }
}

struct ShellStatistics {
    std::uint64_t count = 0;
    std::uint64_t flip_sum = 0;
    FlipCount flip_min = std::numeric_limits<FlipCount>::max();
    FlipCount flip_max = 0;
    long double absolute_magnetization_sum = 0;
    long double magnetization_sq_sum = 0;
    FamilyMetrics families{};
};

ShellStatistics calculate_shell_statistics(
    const std::vector<Spin>& spins, const std::vector<Energy>& energies,
    const std::vector<FlipCount>& flips, const std::vector<FamilyId>& families,
    EngineShape shape, Energy shell) {
    ShellStatistics result;
    std::vector<int> family_histogram(static_cast<std::size_t>(shape.replicas), 0);
    for (int replica = 0; replica < shape.replicas; ++replica) {
        if (energies[replica] != shell) continue;
        ++result.count;
        result.flip_sum += flips[replica];
        result.flip_min = std::min(result.flip_min, flips[replica]);
        result.flip_max = std::max(result.flip_max, flips[replica]);
        Energy magnetization = 0;
        const std::size_t shift
            = static_cast<std::size_t>(replica) * shape.site_count;
        for (int site = 0; site < shape.site_count; ++site)
            magnetization += spins[shift + site];
        result.absolute_magnetization_sum += std::abs(magnetization);
        result.magnetization_sq_sum
            += static_cast<long double>(magnetization) * magnetization;
        ++family_histogram[static_cast<std::size_t>(families[replica])];
    }
    if (result.count == 0)
        throw std::logic_error("selected shell has no replicas");
    for (const int size : family_histogram) {
        if (size == 0) continue;
        ++result.families.count;
        result.families.max_size = std::max(result.families.max_size, size);
        const double fraction = static_cast<double>(size) / result.count;
        result.families.shannon_entropy -= fraction * std::log(fraction);
        result.families.simpson_concentration += fraction * fraction;
    }
    result.families.max_fraction
        = static_cast<double>(result.families.max_size) / result.count;
    result.families.effective_shannon
        = std::exp(result.families.shannon_entropy);
    return result;
}

void write_shell(OutputFiles& files, const ResampleResult& resample,
                 const FamilyMetrics& population_families,
                 const ShellStatistics& shell,
                 const std::vector<Spin>& spins,
                 const std::vector<Energy>& energies,
                 const std::vector<FlipCount>& flips,
                 const std::vector<FamilyId>& measured_families,
                 EngineShape shape, int sweeps, int detailed_cap,
                 double equilibrate_seconds, double copy_d2h_seconds,
                 double resample_seconds, double replica_copy_seconds,
                 double shell_seconds,
                 const RunGpuMetadata* gpu_metadata) {
    if (shell.count != resample.culled)
        throw std::logic_error("strict-shell population does not equal nCull");
    files.main << resample.boundary << '\t'
               << std::setprecision(17) << resample.culling_fraction << '\t'
               << population_families.simpson_concentration << '\t'
               << resample.culled << '\t' << resample.culling_fraction << '\t'
               << resample_status_name(resample.status) << '\t'
               << equilibrate_seconds << '\t' << copy_d2h_seconds << '\t'
               << resample_seconds << '\t' << replica_copy_seconds << '\t'
               << shell_seconds << '\t' << population_families.count << '\t'
               << population_families.max_size << '\t'
               << population_families.max_fraction << '\t'
               << population_families.shannon_entropy << '\t'
               << population_families.effective_shannon;
    if (gpu_metadata != nullptr) {
        files.main << '\t' << gpu_metadata->name
                   << '\t' << gpu_metadata->compute_capability
                   << '\t' << gpu_metadata->total_memory_bytes
                   << '\t' << gpu_metadata->free_memory_before_setup_bytes
                   << '\t' << gpu_metadata->free_memory_after_setup_bytes
                   << '\t' << gpu_metadata->cuda_driver_version
                   << '\t' << gpu_metadata->cuda_runtime_version << '\n';
    } else {
        files.main << "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n";
    }

    const long double count = static_cast<long double>(shell.count);
    const long double attempts_per_replica
        = static_cast<long double>(shape.site_count) * sweeps;
    const long double flip_sum = static_cast<long double>(shell.flip_sum);
    files.aggregate << resample.boundary << '\t' << shell.count << '\t'
                    << shell.flip_sum << '\t' << std::setprecision(21)
                    << flip_sum / count << '\t'
                    << shell.flip_min << '\t' << shell.flip_max << '\t'
                    << flip_sum / (count * attempts_per_replica) << '\t'
                    << shell.absolute_magnetization_sum / count << '\t'
                    << shell.magnetization_sq_sum / count << '\t'
                    << shell.families.count << '\t'
                    << shell.families.max_size << '\t'
                    << shell.families.max_fraction << '\t'
                    << shell.families.simpson_concentration << '\n';

    int written = 0;
    for (int replica = 0;
         replica < shape.replicas && written < detailed_cap; ++replica) {
        if (energies[replica] != resample.boundary) continue;
        Energy magnetization = 0;
        const std::size_t shift
            = static_cast<std::size_t>(replica) * shape.site_count;
        for (int site = 0; site < shape.site_count; ++site)
            magnetization += spins[shift + site];
        const Energy absolute_magnetization = std::abs(magnetization);
        const Energy magnetization_sq = magnetization * magnetization;
        files.detailed << resample.boundary << '\t' << replica << '\t'
                       << measured_families[replica] << '\t' << flips[replica]
                       << '\t' << magnetization << '\t' << absolute_magnetization
                       << '\t' << magnetization_sq << '\t' << detailed_cap
                       << "\tfirst_replica_indices_at_shell\n";
        ++written;
    }
    files.main.flush();
    files.aggregate.flush();
    files.detailed.flush();
}

int run(int argc, char* argv[]) {
    if (argc < 7 || argc > 9) {
        std::cerr << "Usage: " << argv[0]
                  << " seed L blocks threads nSteps heat"
                     " [checkpoint_dir|none] [checkpoint_every_shells]\n"
                     "  heat: 0=cooling, 1=heating\n";
        return 2;
    }

    const std::uint64_t seed = parse_u64(argv[1], "seed");
    const int L = parse_positive_int(argv[2], "L");
    const int blocks = parse_positive_int(argv[3], "blocks");
    const int threads = parse_positive_int(argv[4], "threads");
    const int sweeps = parse_positive_int(argv[5], "nSteps");
    const int heat_value = parse_nonnegative_int(argv[6], "heat");
    if (L < 3) throw std::invalid_argument("L must be at least 3");
    if (heat_value != 0 && heat_value != 1)
        throw std::invalid_argument("heat must be 0 or 1");
    if (threads > 1024) throw std::invalid_argument("threads must not exceed 1024");
    const std::int64_t replicas_wide
        = static_cast<std::int64_t>(blocks) * threads;
    if (replicas_wide > std::numeric_limits<int>::max())
        throw std::overflow_error("blocks*threads exceeds INT_MAX");
    const int replicas = static_cast<int>(replicas_wide);
    const EngineShape shape = make_engine_shape(L, replicas);
    if (shape.site_count > std::numeric_limits<int>::max() / sweeps)
        throw std::overflow_error("L^2*nSteps exceeds the engine attempt range");

    int detailed_cap = 100;
    if (const char* cap = std::getenv("MCPA_DETAILED_CAP"))
        detailed_cap = parse_nonnegative_int(cap, "MCPA_DETAILED_CAP");
    const char* output_root_text = std::getenv("MCPA_OUTPUT_ROOT");
    if (output_root_text == nullptr || *output_root_text == '\0')
        output_root_text = "./datasets";
    const std::filesystem::path output_directory
        = std::filesystem::path(output_root_text) / "2DIsing";
    const std::string basename
        = std::string("2DIsing") + (heat_value ? "Heating" : "")
        + "_N" + std::to_string(shape.site_count)
        + "_R" + std::to_string(shape.replicas)
        + "_nSteps" + std::to_string(sweeps)
        + "_run" + std::to_string(seed);
    const WalkDirection direction = heat_value ? WalkDirection::heating
                                               : WalkDirection::cooling;
    const bool checkpoint_enabled
        = argc >= 8 && std::string(argv[7]) != "none";
    if (argc == 9 && !checkpoint_enabled)
        throw std::invalid_argument(
            "checkpoint_every_shells requires an enabled checkpoint directory");
    const int checkpoint_every = argc == 9
        ? parse_positive_int(argv[8], "checkpoint_every_shells") : 1;
    const CheckpointIdentity checkpoint_identity{
        L, shape.site_count, shape.replicas, sweeps, seed, direction,
        detailed_cap, sizeof(PhiloxState)};
    CheckpointFiles checkpoint_files{};
    CheckpointLoadResult restored{};
    if (checkpoint_enabled) {
        checkpoint_files = make_checkpoint_files(argv[7], basename);
        if (checkpoint_is_done(checkpoint_files, checkpoint_identity)) {
            std::cout << "checkpoint done marker found; completed run is a no-op\n";
            return 0;
        }
        restored = load_checkpoint(checkpoint_files, checkpoint_identity);
        if (restored.had_candidates && !restored.found)
            throw std::runtime_error("no valid checkpoint generation: "
                                     + restored.diagnostics);
    }

    int device_count = 0;
    require_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
    if (device_count == 0) throw std::runtime_error("no accessible NVIDIA GPU");
    int device_index = 0;
    require_cuda(cudaGetDevice(&device_index), "query active CUDA device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device_index),
                 "query CUDA device properties");
    if (threads > properties.maxThreadsPerBlock)
        throw std::invalid_argument("threads exceeds the active GPU limit");
    if (blocks > properties.maxGridSize[0])
        throw std::invalid_argument("blocks exceeds the active GPU grid limit");

    const std::array<std::uint64_t, 3>* resume_offsets
        = restored.found ? &restored.state.output_offsets : nullptr;
    const auto run_output_paths = output_paths(output_directory, basename);
    OutputFiles files = open_outputs(output_directory, basename, resume_offsets);

    std::cout << "2D Ising " << (heat_value ? "heating" : "cooling")
              << ": L=" << L << " N=" << shape.site_count
              << " R=" << shape.replicas << " nSteps=" << sweeps
              << " detailed_cap=" << detailed_cap
              << (restored.found ? " resumed" : " fresh") << '\n';

    RunGpuMetadata gpu_metadata = collect_gpu_metadata_before_setup(properties);
    DeviceRun device(shape);
    collect_gpu_metadata_after_setup(gpu_metadata);
    if (std::getenv("MCPA_DETERMINISTIC_RUN_METADATA") != nullptr) {
        gpu_metadata = RunGpuMetadata{"TEST_GPU", "0.0", 0, 0, 0, 0, 0};
    }
    bool gpu_metadata_pending = !restored.found;

    std::vector<Spin> spins(static_cast<std::size_t>(shape.site_count)
                            * shape.replicas);
    std::vector<Energy> energies(shape.replicas);
    std::vector<FlipCount> flips(shape.replicas);
    std::vector<ReplicaIndex> order(shape.replicas);
    std::vector<ReplicaIndex> parents(shape.replicas);
    std::vector<FamilyId> families(shape.replicas);
    std::vector<FamilyId> measured_families(shape.replicas);
    const SquareTorus model(L);
    Energy boundary = model.outside_spectrum_sentinel(direction);
    std::uint64_t completed_shells = 0;
    ResamplingRngState resampling_rng = seed_resampling_rng(seed);
    if (restored.found) {
        spins = std::move(restored.state.spins);
        energies = std::move(restored.state.energies);
        families = std::move(restored.state.families);
        order = std::move(restored.state.order);
        boundary = restored.state.boundary;
        completed_shells = restored.state.completed_shells;
        resampling_rng = restored.state.resampling_rng;
        require_cuda(cudaMemcpy(device.engine.population.spins, spins.data(),
                                spin_bytes(shape), cudaMemcpyHostToDevice),
                     "restore checkpoint spins");
        require_cuda(cudaMemcpy(device.engine.population.energies, energies.data(),
                                energy_bytes(shape), cudaMemcpyHostToDevice),
                     "restore checkpoint energies");
        require_cuda(cudaMemcpy(device.engine.philox, restored.state.philox.data(),
                                philox_bytes(shape), cudaMemcpyHostToDevice),
                     "restore checkpoint Philox states");
    } else {
        for (int replica = 0; replica < shape.replicas; ++replica)
            families[replica] = replica;
        require_cuda(initialize_philox(device.engine.philox, shape, seed),
                     "launch Philox initialization");
        require_cuda(initialize_population(device.engine, shape),
                     "launch population initialization");
    }
    require_cuda(cudaDeviceSynchronize(), "initialize or restore population");

    const std::int64_t maximum_shells
        = model.maximum_energy() - model.minimum_energy() + 2;
    std::uint64_t shell_count = completed_shells;
    const bool deterministic_timings
        = std::getenv("MCPA_DETERMINISTIC_TIMINGS") != nullptr;
    int checkpoint_pause_ms = 0;
    if (const char* pause = std::getenv("MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS"))
        checkpoint_pause_ms = parse_nonnegative_int(
            pause, "MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS");

    while (true) {
        if (++shell_count > static_cast<std::uint64_t>(maximum_shells))
            throw std::runtime_error("strict boundary failed to progress");
        const auto shell_start = Clock::now();

        const auto equilibrate_start = Clock::now();
        require_cuda(equilibrate(device.engine, shape, sweeps, boundary, direction),
                     "launch equilibrate");
        require_cuda(cudaDeviceSynchronize(), "equilibrate");
        double equilibrate_seconds = seconds_since(equilibrate_start);

        const auto copy_start = Clock::now();
        require_cuda(cudaMemcpy(spins.data(), device.engine.population.spins,
                                spin_bytes(shape), cudaMemcpyDeviceToHost),
                     "copy spins to host");
        require_cuda(cudaMemcpy(energies.data(), device.engine.population.energies,
                                energy_bytes(shape), cudaMemcpyDeviceToHost),
                     "copy energies to host");
        require_cuda(cudaMemcpy(flips.data(), device.engine.population.flip_counts,
                                flip_count_bytes(shape), cudaMemcpyDeviceToHost),
                     "copy flip counts to host");
        double copy_d2h_seconds = seconds_since(copy_start);
        measured_families = families;

        const auto resample_start = Clock::now();
        const ResampleResult result = resample_strict(
            HostResamplingView{energies.data(), order.data(), parents.data(),
                               families.data(), shape.replicas},
            boundary, direction, resampling_rng);
        double resample_seconds = seconds_since(resample_start);
        if (result.status == ResampleStatus::no_next_shell)
            throw std::runtime_error("no next strict shell before a terminal full cull");

        const ShellStatistics shell = calculate_shell_statistics(
            spins, energies, flips, measured_families, shape, result.boundary);
        const FamilyMetrics population_families
            = calculate_family_metrics(families.data(), shape.replicas);

        double replica_copy_seconds = 0.0;
        if (result.status == ResampleStatus::ok) {
            const auto replica_copy_start = Clock::now();
            require_cuda(cudaMemcpy(device.parents, parents.data(),
                                    static_cast<std::size_t>(shape.replicas)
                                        * sizeof(ReplicaIndex),
                                    cudaMemcpyHostToDevice),
                         "copy parent indices to GPU");
            require_cuda(copy_replicas(device.engine.population, device.parents,
                                       device.workspace, shape),
                         "launch replica copy");
            require_cuda(cudaDeviceSynchronize(), "copy replicas");
            replica_copy_seconds = seconds_since(replica_copy_start);
        }

        double shell_seconds = seconds_since(shell_start);
        if (deterministic_timings) {
            equilibrate_seconds = 0.0;
            copy_d2h_seconds = 0.0;
            resample_seconds = 0.0;
            replica_copy_seconds = 0.0;
            shell_seconds = 0.0;
        }
        write_shell(files, result, population_families, shell, spins, energies,
                    flips, measured_families, shape, sweeps, detailed_cap,
                    equilibrate_seconds, copy_d2h_seconds, resample_seconds,
                    replica_copy_seconds, shell_seconds,
                    gpu_metadata_pending ? &gpu_metadata : nullptr);
        gpu_metadata_pending = false;
        std::cout << "shell=" << result.boundary
                  << " nCull=" << result.culled
                  << " X=" << std::setprecision(17) << result.culling_fraction
                  << " status=" << resample_status_name(result.status) << '\n';
        boundary = result.boundary;
        completed_shells = shell_count;
        if (result.status == ResampleStatus::terminal_full_cull) {
            if (checkpoint_enabled) {
                sync_outputs(files, run_output_paths);
                mark_checkpoint_done(checkpoint_files, checkpoint_identity);
            }
            break;
        }

        if (checkpoint_enabled
            && completed_shells % static_cast<std::uint64_t>(checkpoint_every) == 0) {
            sync_outputs(files, run_output_paths);
            CheckpointState checkpoint{};
            checkpoint.boundary = boundary;
            checkpoint.completed_shells = completed_shells;
            checkpoint.output_offsets = output_offsets(files);
            checkpoint.resampling_rng = resampling_rng;
            checkpoint.spins.resize(spins.size());
            checkpoint.energies.resize(energies.size());
            checkpoint.families = families;
            checkpoint.order = order;
            checkpoint.philox.resize(philox_bytes(shape));
            require_cuda(cudaMemcpy(checkpoint.spins.data(),
                                    device.engine.population.spins,
                                    spin_bytes(shape), cudaMemcpyDeviceToHost),
                         "capture checkpoint spins");
            require_cuda(cudaMemcpy(checkpoint.energies.data(),
                                    device.engine.population.energies,
                                    energy_bytes(shape), cudaMemcpyDeviceToHost),
                         "capture checkpoint energies");
            require_cuda(cudaMemcpy(checkpoint.philox.data(), device.engine.philox,
                                    philox_bytes(shape), cudaMemcpyDeviceToHost),
                         "capture checkpoint Philox states");
            save_checkpoint(checkpoint_files, checkpoint_identity, checkpoint);
            if (checkpoint_pause_ms > 0)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(checkpoint_pause_ms));
        }
    }
    std::cout << "completed with one terminal full-cull row after "
              << shell_count << " shells\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
