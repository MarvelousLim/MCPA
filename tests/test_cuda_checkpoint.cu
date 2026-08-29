#include <doctest/doctest.h>

#include "baxterwu_lib.h"
#include "checkpoint.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

void require_checkpoint_cuda(cudaError_t status) {
    INFO(cudaGetErrorString(status));
    REQUIRE(status == cudaSuccess);
}

Params checkpoint_params(bool heat) {
    Params params{};
    params.L = 6;
    params.N = params.L * params.L;
    params.R = 64;
    params.seed = 457;
    params.blocks = 2;
    params.threads = 32;
    params.nSteps = 2;
    params.heat = heat;
    params.fullLatticeByteSize = static_cast<size_t>(params.R) * params.N * sizeof(int);
    params.singleIntRowByteSize = static_cast<size_t>(params.R) * sizeof(int);
    params.replicaStatisticsByteSize = static_cast<size_t>(params.R)
                                     * sizeof(replicaStatistics);
    return params;
}

struct CheckpointPopulation {
    Params params;
    mainMemoryPointers host{};
    mainMemoryPointers device{};
    std::vector<int> spins;
    std::vector<int> energies;
    std::vector<int> order;
    std::vector<int> update;
    std::vector<int> families;
    void* philox = nullptr;

    explicit CheckpointPopulation(const Params& value, bool initialize)
        : params(value),
          spins(static_cast<size_t>(params.R) * params.N),
          energies(params.R), order(params.R), update(params.R), families(params.R) {
        host.spin = spins.data();
        host.E = energies.data();
        host.O = order.data();
        host.update = update.data();
        host.replica_family = families.data();

        require_checkpoint_cuda(cudaMalloc(&device.spin, params.fullLatticeByteSize));
        require_checkpoint_cuda(cudaMalloc(&device.E, params.singleIntRowByteSize));
        require_checkpoint_cuda(cudaMalloc(&device.update, params.singleIntRowByteSize));
        require_checkpoint_cuda(cudaMalloc(&device.replica_statistics,
                                           params.replicaStatisticsByteSize));
        philox = setup_curand_states(params);
        initialize_update_arrays(host, params);
        if (initialize) {
            initialize_resampling_rng(params.seed);
            initialize_population(philox, device, params, random_pop);
            calc_device_energy(device, params);
        }
    }

    ~CheckpointPopulation() {
        cudaFree(philox);
        cudaFree(device.replica_statistics);
        cudaFree(device.update);
        cudaFree(device.E);
        cudaFree(device.spin);
    }

    CheckpointPopulation(const CheckpointPopulation&) = delete;
    CheckpointPopulation& operator=(const CheckpointPopulation&) = delete;
};

struct TrajectorySnapshot {
    std::vector<int> spins;
    std::vector<int> energies;
    std::vector<int> families;
    std::vector<int> order;
    std::vector<unsigned char> philox;
    ResamplingRngState resampling{};
    int U = 0;
};

TrajectorySnapshot snapshot(CheckpointPopulation& population, int U) {
    TrajectorySnapshot result;
    result.spins.resize(population.spins.size());
    result.energies.resize(population.params.R);
    result.families = population.families;
    result.order = population.order;
    result.philox.resize(curand_states_byte_size(population.params));
    require_checkpoint_cuda(cudaMemcpy(result.spins.data(), population.device.spin,
                                       population.params.fullLatticeByteSize,
                                       cudaMemcpyDeviceToHost));
    require_checkpoint_cuda(cudaMemcpy(result.energies.data(), population.device.E,
                                       population.params.singleIntRowByteSize,
                                       cudaMemcpyDeviceToHost));
    require_checkpoint_cuda(cudaMemcpy(result.philox.data(), population.philox,
                                       result.philox.size(), cudaMemcpyDeviceToHost));
    result.resampling = get_resampling_rng_state();
    result.U = U;
    return result;
}

void advance_shell(CheckpointPopulation& population, int& U) {
    equilibrate(population.philox, population.device, population.params, U);
    require_checkpoint_cuda(cudaMemcpy(population.energies.data(), population.device.E,
                                       population.params.singleIntRowByteSize,
                                       cudaMemcpyDeviceToHost));
    int culled = 0;
    const double fraction = prepare_resample_arrays(
        population.host, population.params, &U, &culled);
    CAPTURE(population.params.heat);
    CAPTURE(U);
    CAPTURE(culled);
    REQUIRE(fraction < 1.0);
    require_checkpoint_cuda(cudaMemcpy(population.device.update,
                                       population.update.data(),
                                       population.params.singleIntRowByteSize,
                                       cudaMemcpyHostToDevice));
    update_replicas(population.device, population.params);
}

void check_snapshot_equal(const TrajectorySnapshot& actual,
                          const TrajectorySnapshot& expected) {
    CHECK(actual.spins == expected.spins);
    CHECK(actual.energies == expected.energies);
    CHECK(actual.families == expected.families);
    CHECK(actual.order == expected.order);
    CHECK(actual.philox == expected.philox);
    CHECK(actual.resampling.state == expected.resampling.state);
    CHECK(actual.resampling.stream == expected.resampling.stream);
    CHECK(actual.U == expected.U);
}

struct TemporaryCudaCheckpointDirectory {
    std::string path;

    TemporaryCudaCheckpointDirectory() {
        std::array<char, 64> pattern{};
        std::snprintf(pattern.data(), pattern.size(),
                      "/tmp/mcpa-cuda-checkpoint.XXXXXX");
        char* created = mkdtemp(pattern.data());
        REQUIRE(created != nullptr);
        path = created;
    }

    ~TemporaryCudaCheckpointDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void check_checkpoint_trajectory(bool heat) {
    const Params params = checkpoint_params(heat);
    constexpr int total_shells = 5;
    constexpr int checkpoint_shell = 2;

    CheckpointPopulation uninterrupted(params, true);
    int uninterrupted_U = heat ? -2 * params.N - 2 : 2 * params.N + 2;
    for (int step = 0; step < total_shells; ++step)
        advance_shell(uninterrupted, uninterrupted_U);
    const TrajectorySnapshot expected = snapshot(uninterrupted, uninterrupted_U);

    CheckpointPopulation interrupted(params, true);
    int interrupted_U = heat ? -2 * params.N - 2 : 2 * params.N + 2;
    for (int step = 0; step < checkpoint_shell; ++step)
        advance_shell(interrupted, interrupted_U);
    const TrajectorySnapshot at_checkpoint = snapshot(interrupted, interrupted_U);

    TemporaryCudaCheckpointDirectory directory;
    CheckpointManager manager{};
    checkpoint_init(manager, params.L, params.N, params.R, params.nSteps,
                    params.seed, 0.0f, static_cast<int>(params.heat),
                    directory.path.c_str(), true, 3600,
                    heat ? "bw_heating_trajectory" : "bw_cooling_trajectory");
    manager.step_count = checkpoint_shell;
    const std::array<int64_t, 3> positions{{111, 222, 333}};
    REQUIRE(checkpoint_save(
        manager, params.L, params.N, params.R, params.nSteps, params.seed,
        0.0f, static_cast<int>(params.heat), at_checkpoint.spins.data(),
        at_checkpoint.energies.data(), at_checkpoint.families.data(),
        at_checkpoint.order.data(), at_checkpoint.U,
        at_checkpoint.philox.data(), at_checkpoint.philox.size(),
        at_checkpoint.resampling.state, at_checkpoint.resampling.stream,
        positions.data()));

    CheckpointPopulation resumed(params, false);
    std::vector<unsigned char> loaded_philox(curand_states_byte_size(params));
    int resumed_U = 0;
    int64_t loaded_step = -1;
    uint64_t loaded_resampling_state = 0;
    uint64_t loaded_resampling_stream = 1;
    std::array<int64_t, 3> loaded_positions{{-1, -1, -1}};
    REQUIRE(checkpoint_load(
        manager, params.L, params.N, params.R, params.nSteps, params.seed,
        0.0f, static_cast<int>(params.heat), resumed.spins.data(),
        resumed.energies.data(), resumed.families.data(), resumed.order.data(),
        resumed_U, loaded_step, loaded_philox.data(), loaded_philox.size(),
        loaded_resampling_state, loaded_resampling_stream,
        loaded_positions.data()));
    CHECK(loaded_step == checkpoint_shell);
    CHECK(loaded_positions == positions);

    require_checkpoint_cuda(cudaMemcpy(resumed.device.spin, resumed.spins.data(),
                                       params.fullLatticeByteSize,
                                       cudaMemcpyHostToDevice));
    require_checkpoint_cuda(cudaMemcpy(resumed.device.E, resumed.energies.data(),
                                       params.singleIntRowByteSize,
                                       cudaMemcpyHostToDevice));
    require_checkpoint_cuda(cudaMemcpy(resumed.philox, loaded_philox.data(),
                                       loaded_philox.size(), cudaMemcpyHostToDevice));
    set_resampling_rng_state(
        ResamplingRngState{loaded_resampling_state, loaded_resampling_stream});

    const TrajectorySnapshot loaded = snapshot(resumed, resumed_U);
    check_snapshot_equal(loaded, at_checkpoint);
    for (int step = checkpoint_shell; step < total_shells; ++step)
        advance_shell(resumed, resumed_U);
    check_snapshot_equal(snapshot(resumed, resumed_U), expected);
}

} // namespace

TEST_CASE("Cooling checkpoint resumes the exact physical and RNG trajectory") {
    check_checkpoint_trajectory(false);
}

TEST_CASE("Heating checkpoint resumes the exact physical and RNG trajectory") {
    check_checkpoint_trajectory(true);
}
