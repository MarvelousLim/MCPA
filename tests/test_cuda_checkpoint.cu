#include "checkpoint_1d.h"
#include "legacy_1d_ising_api.cuh"
#include "test_harness.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

void cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

struct TempDirectory {
    std::string path;
    TempDirectory() {
        std::array<char, 64> pattern{};
        std::snprintf(pattern.data(), pattern.size(), "/tmp/mcpa-1d-gpu-checkpoint.XXXXXX");
        char* created = ::mkdtemp(pattern.data());
        ISING_REQUIRE(created != nullptr);
        path = created;
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct DevicePopulation {
    static constexpr int N = 5;
    static constexpr int R = 16;
    static constexpr int n_steps = 2;
    static constexpr int seed = 1709;

    char* spins = nullptr;
    int* energies = nullptr;
    int* updates = nullptr;
    int* flips = nullptr;
    curandStatePhilox4_32_10_t* philox = nullptr;
    std::vector<int> host_energy = std::vector<int>(R);
    std::vector<int> family = std::vector<int>(R);
    std::vector<int> order = std::vector<int>(R);
    std::vector<int> update = std::vector<int>(R);
    int U = N + 2;

    DevicePopulation() {
        cuda_ok(cudaMalloc(&spins, N * R * sizeof(char)), "allocate spins");
        cuda_ok(cudaMalloc(&energies, R * sizeof(int)), "allocate energies");
        cuda_ok(cudaMalloc(&updates, R * sizeof(int)), "allocate updates");
        cuda_ok(cudaMalloc(&flips, R * sizeof(int)), "allocate flips");
        cuda_ok(cudaMalloc(&philox, R * sizeof(*philox)), "allocate Philox");
        for (int r = 0; r < R; ++r) family[r] = order[r] = r;
        setup_kernel<<<1, R>>>(philox, seed);
        initializePopulation<<<1, R>>>(philox, spins, N, 2);
        cuda_ok(cudaMemset(energies, 0, R * sizeof(int)), "clear energy");
        deviceEnergy<<<1, R>>>(spins, energies, N, N);
        cuda_ok(cudaDeviceSynchronize(), "initialize population");
        initializeResamplingRng(seed);
    }

    ~DevicePopulation() {
        cudaFree(philox);
        cudaFree(flips);
        cudaFree(updates);
        cudaFree(energies);
        cudaFree(spins);
    }

    void advance_shell() {
        equilibrate<<<1, R>>>(philox, spins, energies, N, N, R, 2, n_steps, U,
                              flips);
        cuda_ok(cudaMemcpy(host_energy.data(), energies, R * sizeof(int),
                           cudaMemcpyDeviceToHost), "copy equilibrated energy");
        const IsingResampleResult result = resample(
            host_energy.data(), order.data(), update.data(), family.data(),
            R, &U, false);
        ISING_REQUIRE(result.status == IsingResampleStatus::ok);
        cuda_ok(cudaMemcpy(updates, update.data(), R * sizeof(int),
                           cudaMemcpyHostToDevice), "copy parent map");
        updateReplicas<<<1, R>>>(spins, energies, updates, N);
        cuda_ok(cudaDeviceSynchronize(), "copy replicas");
    }

    Ising1DCheckpointState capture(std::uint64_t shells) const {
        Ising1DCheckpointState state;
        state.U = U;
        state.completed_shells = shells;
        state.resampling_rng = getResamplingRngState();
        state.spins.resize(N * R);
        state.energies.resize(R);
        state.families = family;
        state.order = order;
        state.philox.resize(R * sizeof(*philox));
        cuda_ok(cudaMemcpy(state.spins.data(), spins, state.spins.size(),
                           cudaMemcpyDeviceToHost), "capture spins");
        cuda_ok(cudaMemcpy(state.energies.data(), energies, R * sizeof(int),
                           cudaMemcpyDeviceToHost), "capture energy");
        cuda_ok(cudaMemcpy(state.philox.data(), philox, state.philox.size(),
                           cudaMemcpyDeviceToHost), "capture Philox");
        return state;
    }

    void restore(const Ising1DCheckpointState& state) {
        U = state.U;
        family = state.families;
        order = state.order;
        setResamplingRngState(state.resampling_rng);
        cuda_ok(cudaMemcpy(spins, state.spins.data(), state.spins.size(),
                           cudaMemcpyHostToDevice), "restore spins");
        cuda_ok(cudaMemcpy(energies, state.energies.data(), R * sizeof(int),
                           cudaMemcpyHostToDevice), "restore energy");
        cuda_ok(cudaMemcpy(philox, state.philox.data(), state.philox.size(),
                           cudaMemcpyHostToDevice), "restore Philox");
    }
};

void require_same(const Ising1DCheckpointState& a,
                  const Ising1DCheckpointState& b) {
    ISING_REQUIRE(a.U == b.U);
    ISING_REQUIRE(a.completed_shells == b.completed_shells);
    ISING_REQUIRE(a.resampling_rng.state == b.resampling_rng.state);
    ISING_REQUIRE(a.resampling_rng.stream == b.resampling_rng.stream);
    ISING_REQUIRE(a.spins == b.spins);
    ISING_REQUIRE(a.energies == b.energies);
    ISING_REQUIRE(a.families == b.families);
    ISING_REQUIRE(a.order == b.order);
    ISING_REQUIRE(a.philox.size() == b.philox.size());
    ISING_REQUIRE(std::memcmp(a.philox.data(), b.philox.data(), a.philox.size()) == 0);
}

} // namespace

ISING_TEST_CASE("CUDA checkpoint continuation is bitwise equal after a copied shell") {
    TempDirectory directory;
    DevicePopulation population;
    population.advance_shell();
    const Ising1DCheckpointState boundary = population.capture(1);
    const Ising1DCheckpointIdentity identity{
        DevicePopulation::N, DevicePopulation::R, DevicePopulation::n_steps,
        DevicePopulation::seed, 100, boundary.philox.size()};
    const std::string base = directory.path + "/trajectory";
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity, boundary));

    population.advance_shell();
    const Ising1DCheckpointState uninterrupted = population.capture(2);

    Ising1DCheckpointState restored;
    ISING_REQUIRE(load_ising1d_checkpoint(base, identity, &restored)
                  == Ising1DCheckpointLoadStatus::loaded);
    population.restore(restored);
    population.advance_shell();
    const Ising1DCheckpointState continued = population.capture(2);
    require_same(continued, uninterrupted);
}
