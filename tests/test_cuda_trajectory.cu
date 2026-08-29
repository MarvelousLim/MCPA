#include "test_harness.h"

#include "potts_lib.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

using PhiloxState = curandStatePhilox4_32_10_t;

void require_cuda(cudaError_t status) {
    POTTS_REQUIRE(status == cudaSuccess);
}

PottsParams make_params(int q, bool heat, int n_steps, int seed) {
    PottsParams params{};
    params.seed = seed;
    params.L = 3;
    params.N = 9;
    params.R = 32;
    params.blocks = 1;
    params.threads = 32;
    params.nSteps = n_steps;
    params.q = q;
    params.heat = heat;
    params.fullLatticeByteSize = static_cast<size_t>(params.R) * params.N;
    params.singleIntRowByteSize = static_cast<size_t>(params.R) * sizeof(int);
    return params;
}

struct DevicePopulation {
    PottsMemoryPointers memory{};

    explicit DevicePopulation(const PottsParams& params) {
        require_cuda(cudaMalloc(&memory.spin, params.fullLatticeByteSize));
        require_cuda(cudaMalloc(&memory.E, params.singleIntRowByteSize));
        require_cuda(cudaMalloc(&memory.accepted_flips,
                                static_cast<size_t>(params.R) * sizeof(uint64_t)));
    }

    ~DevicePopulation() {
        cudaFree(memory.accepted_flips);
        cudaFree(memory.E);
        cudaFree(memory.spin);
    }

    DevicePopulation(const DevicePopulation&) = delete;
    DevicePopulation& operator=(const DevicePopulation&) = delete;
};

int independent_energy(const char* spins, int L) {
    int energy = 0;
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = x + y * L;
            const int right = ((x + 1) % L) + y * L;
            const int down = x + ((y + 1) % L) * L;
            energy -= spins[site] == spins[right];
            energy -= spins[site] == spins[down];
        }
    }
    return energy;
}

std::vector<int> independent_population_energy(const std::vector<char>& spins,
                                               const PottsParams& params) {
    std::vector<int> energies(params.R);
    for (int r = 0; r < params.R; ++r) {
        energies[r] = independent_energy(
            spins.data() + static_cast<size_t>(r) * params.N, params.L);
    }
    return energies;
}

std::vector<char> deterministic_population(const PottsParams& params) {
    std::vector<char> spins(static_cast<size_t>(params.R) * params.N);
    for (int r = 0; r < params.R; ++r) {
        for (int site = 0; site < params.N; ++site) {
            const int x = site % params.L;
            const int y = site / params.L;
            spins[static_cast<size_t>(r) * params.N + site] =
                static_cast<char>((7 * r + 5 * x + 3 * y + x * y) % params.q);
        }
    }
    return spins;
}

__device__ int reference_local_energy(const char* spins, long long shift,
                                      int site, int L, char proposed) {
    const int x = site % L;
    const int y = site / L;
    const int left = (x - 1 + L) % L + y * L;
    const int right = (x + 1) % L + y * L;
    const int up = x + ((y - 1 + L) % L) * L;
    const int down = x + ((y + 1) % L) * L;
    return -(proposed == spins[shift + left])
           -(proposed == spins[shift + right])
           -(proposed == spins[shift + up])
           -(proposed == spins[shift + down]);
}

__global__ void reference_initialize_kernel(PhiloxState* states, char* spins,
                                            PottsParams params) {
    const int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    const long long shift = static_cast<long long>(r) * params.N;
    for (int site = 0; site < params.N; ++site)
        spins[shift + site] = static_cast<char>(curand(&states[r]) % params.q);
}

__global__ void reference_equilibrate_kernel(PhiloxState* states,
                                             PottsMemoryPointers device,
                                             PottsParams params, int U,
                                             uint64_t* accepted) {
    const int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    const long long shift = static_cast<long long>(r) * params.N;
    uint64_t count = 0;
    const long long attempts = static_cast<long long>(params.N) * params.nSteps;
    for (long long attempt = 0; attempt < attempts; ++attempt) {
        const int site = static_cast<int>(curand(&states[r]) % params.N);
        const char old_spin = device.spin[shift + site];
        const char proposed = static_cast<char>(curand(&states[r]) % params.q);
        const int delta = reference_local_energy(
                              device.spin, shift, site, params.L, proposed)
                        - reference_local_energy(
                              device.spin, shift, site, params.L, old_spin);
        const int proposed_energy = device.E[r] + delta;
        if ((!params.heat && proposed_energy < U)
            || (params.heat && proposed_energy > U)) {
            device.E[r] = proposed_energy;
            device.spin[shift + site] = proposed;
            ++count;
        }
    }
    accepted[r] = count;
}

std::vector<unsigned char> copy_rng_bytes(const void* states,
                                          const PottsParams& params) {
    const size_t bytes = static_cast<size_t>(params.R) * sizeof(PhiloxState);
    std::vector<unsigned char> result(bytes);
    require_cuda(cudaMemcpy(result.data(), states, bytes, cudaMemcpyDeviceToHost));
    return result;
}

void compare_trajectory(const PottsParams& params, int U,
                        int expected_accepted) {
    const std::vector<char> initial = deterministic_population(params);
    const std::vector<int> initial_energy =
        independent_population_energy(initial, params);
    DevicePopulation candidate(params);
    DevicePopulation reference(params);
    require_cuda(cudaMemcpy(candidate.memory.spin, initial.data(),
                            params.fullLatticeByteSize, cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(reference.memory.spin, initial.data(),
                            params.fullLatticeByteSize, cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(candidate.memory.E, initial_energy.data(),
                            params.singleIntRowByteSize, cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(reference.memory.E, initial_energy.data(),
                            params.singleIntRowByteSize, cudaMemcpyHostToDevice));

    void* candidate_states = setup_curand_states(params);
    PhiloxState* reference_states = nullptr;
    const size_t state_bytes = static_cast<size_t>(params.R) * sizeof(PhiloxState);
    require_cuda(cudaMalloc(&reference_states, state_bytes));
    require_cuda(cudaMemcpy(reference_states, candidate_states, state_bytes,
                            cudaMemcpyDeviceToDevice));
    uint64_t* device_accepted = nullptr;
    const size_t flip_bytes = static_cast<size_t>(params.R) * sizeof(uint64_t);
    require_cuda(cudaMalloc(&device_accepted, flip_bytes));

    equilibrate(candidate_states, candidate.memory, params, U);
    reference_equilibrate_kernel<<<params.blocks, params.threads>>>(
        reference_states, reference.memory, params, U, device_accepted);
    require_cuda(cudaPeekAtLastError());
    require_cuda(cudaDeviceSynchronize());

    std::vector<char> candidate_spins(initial.size());
    std::vector<char> reference_spins(initial.size());
    std::vector<int> candidate_energy(params.R);
    std::vector<int> reference_energy(params.R);
    std::vector<uint64_t> candidate_accepted(params.R);
    std::vector<uint64_t> accepted(params.R);
    require_cuda(cudaMemcpy(candidate_spins.data(), candidate.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_spins.data(), reference.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_energy.data(), candidate.memory.E,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_energy.data(), reference.memory.E,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_accepted.data(),
                            candidate.memory.accepted_flips,
                            flip_bytes, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(accepted.data(), device_accepted,
                            flip_bytes, cudaMemcpyDeviceToHost));

    POTTS_REQUIRE(candidate_spins == reference_spins);
    POTTS_REQUIRE(candidate_energy == reference_energy);
    POTTS_REQUIRE(candidate_accepted == accepted);
    POTTS_REQUIRE(copy_rng_bytes(candidate_states, params)
                  == copy_rng_bytes(reference_states, params));
    POTTS_REQUIRE(independent_population_energy(candidate_spins, params)
                  == candidate_energy);

    const uint64_t accepted_total = std::accumulate(
        accepted.begin(), accepted.end(), uint64_t{0});
    const uint64_t total_attempts = static_cast<uint64_t>(params.R)
                                  * params.N * params.nSteps;
    if (expected_accepted == 0) POTTS_REQUIRE(accepted_total == 0);
    else if (static_cast<uint64_t>(expected_accepted) == total_attempts)
        POTTS_REQUIRE(accepted_total == total_attempts);
    else {
        POTTS_REQUIRE(accepted_total > 0);
        POTTS_REQUIRE(accepted_total < total_attempts);
    }

    cudaFree(device_accepted);
    cudaFree(reference_states);
    cudaFree(candidate_states);
}

} // namespace

POTTS_TEST_CASE("CUDA initialization and tracked energies match independent oracles") {
    for (int q : {2, 3, 4}) {
        const PottsParams params = make_params(q, false, 1, 100 + q);
        DevicePopulation candidate(params);
        DevicePopulation reference(params);
        void* candidate_states = setup_curand_states(params);
        PhiloxState* reference_states = nullptr;
        const size_t state_bytes = static_cast<size_t>(params.R) * sizeof(PhiloxState);
        require_cuda(cudaMalloc(&reference_states, state_bytes));
        require_cuda(cudaMemcpy(reference_states, candidate_states, state_bytes,
                                cudaMemcpyDeviceToDevice));

        initialize_population(candidate_states, candidate.memory, params);
        reference_initialize_kernel<<<params.blocks, params.threads>>>(
            reference_states, reference.memory.spin, params);
        require_cuda(cudaPeekAtLastError());
        require_cuda(cudaDeviceSynchronize());
        calc_device_energy(candidate.memory, params);

        std::vector<char> candidate_spins(
            static_cast<size_t>(params.R) * params.N);
        std::vector<char> reference_spins(candidate_spins.size());
        std::vector<int> tracked(params.R);
        require_cuda(cudaMemcpy(candidate_spins.data(), candidate.memory.spin,
                                params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(reference_spins.data(), reference.memory.spin,
                                params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(tracked.data(), candidate.memory.E,
                                params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
        POTTS_REQUIRE(candidate_spins == reference_spins);
        POTTS_REQUIRE(copy_rng_bytes(candidate_states, params)
                      == copy_rng_bytes(reference_states, params));
        POTTS_REQUIRE(std::all_of(candidate_spins.begin(), candidate_spins.end(),
                                  [q](char spin) { return spin >= 0 && spin < q; }));
        POTTS_REQUIRE(independent_population_energy(candidate_spins, params) == tracked);

        cudaFree(reference_states);
        cudaFree(candidate_states);
    }
}

POTTS_TEST_CASE("CUDA cooling and heating open and frozen trajectories are bitwise exact") {
    for (int q : {2, 3, 4}) {
        for (bool heat : {false, true}) {
            const PottsParams params = make_params(q, heat, 2, 200 + 10 * q + heat);
            const int attempts = params.R * params.N * params.nSteps;
            const int open_U = heat ? -2 * params.N - 1 : 1;
            const int frozen_U = heat ? 0 : -2 * params.N;
            compare_trajectory(params, open_U, attempts);
            compare_trajectory(params, frozen_U, 0);
        }
    }
}

POTTS_TEST_CASE("CUDA mixed cooling and heating trajectories are bitwise exact") {
    for (int q : {2, 3, 4}) {
        for (bool heat : {false, true}) {
            const PottsParams params = make_params(
                q, heat, 2, 300 + 10 * q + heat);
            const int initial_energy = q == 2 ? -10 : (q == 3 ? -6 : -4);
            compare_trajectory(params, initial_energy + (heat ? -1 : 1), -1);
        }
    }
}
