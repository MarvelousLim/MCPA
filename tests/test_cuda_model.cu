#include "test_harness.h"

#include "blumeCapel_lib.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

void require_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

Params make_params(bool heat = false) {
    Params params{};
    params.L = 3;
    params.N = 9;
    params.R = 32;
    params.seed = 271;
    params.blocks = 1;
    params.threads = 32;
    params.nSteps = 2;
    params.D_num = 49;
    params.D_denum = 25;
    params.heat = heat;
    params.fullLatticeByteSize = static_cast<std::size_t>(params.R) * params.N * sizeof(int);
    params.singleIntRowByteSize = static_cast<std::size_t>(params.R) * sizeof(int);
    params.replicaStatisticsByteSize = static_cast<std::size_t>(params.R)
                                     * sizeof(replicaStatistics);
    return params;
}

struct DevicePopulation {
    mainMemoryPointers memory{};

    explicit DevicePopulation(const Params& params) {
        require_cuda(cudaMalloc(&memory.spin, params.fullLatticeByteSize));
        require_cuda(cudaMalloc(&memory.e_j, params.singleIntRowByteSize));
        require_cuda(cudaMalloc(&memory.e_delta, params.singleIntRowByteSize));
        require_cuda(cudaMalloc(&memory.replica_statistics,
                                params.replicaStatisticsByteSize));
        require_cuda(cudaMemset(memory.replica_statistics, 0,
                                params.replicaStatisticsByteSize));
    }

    ~DevicePopulation() {
        cudaFree(memory.spin);
        cudaFree(memory.e_j);
        cudaFree(memory.e_delta);
        cudaFree(memory.replica_statistics);
    }

    DevicePopulation(const DevicePopulation&) = delete;
    DevicePopulation& operator=(const DevicePopulation&) = delete;
};

struct HostEnergy {
    int e_j = 0;
    int e_delta = 0;
};

HostEnergy independent_energy(const int* spins, int L) {
    HostEnergy energy{};
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int j = x + y * L;
            const int right = (x + 1) % L + y * L;
            const int down = x + ((y + 1) % L) * L;
            energy.e_j -= spins[j] * (spins[right] + spins[down]);
            energy.e_delta += spins[j] * spins[j];
        }
    }
    return energy;
}

std::vector<int> deterministic_population(const Params& params) {
    std::vector<int> spins(static_cast<std::size_t>(params.R) * params.N);
    for (int r = 0; r < params.R; ++r) {
        for (int y = 0; y < params.L; ++y) {
            for (int x = 0; x < params.L; ++x) {
                const int j = x + y * params.L;
                spins[static_cast<std::size_t>(r) * params.N + j]
                    = (x + 2 * y + r) % 3 - 1;
            }
        }
    }
    return spins;
}

void upload_population(DevicePopulation& population, const Params& params,
                       const std::vector<int>& spins,
                       std::vector<int>* e_j_out = nullptr,
                       std::vector<int>* e_delta_out = nullptr) {
    std::vector<int> e_j(params.R);
    std::vector<int> e_delta(params.R);
    for (int r = 0; r < params.R; ++r) {
        const HostEnergy energy = independent_energy(
            spins.data() + static_cast<std::size_t>(r) * params.N, params.L);
        e_j[r] = energy.e_j;
        e_delta[r] = energy.e_delta;
    }
    require_cuda(cudaMemcpy(population.memory.spin, spins.data(),
                            params.fullLatticeByteSize, cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(population.memory.e_j, e_j.data(),
                            params.singleIntRowByteSize, cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(population.memory.e_delta, e_delta.data(),
                            params.singleIntRowByteSize, cudaMemcpyHostToDevice));
    if (e_j_out) *e_j_out = e_j;
    if (e_delta_out) *e_delta_out = e_delta;
}

__global__ void setup_reference_states(curandStatePhilox4_32_10_t* states,
                                       int seed, int R) {
    const int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r < R) curand_init(seed, r, 0, &states[r]);
}

__device__ int reference_local_j(const int* spins, long long shift, int j,
                                 int sigma, int L) {
    const int x = j % L;
    const int y = j / L;
    const int left = (x - 1 + L) % L + y * L;
    const int right = (x + 1) % L + y * L;
    const int up = x + ((y - 1 + L) % L) * L;
    const int down = x + ((y + 1) % L) * L;
    return -sigma * (spins[shift + left] + spins[shift + right]
                    + spins[shift + up] + spins[shift + down]);
}

__global__ void reference_equilibrate(curandStatePhilox4_32_10_t* states,
                                      mainMemoryPointers device,
                                      Params params, int U) {
    const int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    const long long shift = static_cast<long long>(r) * params.N;
    int flips = 0;
    for (int k = 0; k < params.N * params.nSteps; ++k) {
        const int j = static_cast<int>(curand(&states[r])
                                    % static_cast<unsigned int>(params.N));
        const int old_spin = device.spin[shift + j];
        const unsigned int proposal_draw = curand(&states[r]);
        const int old_index = old_spin + 1;
        const int new_spin = (old_index + 1 + static_cast<int>(proposal_draw & 1u)) % 3 - 1;
        const int delta_j = reference_local_j(device.spin, shift, j, new_spin, params.L)
                          - reference_local_j(device.spin, shift, j, old_spin, params.L);
        const int delta_d = new_spin * new_spin - old_spin * old_spin;
        const int new_e_j = device.e_j[r] + delta_j;
        const int new_e_delta = device.e_delta[r] + delta_d;
        const int new_energy = params.D_denum * new_e_j + params.D_num * new_e_delta;
        const bool accepted = params.heat ? new_energy > U : new_energy < U;
        if (accepted) {
            device.spin[shift + j] = new_spin;
            device.e_j[r] = new_e_j;
            device.e_delta[r] = new_e_delta;
            ++flips;
        }
    }
    device.replica_statistics[r].flip_count = flips;
}

void check_population_energy(const std::vector<int>& spins,
                             const std::vector<int>& e_j,
                             const std::vector<int>& e_delta,
                             const Params& params) {
    for (int r = 0; r < params.R; ++r) {
        const int* replica = spins.data() + static_cast<std::size_t>(r) * params.N;
        const HostEnergy exact = independent_energy(replica, params.L);
        BC_REQUIRE(e_j[r] == exact.e_j);
        BC_REQUIRE(e_delta[r] == exact.e_delta);
        for (int j = 0; j < params.N; ++j) {
            BC_REQUIRE(replica[j] >= -1);
            BC_REQUIRE(replica[j] <= 1);
        }
    }
}

void check_trajectory(bool heat, bool open) {
    const Params params = make_params(heat);
    const std::vector<int> initial = deterministic_population(params);
    DevicePopulation candidate(params);
    DevicePopulation reference(params);
    upload_population(candidate, params, initial);
    upload_population(reference, params, initial);

    curandStatePhilox4_32_10_t* candidate_states = nullptr;
    curandStatePhilox4_32_10_t* reference_states = nullptr;
    const std::size_t state_bytes = static_cast<std::size_t>(params.R)
                                  * sizeof(curandStatePhilox4_32_10_t);
    require_cuda(cudaMalloc(&candidate_states, state_bytes));
    require_cuda(cudaMalloc(&reference_states, state_bytes));
    setup_reference_states<<<params.blocks, params.threads>>>(
        candidate_states, params.seed, params.R);
    require_cuda(cudaDeviceSynchronize());
    require_cuda(cudaMemcpy(reference_states, candidate_states, state_bytes,
                            cudaMemcpyDeviceToDevice));

    const int low = -4096;
    const int high = 4096;
    const int U = heat ? (open ? low : high) : (open ? high : low);
    equilibrate(candidate_states, candidate.memory, params, U);
    reference_equilibrate<<<params.blocks, params.threads>>>(
        reference_states, reference.memory, params, U);
    require_cuda(cudaDeviceSynchronize());
    require_cuda(cudaGetLastError());

    std::vector<int> candidate_spins(static_cast<std::size_t>(params.R) * params.N);
    std::vector<int> reference_spins(candidate_spins.size());
    std::vector<int> candidate_e_j(params.R), reference_e_j(params.R);
    std::vector<int> candidate_e_delta(params.R), reference_e_delta(params.R);
    std::vector<replicaStatistics> candidate_stats(params.R), reference_stats(params.R);
    std::vector<unsigned char> candidate_rng(state_bytes), reference_rng(state_bytes);

    require_cuda(cudaMemcpy(candidate_spins.data(), candidate.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_spins.data(), reference.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_e_j.data(), candidate.memory.e_j,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_e_j.data(), reference.memory.e_j,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_e_delta.data(), candidate.memory.e_delta,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_e_delta.data(), reference.memory.e_delta,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_stats.data(), candidate.memory.replica_statistics,
                            params.replicaStatisticsByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_stats.data(), reference.memory.replica_statistics,
                            params.replicaStatisticsByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_rng.data(), candidate_states, state_bytes,
                            cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_rng.data(), reference_states, state_bytes,
                            cudaMemcpyDeviceToHost));

    BC_REQUIRE(candidate_spins == reference_spins);
    BC_REQUIRE(candidate_e_j == reference_e_j);
    BC_REQUIRE(candidate_e_delta == reference_e_delta);
    BC_REQUIRE(std::memcmp(candidate_stats.data(), reference_stats.data(),
                           params.replicaStatisticsByteSize) == 0);
    BC_REQUIRE(candidate_rng == reference_rng);
    check_population_energy(candidate_spins, candidate_e_j, candidate_e_delta, params);

    long long accepted = 0;
    for (const replicaStatistics& stats : candidate_stats) accepted += stats.flip_count;
    const long long attempts = static_cast<long long>(params.R) * params.N * params.nSteps;
    BC_REQUIRE(open ? accepted == attempts : accepted == 0);

    cudaFree(candidate_states);
    cudaFree(reference_states);
}

} // namespace

BC_TEST_CASE("Initialized GPU population has valid spins and exact energy parts") {
    const Params params = make_params();
    DevicePopulation population(params);
    curandStatePhilox4_32_10_t* states = nullptr;
    const std::size_t state_bytes = static_cast<std::size_t>(params.R)
                                  * sizeof(curandStatePhilox4_32_10_t);
    require_cuda(cudaMalloc(&states, state_bytes));
    setup_reference_states<<<params.blocks, params.threads>>>(states, params.seed, params.R);
    require_cuda(cudaDeviceSynchronize());
    initialize_population(states, population.memory, params);
    calc_device_energy(population.memory, params);

    std::vector<int> spins(static_cast<std::size_t>(params.R) * params.N);
    std::vector<int> e_j(params.R), e_delta(params.R);
    require_cuda(cudaMemcpy(spins.data(), population.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(e_j.data(), population.memory.e_j,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(e_delta.data(), population.memory.e_delta,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    check_population_energy(spins, e_j, e_delta, params);
    cudaFree(states);
}

BC_TEST_CASE("Cooling open trajectory and Philox state match the reference") {
    check_trajectory(false, true);
}

BC_TEST_CASE("Cooling frozen trajectory and Philox state match the reference") {
    check_trajectory(false, false);
}

BC_TEST_CASE("Heating open trajectory and Philox state match the reference") {
    check_trajectory(true, true);
}

BC_TEST_CASE("Heating frozen trajectory and Philox state match the reference") {
    check_trajectory(true, false);
}
