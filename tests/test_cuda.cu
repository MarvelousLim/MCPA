#include <doctest/doctest.h>

#include "baxterwu_lib.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

void require_cuda(cudaError_t status) {
    INFO(cudaGetErrorString(status));
    REQUIRE(status == cudaSuccess);
}

Params make_params(int L, int R, int n_steps, bool heat = false) {
    Params params{};
    params.L = L;
    params.N = L * L;
    params.R = R;
    params.threads = 32;
    params.blocks = (R + params.threads - 1) / params.threads;
    params.nSteps = n_steps;
    params.seed = 173;
    params.heat = heat;
    params.singleIntRowByteSize = static_cast<size_t>(R) * sizeof(int);
    params.fullLatticeByteSize = static_cast<size_t>(R) * params.N * sizeof(int);
    params.replicaStatisticsByteSize = static_cast<size_t>(R) * sizeof(replicaStatistics);
    return params;
}

struct DevicePopulation {
    mainMemoryPointers memory{};

    explicit DevicePopulation(const Params& params) {
        require_cuda(cudaMalloc(&memory.spin, params.fullLatticeByteSize));
        require_cuda(cudaMalloc(&memory.E, params.singleIntRowByteSize));
        require_cuda(cudaMalloc(&memory.replica_statistics, params.replicaStatisticsByteSize));
        require_cuda(cudaMemset(memory.replica_statistics, 0, params.replicaStatisticsByteSize));
    }

    ~DevicePopulation() {
        cudaFree(memory.spin);
        cudaFree(memory.E);
        cudaFree(memory.replica_statistics);
    }

    DevicePopulation(const DevicePopulation&) = delete;
    DevicePopulation& operator=(const DevicePopulation&) = delete;
};

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

__global__ void reference_equilibrate_kernel(
    curandStatePhilox4_32_10_t* states,
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

std::vector<int> striped_population(const Params& params,
                                    const std::array<int, 3>& branch) {
    std::vector<int> spins(static_cast<size_t>(params.R) * params.N);
    for (int r = 0; r < params.R; ++r) {
        for (int j = 0; j < params.N; ++j) {
            const int x = j % params.L;
            const int y = j / params.L;
            spins[static_cast<size_t>(r) * params.N + j] = branch[(x + y) % 3];
        }
    }
    return spins;
}

std::vector<int> mixed_population(const Params& params) {
    std::vector<int> spins = striped_population(params, {{1, 1, 1}});
    // Isolated defects one lattice spacing-three cell apart.  No triangle
    // contains two defects, so the fixture has both increasing- and
    // decreasing-energy proposals without depending on the production loop.
    for (int r = 0; r < params.R; ++r) {
        if (params.heat) {
            if (r % 2 != 0) {
                const size_t first = static_cast<size_t>(r) * params.N;
                for (int j = 0; j < params.N; ++j) spins[first + j] *= -1;
            }
            continue;
        }
        for (int y = 0; y < params.L; y += 3) {
            for (int x = 0; x < params.L; x += 3) {
                const size_t index = static_cast<size_t>(r) * params.N
                                   + y * params.L + x;
                spins[index] = -spins[index];
            }
        }
    }
    return spins;
}

enum class AcceptanceRegime { open, frozen, mixed };

const char* regime_name(AcceptanceRegime regime) {
    switch (regime) {
        case AcceptanceRegime::open: return "open";
        case AcceptanceRegime::frozen: return "frozen";
        case AcceptanceRegime::mixed: return "mixed";
    }
    return "unknown";
}

} // namespace

TEST_CASE("GPU memory allocation and round-trip transfer succeed") {
    const Params params = make_params(3, 32, 1);
    DevicePopulation population(params);
    std::vector<int> sent(static_cast<size_t>(params.R) * params.N);
    std::iota(sent.begin(), sent.end(), -static_cast<int>(sent.size() / 2));
    std::vector<int> received(sent.size());
    require_cuda(cudaMemcpy(population.memory.spin, sent.data(), params.fullLatticeByteSize,
                            cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(received.data(), population.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    CHECK(received == sent);
}

TEST_CASE("Energy kernel recognizes all four ground-state branches") {
    const Params params = make_params(6, 32, 1);
    const std::array<std::array<int, 3>, 4> branches{{
        {{1, 1, 1}}, {{1, -1, -1}}, {{-1, 1, -1}}, {{-1, -1, 1}}
    }};

    DevicePopulation population(params);
    for (const auto& branch : branches) {
        const std::vector<int> spins = striped_population(params, branch);
        require_cuda(cudaMemcpy(population.memory.spin, spins.data(), params.fullLatticeByteSize,
                                cudaMemcpyHostToDevice));
        calc_device_energy(population.memory, params);
        std::vector<int> energies(params.R);
        require_cuda(cudaMemcpy(energies.data(), population.memory.E,
                                params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
        CAPTURE(branch[0]);
        CAPTURE(branch[1]);
        CAPTURE(branch[2]);
        for (const int energy : energies) CHECK(energy == -2 * params.N);
    }
}

namespace {

void check_equilibration_equivalence(bool heat, AcceptanceRegime regime) {
    Params params = make_params(6, 64, 2, heat);
    const std::vector<int> initial = regime == AcceptanceRegime::mixed
        ? mixed_population(params)
        : striped_population(params, {{1, 1, 1}});
    DevicePopulation candidate(params);
    DevicePopulation reference(params);
    require_cuda(cudaMemcpy(candidate.memory.spin, initial.data(),
                            params.fullLatticeByteSize, cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(reference.memory.spin, initial.data(),
                            params.fullLatticeByteSize, cudaMemcpyHostToDevice));
    calc_device_energy(candidate.memory, params);
    calc_device_energy(reference.memory, params);

    std::vector<int> initial_energy(params.R);
    require_cuda(cudaMemcpy(initial_energy.data(), candidate.memory.E,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    const int mixed_energy = -2 * params.N / 3;
    int U = 0;
    if (regime == AcceptanceRegime::open) {
        U = heat ? -2 * params.N - 2 : 2 * params.N + 2;
    } else if (regime == AcceptanceRegime::frozen) {
        U = heat ? 2 * params.N - 2 : -2 * params.N + 8;
    } else {
        for (int r = 0; r < params.R; ++r) {
            const int expected = heat
                ? (r % 2 == 0 ? -2 * params.N : 2 * params.N)
                : mixed_energy;
            CHECK(initial_energy[r] == expected);
        }
        U = heat ? 0 : mixed_energy + 1;
    }

    void* candidate_states = setup_curand_states(params);
    curandStatePhilox4_32_10_t* reference_states = nullptr;
    const size_t state_bytes = static_cast<size_t>(params.R)
                             * sizeof(curandStatePhilox4_32_10_t);
    require_cuda(cudaMalloc(&reference_states, state_bytes));
    require_cuda(cudaMemcpy(reference_states, candidate_states, state_bytes,
                            cudaMemcpyDeviceToDevice));

    equilibrate(candidate_states, candidate.memory, params, U);
    reference_equilibrate_kernel<<<params.blocks, params.threads>>>(
        reference_states, reference.memory, params, U);
    require_cuda(cudaDeviceSynchronize());

    std::vector<int> candidate_spins(static_cast<size_t>(params.R) * params.N);
    std::vector<int> reference_spins(candidate_spins.size());
    std::vector<int> candidate_energy(params.R);
    std::vector<int> reference_energy(params.R);
    std::vector<replicaStatistics> candidate_stats(params.R);
    std::vector<replicaStatistics> reference_stats(params.R);
    std::vector<unsigned char> candidate_rng(state_bytes);
    std::vector<unsigned char> reference_rng(state_bytes);

    require_cuda(cudaMemcpy(candidate_spins.data(), candidate.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_spins.data(), reference.memory.spin,
                            params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_energy.data(), candidate.memory.E,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_energy.data(), reference.memory.E,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_stats.data(), candidate.memory.replica_statistics,
                            params.replicaStatisticsByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_stats.data(), reference.memory.replica_statistics,
                            params.replicaStatisticsByteSize, cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(candidate_rng.data(), candidate_states, state_bytes,
                            cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(reference_rng.data(), reference_states, state_bytes,
                            cudaMemcpyDeviceToHost));

    CAPTURE(heat);
    CAPTURE(regime_name(regime));
    CHECK(candidate_spins == reference_spins);
    CHECK(candidate_energy == reference_energy);
    CHECK(std::memcmp(candidate_stats.data(), reference_stats.data(),
                      params.replicaStatisticsByteSize) == 0);
    CHECK(candidate_rng == reference_rng);

    const long long accepted = std::accumulate(
        candidate_stats.begin(), candidate_stats.end(), 0LL,
        [](long long sum, const replicaStatistics& stats) {
            return sum + stats.flip_count;
        });
    const long long attempts = static_cast<long long>(params.R)
                             * params.N * params.nSteps;
    if (regime == AcceptanceRegime::open) CHECK(accepted == attempts);
    else if (regime == AcceptanceRegime::frozen) CHECK(accepted == 0);
    else {
        CHECK(accepted * 20 >= attempts);
        CHECK(accepted * 20 <= attempts * 19);
    }

    cudaFree(candidate_states);
    cudaFree(reference_states);
}

void check_partial_curand4_tail(bool heat) {
        // L=3 and nSteps=1 give nine attempts: two full curand4 groups plus
        // one used value from a third group.  The remaining three values must
        // still advance the stored Philox state exactly as in the reference.
        Params params = make_params(3, 64, 1, heat);
        const std::vector<int> initial = striped_population(params, {{1, 1, 1}});
        DevicePopulation candidate(params);
        DevicePopulation reference(params);
        require_cuda(cudaMemcpy(candidate.memory.spin, initial.data(),
                                params.fullLatticeByteSize, cudaMemcpyHostToDevice));
        require_cuda(cudaMemcpy(reference.memory.spin, initial.data(),
                                params.fullLatticeByteSize, cudaMemcpyHostToDevice));
        calc_device_energy(candidate.memory, params);
        calc_device_energy(reference.memory, params);

        void* candidate_states = setup_curand_states(params);
        curandStatePhilox4_32_10_t* reference_states = nullptr;
        const size_t state_bytes = static_cast<size_t>(params.R)
                                 * sizeof(curandStatePhilox4_32_10_t);
        require_cuda(cudaMalloc(&reference_states, state_bytes));
        require_cuda(cudaMemcpy(reference_states, candidate_states, state_bytes,
                                cudaMemcpyDeviceToDevice));

        const int U = heat ? -2 * params.N - 2 : 2 * params.N + 2;
        equilibrate(candidate_states, candidate.memory, params, U);
        reference_equilibrate_kernel<<<params.blocks, params.threads>>>(
            reference_states, reference.memory, params, U);
        require_cuda(cudaDeviceSynchronize());

        std::vector<int> candidate_spins(static_cast<size_t>(params.R) * params.N);
        std::vector<int> reference_spins(candidate_spins.size());
        std::vector<int> candidate_energy(params.R);
        std::vector<int> reference_energy(params.R);
        std::vector<replicaStatistics> candidate_stats(params.R);
        std::vector<replicaStatistics> reference_stats(params.R);
        std::vector<unsigned char> candidate_rng(state_bytes);
        std::vector<unsigned char> reference_rng(state_bytes);

        require_cuda(cudaMemcpy(candidate_spins.data(), candidate.memory.spin,
                                params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(reference_spins.data(), reference.memory.spin,
                                params.fullLatticeByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(candidate_energy.data(), candidate.memory.E,
                                params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(reference_energy.data(), reference.memory.E,
                                params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(candidate_stats.data(), candidate.memory.replica_statistics,
                                params.replicaStatisticsByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(reference_stats.data(), reference.memory.replica_statistics,
                                params.replicaStatisticsByteSize, cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(candidate_rng.data(), candidate_states, state_bytes,
                                cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(reference_rng.data(), reference_states, state_bytes,
                                cudaMemcpyDeviceToHost));

        CAPTURE(heat);
        CHECK(candidate_spins == reference_spins);
        CHECK(candidate_energy == reference_energy);
        CHECK(std::memcmp(candidate_stats.data(), reference_stats.data(),
                          params.replicaStatisticsByteSize) == 0);
        CHECK(candidate_rng == reference_rng);

        cudaFree(candidate_states);
        cudaFree(reference_states);
}

} // namespace

TEST_CASE("Cooling open trajectory is bitwise equal to the reference") {
    check_equilibration_equivalence(false, AcceptanceRegime::open);
}

TEST_CASE("Cooling frozen trajectory is bitwise equal to the reference") {
    check_equilibration_equivalence(false, AcceptanceRegime::frozen);
}

TEST_CASE("Cooling mixed trajectory is bitwise equal to the reference") {
    check_equilibration_equivalence(false, AcceptanceRegime::mixed);
}

TEST_CASE("Heating open trajectory is bitwise equal to the reference") {
    check_equilibration_equivalence(true, AcceptanceRegime::open);
}

TEST_CASE("Heating frozen trajectory is bitwise equal to the reference") {
    check_equilibration_equivalence(true, AcceptanceRegime::frozen);
}

TEST_CASE("Heating mixed trajectory is bitwise equal to the reference") {
    check_equilibration_equivalence(true, AcceptanceRegime::mixed);
}

TEST_CASE("Cooling partial curand4 tail preserves Philox consumption") {
    check_partial_curand4_tail(false);
}

TEST_CASE("Heating partial curand4 tail preserves Philox consumption") {
    check_partial_curand4_tail(true);
}

TEST_CASE("Tracked energy equals a full GPU recomputation") {
    Params params = make_params(6, 64, 3, false);
    DevicePopulation population(params);
    const std::vector<int> initial = striped_population(params, {{1, 1, 1}});
    require_cuda(cudaMemcpy(population.memory.spin, initial.data(), params.fullLatticeByteSize,
                            cudaMemcpyHostToDevice));
    calc_device_energy(population.memory, params);
    void* states = setup_curand_states(params);
    equilibrate(states, population.memory, params, -2 * params.N + 12);

    std::vector<int> tracked(params.R);
    require_cuda(cudaMemcpy(tracked.data(), population.memory.E,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    calc_device_energy(population.memory, params);
    std::vector<int> recomputed(params.R);
    require_cuda(cudaMemcpy(recomputed.data(), population.memory.E,
                            params.singleIntRowByteSize, cudaMemcpyDeviceToHost));
    CHECK(tracked == recomputed);
    cudaFree(states);
}
