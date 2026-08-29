#include <doctest/doctest.h>

#include "baxterwu_lib.h"

#include <cuda_runtime.h>

#include <array>
#include <vector>

namespace {

void require_diagnostic_cuda(cudaError_t status) {
    INFO(cudaGetErrorString(status));
    REQUIRE(status == cudaSuccess);
}

Params fourier_params() {
    Params params{};
    params.L = 6;
    params.N = 36;
    params.R = 4;
    params.threads = 32;
    params.blocks = 1;
    params.singleIntRowByteSize = static_cast<size_t>(params.R) * sizeof(int);
    params.fullLatticeByteSize = static_cast<size_t>(params.R) * params.N * sizeof(int);
    params.replicaStatisticsByteSize = static_cast<size_t>(params.R) * sizeof(replicaStatistics);
    return params;
}

} // namespace

TEST_CASE("Structure factors are branch invariant and measurement only") {
    const Params params = fourier_params();
    const std::array<std::array<int, 3>, 4> branches{{
        {{1, 1, 1}}, {{1, -1, -1}}, {{-1, 1, -1}}, {{-1, -1, 1}}
    }};
    std::vector<int> spins(static_cast<size_t>(params.R) * params.N);
    for (int r = 0; r < params.R; ++r) {
        for (int j = 0; j < params.N; ++j) {
            const int x = j % params.L;
            const int y = j / params.L;
            spins[static_cast<size_t>(r) * params.N + j] = branches[r][(x + y) % 3];
        }
        // The same local defect in all four symmetry-related branches leaves
        // each minimum-wavevector modulus square unchanged.
        spins[static_cast<size_t>(r) * params.N] *= -1;
    }

    mainMemoryPointers device{};
    require_diagnostic_cuda(cudaMalloc(&device.spin, params.fullLatticeByteSize));
    require_diagnostic_cuda(cudaMalloc(&device.E, params.singleIntRowByteSize));
    require_diagnostic_cuda(cudaMalloc(&device.replica_statistics,
                                       params.replicaStatisticsByteSize));
    const size_t phase_bytes = static_cast<size_t>(3) * params.N * sizeof(float);
    require_diagnostic_cuda(cudaMalloc(&device.fourier_phase_cos, phase_bytes));
    require_diagnostic_cuda(cudaMalloc(&device.fourier_phase_sin, phase_bytes));
    require_diagnostic_cuda(cudaMemcpy(device.spin, spins.data(), params.fullLatticeByteSize,
                                       cudaMemcpyHostToDevice));

    initialize_fourier_phases(device, params);
    calc_device_energy(device, params);
    std::array<int, 4> energy_before{};
    require_diagnostic_cuda(cudaMemcpy(energy_before.data(), device.E,
                                       params.singleIntRowByteSize,
                                       cudaMemcpyDeviceToHost));
    for (int r = 1; r < params.R; ++r) CHECK(energy_before[r] == energy_before[0]);

    calc_replica_statistics(device, params, energy_before[0]);
    std::vector<int> spins_after(spins.size());
    std::array<int, 4> energy_after{};
    std::array<replicaStatistics, 4> statistics{};
    require_diagnostic_cuda(cudaMemcpy(spins_after.data(), device.spin,
                                       params.fullLatticeByteSize,
                                       cudaMemcpyDeviceToHost));
    require_diagnostic_cuda(cudaMemcpy(energy_after.data(), device.E,
                                       params.singleIntRowByteSize,
                                       cudaMemcpyDeviceToHost));
    require_diagnostic_cuda(cudaMemcpy(statistics.data(), device.replica_statistics,
                                       params.replicaStatisticsByteSize,
                                       cudaMemcpyDeviceToHost));

    CHECK(spins_after == spins);
    CHECK(energy_after == energy_before);
    for (int r = 0; r < params.R; ++r) {
        for (int q = 0; q < 3; ++q) {
            CHECK(statistics[r].order_structure_factor[q]
                  == doctest::Approx(4.0 / params.N).epsilon(2e-5));
            CHECK(statistics[r].order_structure_factor[q]
                  == doctest::Approx(statistics[0].order_structure_factor[q]).epsilon(1e-6));
        }
    }

    cudaFree(device.fourier_phase_sin);
    cudaFree(device.fourier_phase_cos);
    cudaFree(device.replica_statistics);
    cudaFree(device.E);
    cudaFree(device.spin);
}
