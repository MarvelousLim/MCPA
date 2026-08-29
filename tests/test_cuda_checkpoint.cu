#include "test_harness.h"

#include "potts_checkpoint.h"
#include "potts_lib.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using PhiloxState = curandStatePhilox4_32_10_t;

void cuda_require(cudaError_t result) {
    POTTS_REQUIRE(result == cudaSuccess);
}

struct TempDirectory {
    std::string path;
    TempDirectory() {
        std::array<char, 64> pattern{};
        std::snprintf(pattern.data(), pattern.size(), "/tmp/mcpa-potts-gpu-checkpoint.XXXXXX");
        char* created = ::mkdtemp(pattern.data());
        POTTS_REQUIRE(created != nullptr);
        path = created;
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct Population {
    PottsParams params{};
    PottsMemoryPointers device{};
    void* philox = nullptr;
    std::vector<int> energies;
    std::vector<int> update;
    std::vector<int> family;
    std::vector<int> order;
    int U = potts_cooling_start_U();

    explicit Population(int q)
        : energies(32), update(32), family(32), order(32) {
        params.seed = 1900 + q;
        params.L = 3;
        params.N = 9;
        params.blocks = 1;
        params.threads = 32;
        params.R = 32;
        params.nSteps = 2;
        params.q = q;
        params.heat = false;
        params.fullLatticeByteSize = static_cast<size_t>(params.N) * params.R;
        params.singleIntRowByteSize = static_cast<size_t>(params.R) * sizeof(int);
        cuda_require(cudaMalloc(&device.spin, params.fullLatticeByteSize));
        cuda_require(cudaMalloc(&device.E, params.singleIntRowByteSize));
        cuda_require(cudaMalloc(&device.update, params.singleIntRowByteSize));
        for (int r = 0; r < params.R; ++r) family[r] = order[r] = r;
        philox = setup_curand_states(params);
        initialize_resampling_rng(params.seed);
        initialize_population(philox, device, params);
        calc_device_energy(device, params);
    }

    ~Population() {
        cudaFree(philox);
        cudaFree(device.update);
        cudaFree(device.E);
        cudaFree(device.spin);
    }

    void advance_shell() {
        equilibrate(philox, device, params, U);
        cuda_require(cudaMemcpy(energies.data(), device.E,
                                params.singleIntRowByteSize,
                                cudaMemcpyDeviceToHost));
        const PottsResampleResult result = resample(
            energies.data(), order.data(), update.data(), family.data(),
            params.R, &U, false);
        POTTS_REQUIRE(result.status == POTTS_RESAMPLE_OK);
        cuda_require(cudaMemcpy(device.update, update.data(),
                                params.singleIntRowByteSize,
                                cudaMemcpyHostToDevice));
        update_replicas(device, params);
    }

    PottsCheckpointIdentity identity() const {
        return {params.L, params.N, params.R, params.nSteps, params.seed,
                params.q, params.heat, 100,
                static_cast<size_t>(params.R) * sizeof(PhiloxState)};
    }

    PottsCheckpointState capture(uint64_t shells) const {
        PottsCheckpointState state;
        state.U = U;
        state.completed_shells = shells;
        state.resampling_rng = get_resampling_rng_state();
        state.spins.resize(params.fullLatticeByteSize);
        state.energies.resize(params.R);
        state.families = family;
        state.order = order;
        state.philox.resize(identity().philox_bytes);
        cuda_require(cudaMemcpy(state.spins.data(), device.spin,
                                params.fullLatticeByteSize,
                                cudaMemcpyDeviceToHost));
        cuda_require(cudaMemcpy(state.energies.data(), device.E,
                                params.singleIntRowByteSize,
                                cudaMemcpyDeviceToHost));
        cuda_require(cudaMemcpy(state.philox.data(), philox,
                                state.philox.size(), cudaMemcpyDeviceToHost));
        return state;
    }

    void restore(const PottsCheckpointState& state) {
        U = state.U;
        family = state.families;
        order = state.order;
        set_resampling_rng_state(state.resampling_rng);
        cuda_require(cudaMemcpy(device.spin, state.spins.data(),
                                params.fullLatticeByteSize,
                                cudaMemcpyHostToDevice));
        cuda_require(cudaMemcpy(device.E, state.energies.data(),
                                params.singleIntRowByteSize,
                                cudaMemcpyHostToDevice));
        cuda_require(cudaMemcpy(philox, state.philox.data(), state.philox.size(),
                                cudaMemcpyHostToDevice));
    }
};

void require_equal(const PottsCheckpointState& a,
                   const PottsCheckpointState& b) {
    POTTS_REQUIRE(a.U == b.U);
    POTTS_REQUIRE(a.completed_shells == b.completed_shells);
    POTTS_REQUIRE(a.resampling_rng.state == b.resampling_rng.state);
    POTTS_REQUIRE(a.resampling_rng.stream == b.resampling_rng.stream);
    POTTS_REQUIRE(a.spins == b.spins);
    POTTS_REQUIRE(a.energies == b.energies);
    POTTS_REQUIRE(a.families == b.families);
    POTTS_REQUIRE(a.order == b.order);
    POTTS_REQUIRE(a.philox.size() == b.philox.size());
    POTTS_REQUIRE(std::memcmp(a.philox.data(), b.philox.data(), a.philox.size()) == 0);
}

} // namespace

POTTS_TEST_CASE("CUDA Potts q=2,3,4 checkpoint continuations are bitwise exact") {
    TempDirectory directory;
    for (int q : {2, 3, 4}) {
        Population population(q);
        population.advance_shell();
        const PottsCheckpointState boundary = population.capture(1);
        const std::string base = directory.path + "/q" + std::to_string(q);
        POTTS_REQUIRE(save_potts_checkpoint(base, population.identity(), boundary));

        population.advance_shell();
        const PottsCheckpointState uninterrupted = population.capture(2);

        PottsCheckpointState restored;
        POTTS_REQUIRE(load_potts_checkpoint(base, population.identity(), &restored)
                      == PottsCheckpointLoadStatus::loaded);
        population.restore(restored);
        population.advance_shell();
        const PottsCheckpointState continued = population.capture(2);
        require_equal(continued, uninterrupted);
    }
}
