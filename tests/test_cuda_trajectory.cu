#include "legacy_1d_ising_api.cuh"
#include "test_harness.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

void require_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

__global__ void reference_initialize(curandStatePhilox4_32_10_t* states,
                                     char* spins, int N) {
    const int replica = threadIdx.x + blockIdx.x * blockDim.x;
    for (int site = 0; site < N; ++site)
        spins[replica * N + site] = 2 * (curand(&states[replica]) % 2) - 1;
}

__global__ void reference_equilibrate(curandStatePhilox4_32_10_t* states,
                                      char* spins, int* energies,
                                      int N, int nSteps, int ceiling,
                                      int* flip_counts) {
    const int replica = threadIdx.x + blockIdx.x * blockDim.x;
    const int shift = replica * N;
    flip_counts[replica] = 0;
    for (int attempt = 0; attempt < N * nSteps; ++attempt) {
        const int site = curand(&states[replica]) % N;
        const int left = (site - 1 + N) % N;
        const int right = (site + 1) % N;
        const char old_spin = spins[shift + site];
        const int delta = 2 * old_spin
                        * (spins[shift + left] + spins[shift + right]);
        const int candidate = energies[replica] + delta;
        if (candidate < ceiling) {
            energies[replica] = candidate;
            spins[shift + site] = -old_spin;
            ++flip_counts[replica];
        }
    }
}

int independent_energy(const char* spins, int N) {
    int energy = 0;
    for (int site = 0; site < N; ++site)
        energy -= spins[site] * spins[(site + 1) % N];
    return energy;
}

class TrajectoryFixture {
public:
    TrajectoryFixture(int N, int replicas, int seed)
        : N_(N), replicas_(replicas), seed_(seed),
          spin_bytes_(static_cast<std::size_t>(N) * replicas * sizeof(char)),
          energy_bytes_(static_cast<std::size_t>(replicas) * sizeof(int)),
          rng_bytes_(static_cast<std::size_t>(replicas)
                     * sizeof(curandStatePhilox4_32_10_t)) {
        require_cuda(cudaMalloc(&production_spins_, spin_bytes_), "cudaMalloc production spins");
        require_cuda(cudaMalloc(&reference_spins_, spin_bytes_), "cudaMalloc reference spins");
        require_cuda(cudaMalloc(&production_energy_, energy_bytes_), "cudaMalloc production energy");
        require_cuda(cudaMalloc(&reference_energy_, energy_bytes_), "cudaMalloc reference energy");
        require_cuda(cudaMalloc(&production_rng_, rng_bytes_), "cudaMalloc production RNG");
        require_cuda(cudaMalloc(&reference_rng_, rng_bytes_), "cudaMalloc reference RNG");
        require_cuda(cudaMalloc(&production_flips_, energy_bytes_), "cudaMalloc production flips");
        require_cuda(cudaMalloc(&reference_flips_, energy_bytes_), "cudaMalloc reference flips");
    }

    ~TrajectoryFixture() {
        cudaFree(reference_flips_);
        cudaFree(production_flips_);
        cudaFree(reference_rng_);
        cudaFree(production_rng_);
        cudaFree(reference_energy_);
        cudaFree(production_energy_);
        cudaFree(reference_spins_);
        cudaFree(production_spins_);
    }

    void initialize_and_compare() {
        setup_kernel<<<1, replicas_>>>(production_rng_, seed_);
        require_launch("setup_kernel");
        require_cuda(cudaMemcpy(reference_rng_, production_rng_, rng_bytes_,
                                cudaMemcpyDeviceToDevice), "copy starting RNG");

        initializePopulation<<<1, replicas_>>>(production_rng_, production_spins_, N_, 2);
        reference_initialize<<<1, replicas_>>>(reference_rng_, reference_spins_, N_);
        require_launch("initialization kernels");
        compare_spins();
        compare_rng();

        deviceEnergy<<<1, replicas_>>>(production_spins_, production_energy_, N_, N_);
        require_launch("deviceEnergy");
        verify_tracked_energy();
        require_cuda(cudaMemcpy(reference_energy_, production_energy_, energy_bytes_,
                                cudaMemcpyDeviceToDevice), "copy starting energy");
    }

    void run_and_compare(int nSteps, int ceiling) {
        equilibrate<<<1, replicas_>>>(production_rng_, production_spins_, production_energy_,
                                     N_, N_, replicas_, 2, nSteps, ceiling,
                                     production_flips_);
        reference_equilibrate<<<1, replicas_>>>(reference_rng_, reference_spins_,
                                               reference_energy_, N_, nSteps, ceiling,
                                               reference_flips_);
        require_launch("equilibration kernels");
        compare_spins();
        compare_energy_arrays();
        compare_rng();
        ISING_REQUIRE(copy_energies(production_flips_)
                      == copy_energies(reference_flips_));
        verify_tracked_energy();
    }

private:
    void require_launch(const char* operation) {
        require_cuda(cudaPeekAtLastError(), operation);
        require_cuda(cudaDeviceSynchronize(), operation);
    }

    std::vector<char> copy_spins(char* source) const {
        std::vector<char> result(spin_bytes_);
        require_cuda(cudaMemcpy(result.data(), source, spin_bytes_, cudaMemcpyDeviceToHost),
                     "copy spins to host");
        return result;
    }

    std::vector<int> copy_energies(int* source) const {
        std::vector<int> result(static_cast<std::size_t>(replicas_));
        require_cuda(cudaMemcpy(result.data(), source, energy_bytes_, cudaMemcpyDeviceToHost),
                     "copy energies to host");
        return result;
    }

    std::vector<unsigned char> copy_rng(curandStatePhilox4_32_10_t* source) const {
        std::vector<unsigned char> result(rng_bytes_);
        require_cuda(cudaMemcpy(result.data(), source, rng_bytes_, cudaMemcpyDeviceToHost),
                     "copy RNG to host");
        return result;
    }

    void compare_spins() const {
        const std::vector<char> production = copy_spins(production_spins_);
        const std::vector<char> reference = copy_spins(reference_spins_);
        ISING_REQUIRE(production == reference);
        for (char spin : production) ISING_REQUIRE(spin == -1 || spin == 1);
    }

    void compare_energy_arrays() const {
        ISING_REQUIRE(copy_energies(production_energy_)
                      == copy_energies(reference_energy_));
    }

    void compare_rng() const {
        const std::vector<unsigned char> production = copy_rng(production_rng_);
        const std::vector<unsigned char> reference = copy_rng(reference_rng_);
        ISING_REQUIRE(production.size() == reference.size());
        ISING_REQUIRE(std::memcmp(production.data(), reference.data(), production.size()) == 0);
    }

    void verify_tracked_energy() const {
        const std::vector<char> spins = copy_spins(production_spins_);
        const std::vector<int> energies = copy_energies(production_energy_);
        for (int replica = 0; replica < replicas_; ++replica)
            ISING_REQUIRE(energies[replica]
                          == independent_energy(spins.data() + replica * N_, N_));
    }

    int N_;
    int replicas_;
    int seed_;
    std::size_t spin_bytes_;
    std::size_t energy_bytes_;
    std::size_t rng_bytes_;
    char* production_spins_ = nullptr;
    char* reference_spins_ = nullptr;
    int* production_energy_ = nullptr;
    int* reference_energy_ = nullptr;
    curandStatePhilox4_32_10_t* production_rng_ = nullptr;
    curandStatePhilox4_32_10_t* reference_rng_ = nullptr;
    int* production_flips_ = nullptr;
    int* reference_flips_ = nullptr;
};

} // namespace

ISING_TEST_CASE("CUDA initialization stays in domain with exact tracked energy") {
    for (int N : {5, 6}) {
        TrajectoryFixture fixture(N, 8, 731 + N);
        fixture.initialize_and_compare();
    }
}

ISING_TEST_CASE("Even N=6 spins energies flips and Philox match the reference") {
    TrajectoryFixture fixture(6, 8, 1701);
    fixture.initialize_and_compare();
    fixture.run_and_compare(4, 0);
}

ISING_TEST_CASE("Odd N=5 spins energies flips and Philox match the reference") {
    TrajectoryFixture fixture(5, 8, 1702);
    fixture.initialize_and_compare();
    fixture.run_and_compare(4, 0);
}
