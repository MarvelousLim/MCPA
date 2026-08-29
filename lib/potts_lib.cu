#include "potts_lib.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(ans) { gpu_assert((int)(ans), __FILE__, __LINE__); }

namespace {

PottsResamplingRngState resampling_rng{0, 1};

uint32_t next_resampling_random() {
    const uint64_t old_state = resampling_rng.state;
    resampling_rng.state = old_state * 6364136223846793005ULL
                         + (resampling_rng.stream | 1ULL);
    const uint32_t xorshifted = static_cast<uint32_t>(
        ((old_state >> 18U) ^ old_state) >> 27U);
    const uint32_t rotation = static_cast<uint32_t>(old_state >> 59U);
    return (xorshifted >> rotation)
         | (xorshifted << ((-rotation) & 31U));
}

} // namespace

POTTS_HOST_DEVICE struct neighborsIndexes SLF(int j, int L, int N) {
    struct neighborsIndexes result;
    result.right = (j + 1) % L + L * (j / L);
    result.left = (j - 1 + L) % L + L * (j / L);
    result.down = (j + L) % N;
    result.up = (j - L + N) % N;
    return result;
}

static __device__ struct neighborsValues get_neighbors_values(
        const char* spin, struct neighborsIndexes indexes, size_t replica_shift) {
    struct neighborsValues result;
    result.up = spin[indexes.up + replica_shift];
    result.down = spin[indexes.down + replica_shift];
    result.left = spin[indexes.left + replica_shift];
    result.right = spin[indexes.right + replica_shift];
    return result;
}

POTTS_HOST_DEVICE int local_energy(char currentSpin, struct neighborsValues neighbors) {
    return -(currentSpin == neighbors.up)
           -(currentSpin == neighbors.down)
           -(currentSpin == neighbors.left)
           -(currentSpin == neighbors.right);
}

static __device__ int delta_energy(
        char currentSpin, char suggestedSpin, struct neighborsValues neighbors) {
    return local_energy(suggestedSpin, neighbors)
           - local_energy(currentSpin, neighbors);
}

void swap_order(int* order, int i, int j) {
    int temp = order[i];
    order[i] = order[j];
    order[j] = temp;
}

void quicksort(int* E, int* order, int left, int right, int direction) {
    int middle = (left + right) / 2;
    int i = left;
    int j = right;
    int pivot = direction * E[order[middle]];

    while (left < j || i < right) {
        while (direction * E[order[i]] > pivot) i++;
        while (direction * E[order[j]] < pivot) j--;

        if (i <= j) {
            swap_order(order, i, j);
            i++;
            j--;
        } else {
            if (left < j) quicksort(E, order, left, j, direction);
            if (i < right) quicksort(E, order, i, right, direction);
            return;
        }
    }
}

void gpu_assert(int code, const char* file, int line, bool abort) {
    cudaError_t error = (cudaError_t)code;
    if (error != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
        if (abort) exit(code);
    }
}

struct MemEstimate estimate_potts_setup_memory(int N, int R) {
    size_t host_aux = 4 * (size_t)R * sizeof(int)
                    + (size_t)R * sizeof(uint64_t);
    size_t device_aux = 2 * (size_t)R * sizeof(int)
                      + (size_t)R * sizeof(uint64_t);
    size_t device_rng = (size_t)R * sizeof(curandStatePhilox4_32_10_t);
    return estimate_setup_memory(N, R, sizeof(char), host_aux, device_aux, device_rng);
}

static __global__ void setup_curand_kernel(
        curandStatePhilox4_32_10_t* states, int seed, int R) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r < R) curand_init(seed, r, 0, states + r);
}

void* setup_curand_states(struct PottsParams params) {
    curandStatePhilox4_32_10_t* states = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&states,
                          (size_t)params.R * sizeof(curandStatePhilox4_32_10_t)));
    setup_curand_kernel<<<params.blocks, params.threads>>>(states, params.seed, params.R);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    return states;
}

void initialize_resampling_rng(int seed) {
    const uint64_t unsigned_seed = static_cast<uint64_t>(static_cast<uint32_t>(seed));
    resampling_rng.state = 0;
    resampling_rng.stream = (unsigned_seed << 1U) | 1U;
    (void)next_resampling_random();
    resampling_rng.state += unsigned_seed ^ 0x9e3779b97f4a7c15ULL;
    (void)next_resampling_random();
}

struct PottsResamplingRngState get_resampling_rng_state() {
    return resampling_rng;
}

void set_resampling_rng_state(struct PottsResamplingRngState state) {
    state.stream |= 1ULL;
    resampling_rng = state;
}

static __global__ void initialize_population_kernel(
        curandStatePhilox4_32_10_t* states, struct PottsMemoryPointers device,
        struct PottsParams params) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    size_t replica_shift = (size_t)r * (size_t)params.N;
    for (int k = 0; k < params.N; k++)
        device.spin[replica_shift + k] = (char)(curand(&states[r]) % params.q);
}

void initialize_population(void* curand_states, struct PottsMemoryPointers device,
                           struct PottsParams params) {
    initialize_population_kernel<<<params.blocks, params.threads>>>(
        (curandStatePhilox4_32_10_t*)curand_states, device, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}

static __global__ void calc_device_energy_kernel(
        struct PottsMemoryPointers device, struct PottsParams params) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    int sum = 0;
    size_t replica_shift = (size_t)r * (size_t)params.N;
    for (int j = 0; j < params.N; j++) {
        char current = device.spin[replica_shift + j];
        struct neighborsIndexes indexes = SLF(j, params.L, params.N);
        struct neighborsValues neighbors =
            get_neighbors_values(device.spin, indexes, replica_shift);
        sum += local_energy(current, neighbors);
    }
    device.E[r] = sum / 2;
}

void calc_device_energy(struct PottsMemoryPointers device, struct PottsParams params) {
    calc_device_energy_kernel<<<params.blocks, params.threads>>>(device, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}

static __global__ void equilibrate_kernel(
        curandStatePhilox4_32_10_t* states, struct PottsMemoryPointers device,
        struct PottsParams params, int U) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    size_t replica_shift = (size_t)r * (size_t)params.N;
    long long attempts = (long long)params.N * params.nSteps;
    uint64_t accepted = 0;

    for (long long k = 0; k < attempts; k++) {
        int j = curand(&states[r]) % params.N;
        char current = device.spin[replica_shift + j];
        char suggested = (char)(curand(&states[r]) % params.q);
        struct neighborsIndexes indexes = SLF(j, params.L, params.N);
        struct neighborsValues neighbors =
            get_neighbors_values(device.spin, indexes, replica_shift);
        int dE = delta_energy(current, suggested, neighbors);
        if ((!params.heat && device.E[r] + dE < U)
                || (params.heat && device.E[r] + dE > U)) {
            device.E[r] += dE;
            device.spin[replica_shift + j] = suggested;
            ++accepted;
        }
    }
    if (device.accepted_flips) device.accepted_flips[r] = accepted;
}

void equilibrate(void* curand_states, struct PottsMemoryPointers device,
                 struct PottsParams params, int U) {
    equilibrate_kernel<<<params.blocks, params.threads>>>(
        (curandStatePhilox4_32_10_t*)curand_states, device, params, U);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}

static __global__ void update_replicas_kernel(
        struct PottsMemoryPointers device, struct PottsParams params) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    int source_r = device.update[r];
    if (source_r == r) return;

    size_t replica_shift = (size_t)r * (size_t)params.N;
    size_t source_shift = (size_t)source_r * (size_t)params.N;
    for (int j = 0; j < params.N; j++)
        device.spin[replica_shift + j] = device.spin[source_shift + j];
    device.E[r] = device.E[source_r];
}

void update_replicas(struct PottsMemoryPointers device, struct PottsParams params) {
    update_replicas_kernel<<<params.blocks, params.threads>>>(device, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}

PottsResampleResult resample(int* E, int* order, int* update,
                             int* replica_family, int R, int* U, bool heat) {
    PottsResampleResult result{POTTS_RESAMPLE_NO_NEXT_SHELL, *U, *U, 0, 0.0};
    quicksort(E, order, 0, R - 1, 1 - 2 * heat);

    int old_U = *U;
    int new_U = old_U;
    for (int i = 0; i < R; i++) {
        int candidate = E[order[i]];
        if ((!heat && candidate < old_U) || (heat && candidate > old_U)) {
            new_U = candidate;
            break;
        }
    }
    if (new_U == old_U) return result;

    int nCull = 0;
    while (nCull < R
            && ((!heat && E[order[nCull]] >= new_U)
                || (heat && E[order[nCull]] <= new_U)))
        nCull++;

    *U = new_U;
    result.new_U = new_U;
    result.n_cull = nCull;
    result.culling_fraction = static_cast<double>(nCull) / R;

    for (int i = 0; i < R; i++) update[i] = i;
    if (nCull == R) {
        result.status = POTTS_RESAMPLE_TERMINAL_FULL_CULL;
        return result;
    }

    for (int i = 0; i < nCull; i++) {
        const uint32_t draw = next_resampling_random();
        int source = (int)(draw % (uint32_t)(R - nCull)) + nCull;
        update[order[i]] = order[source];
        replica_family[order[i]] = replica_family[order[source]];
    }
    result.status = POTTS_RESAMPLE_OK;
    return result;
}
