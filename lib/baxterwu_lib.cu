#include "baxterwu_lib.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>


#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

namespace {

ResamplingRngState resampling_rng{0, 1};

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

DECLSPEC bool between(float x, float a, float b) {
    return (x <= b && x >= a) || (x >= b && x <= a);
}

DECLSPEC void gpu_assert(int code, const char* file, int line, bool abort) {
    cudaError_t cudaCode = (cudaError_t)code;
    if (cudaCode != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(cudaCode), file, line);
        if (abort) exit(code);
    }
}

DECLSPEC void print_spin_sample(int* s, int r, struct Params params) {
    long long arrayIndex = (long long)r * params.N;
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            printf("%2d ", s[arrayIndex]);
            arrayIndex++;
        }
        printf("\n");
    }
    printf("\n");
};

DECLSPEC void print_replica_row(int* e, struct Params params, int limit) {
    for (int i = 0; i < params.R && i < limit; i++) {
        printf("%d", e[i]);
    }
    printf("\n");
};

DECLSPEC __host__ __device__ struct neiborsIndexes SLF(int j, struct Params params) {
    //spin lookup function
    struct neiborsIndexes result;
    int L = params.L;
    // j = x + j * L;
    int x = j % L;
    int y = j / L;

    result.left = (x - 1 + L) % L + y * L;
    result.right = (x + 1) % L + y * L;

    result.up = x + ((y - 1 + L) % L) * L;
    result.down = x + ((y + 1) % L) * L;

    result.diag_left = (x - 1 + L) % L + ((y - 1 + L) % L) * L;
    result.diag_right = (x + 1) % L + ((y + 1) % L) * L;

    return result;
}

#if defined(MCPA_BW_NEIGHBOR_TABLE)
namespace {

neiborsIndexes* cached_equilibration_neighbors = nullptr;
int cached_equilibration_L = 0;
int cached_equilibration_N = 0;

__global__ void initialize_equilibration_neighbors_kernel(
    neiborsIndexes* table, Params params) {
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < params.N) table[j] = SLF(j, params);
}

void attach_equilibration_neighbors(mainMemoryPointers& device, Params params) {
    if (cached_equilibration_neighbors == nullptr
        || cached_equilibration_L != params.L
        || cached_equilibration_N != params.N) {
        if (cached_equilibration_neighbors != nullptr)
            CUDA_CHECK(cudaFree(cached_equilibration_neighbors));
        CUDA_CHECK(cudaMalloc(&cached_equilibration_neighbors,
                              static_cast<size_t>(params.N)
                                  * sizeof(neiborsIndexes)));
        constexpr int threads = 256;
        const int blocks = (params.N + threads - 1) / threads;
        initialize_equilibration_neighbors_kernel<<<blocks, threads>>>(
            cached_equilibration_neighbors, params);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cached_equilibration_L = params.L;
        cached_equilibration_N = params.N;
    }
    device.equilibration_neighbors = cached_equilibration_neighbors;
}

} // namespace
#endif


// hardcoded spin suggestion for init
__device__ int suggest_spin(curandStatePhilox4_32_10_t* curand_states, int r) {
    return (2 * (curand(&curand_states[r]) % 2)) - 1;
};

__global__ void initialize_population_kernel(curandStatePhilox4_32_10_t* curand_states, struct mainMemoryPointers device, struct Params params, initializePopulationMode mode, int s_a, int s_b, int s_c) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    long long replica_shift = (long long)r * params.N;

    for (int k = 0; k < params.N; k++) {
        long long arrayIndex = replica_shift + k;
        if (mode == random_pop) {
            device.spin[arrayIndex] = suggest_spin(curand_states, r);
        }
        else if (mode == by_sublattice) {
            device.spin[arrayIndex] = (k % 3 == 0) * s_a + (k % 3 == 1) * s_b + (k % 3 == 2) * s_c;
        }
        else if (mode == strips) {
            int x = k % params.L;
            int y = k / params.L;

            device.spin[arrayIndex] = ((x + y) % 3 == 0) * s_a + ((x + y) % 3 == 1) * s_b + ((x + y) % 3 == 2) * s_c;
        }
        else {};
    }
};

DECLSPEC void initialize_population(void* curand_states, struct mainMemoryPointers device, struct Params params, initializePopulationMode mode, int s_a, int s_b, int s_c) {
    initialize_population_kernel << < params.blocks, params.threads >> > ((curandStatePhilox4_32_10_t*)curand_states, device, params, mode, s_a, s_b, s_c);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


DECLSPEC void initialize_update_arrays(struct mainMemoryPointers host, struct Params params) {
    for (int i = 0; i < params.R; i++) {
        host.O[i] = i;
        host.replica_family[i] = i;
    }
}

__global__ void setup_curand_kernel(curandStatePhilox4_32_10_t* state, int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, state + id);
}

DECLSPEC void* setup_curand_states(struct Params params) {
    curandStatePhilox4_32_10_t* curand_states = nullptr;

    printf("Allocating %d random states...\n", params.R);
    CUDA_CHECK(cudaMalloc((void**)&curand_states, params.R * sizeof(curandStatePhilox4_32_10_t)));

    printf("Launching kernel with %d blocks, %d threads...\n", params.blocks, params.threads);
    setup_curand_kernel << < params.blocks, params.threads >> > (curand_states, params.seed);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return (void*)curand_states;
}

DECLSPEC size_t curand_states_byte_size(struct Params params) {
    return static_cast<size_t>(params.R) * sizeof(curandStatePhilox4_32_10_t);
}

DECLSPEC void initialize_resampling_rng(int seed) {
    const uint64_t unsigned_seed = static_cast<uint64_t>(
        static_cast<uint32_t>(seed));
    resampling_rng.state = 0;
    resampling_rng.stream = (unsigned_seed << 1U) | 1U;
    (void)next_resampling_random();
    resampling_rng.state += unsigned_seed ^ 0x9e3779b97f4a7c15ULL;
    (void)next_resampling_random();
}

DECLSPEC struct ResamplingRngState get_resampling_rng_state() {
    return resampling_rng;
}

DECLSPEC void set_resampling_rng_state(struct ResamplingRngState state) {
    // PCG requires an odd stream selector.  Checkpoints written by this code
    // already satisfy it; forcing the low bit also makes malformed input safe.
    state.stream |= 1ULL;
    resampling_rng = state;
}

DECLSPEC void copyHostToDevice(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

DECLSPEC void copyDeviceToHost(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__device__ __host__ struct neiborsValues SVLF(struct mainMemoryPointers device, struct neiborsIndexes n_i, long long replica_shift) {
    struct neiborsValues result;
    result.right = device.spin[n_i.right + replica_shift];
    result.down = device.spin[n_i.down + replica_shift];
    result.left = device.spin[n_i.left + replica_shift];
    result.up = device.spin[n_i.up + replica_shift];
    result.diag_left = device.spin[n_i.diag_left + replica_shift];
    result.diag_right = device.spin[n_i.diag_right + replica_shift];

    return result;
}

__device__ __host__ int local_energy(int currentSpin, struct neiborsValues n) {
    // Computes energy of spin i with its neigborts triangles (6)
    // it summirezes each triangle 3 times
    int result = 0;
    result += n.diag_left * n.up * currentSpin;
    result += n.diag_left * n.left * currentSpin;

    result += n.diag_right * n.down * currentSpin;
    result += n.diag_right * n.right * currentSpin;

    result += n.down * n.left * currentSpin;
    result += n.up * n.right * currentSpin;

    return -1 * result;
}


__global__ void calc_device_energy_kernel(struct mainMemoryPointers device, struct Params params) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    long long sum = 0;
    long long replica_shift = (long long)r * params.N;

    for (int j = 0; j < params.N; j++) {
        int currentSpin = device.spin[j + replica_shift];
        struct neiborsIndexes n_i = SLF(j, params);

        struct neiborsValues n = SVLF(device, n_i, replica_shift); // we look into r replica and j spin
        int le = local_energy(currentSpin, n);
        sum += le;
    }

    device.E[r] = (int)(sum / 3); // local energy calcs parkets with overlap 2
}

DECLSPEC void calc_device_energy(struct mainMemoryPointers device, struct Params params) {
    calc_device_energy_kernel << < params.blocks, params.threads >> > (device, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// hardcoded spin suggestion for equilibration
__device__ int suggest_spin_swap(int current_spin) {
    return -current_spin;
}

__global__ void equilibrate_kernel(curandStatePhilox4_32_10_t* curand_states, struct mainMemoryPointers device, struct Params params, int U) {
    /*---------------------------------------------------------------------------------------------
        Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
        population
    ---------------------------------------------------------------------------------------------*/
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;

    long long replica_shift = (long long)r * params.N;
    int replica_energy = device.E[r];
    curandStatePhilox4_32_10_t rng_state = curand_states[r];
    // This counter is per equilibration call; preserve the historical reset to zero.
    int flip_count = 0;

    // BaxterWu: only 1 random per iteration (site only; spin is always flipped).
    // curand4 generates 4 uint32s at once → serve 4 iterations per call (4× savings).
    uint4     rnd;
    uint32_t* rv = (uint32_t*)&rnd;   // treat uint4 as array of 4 uint32s

#if defined(MCPA_BW_GROUPED_ATTEMPTS)
    const int total_attempts = params.N * params.nSteps;
    for (int group_start = 0; group_start < total_attempts; group_start += 4) {
        rnd = curand4(&rng_state);
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            if (group_start + lane >= total_attempts) break;
            int j = (int)(rv[lane] % (uint32_t)params.N);

            int current_spin = device.spin[j + replica_shift];
            int suggested_spin = suggest_spin_swap(current_spin);

            struct neiborsIndexes n_i = SLF(j, params);

            struct neiborsValues n = SVLF(device, n_i, replica_shift);
            int current_local_energy = local_energy(current_spin, n);
            int suggested_local_energy = -current_local_energy;

            int suggested_energy = replica_energy + suggested_local_energy - current_local_energy;

            if ((!params.heat && (suggested_energy < U)) || (params.heat && (suggested_energy > U))) {
                replica_energy = suggested_energy;
                device.spin[j + replica_shift] = suggested_spin;
                flip_count++;
            }
        }
    }
#else
    for (int k = 0; k < params.N * params.nSteps; k++)
    {
        if ((k & 3) == 0) rnd = curand4(&rng_state);
        int j = (int)(rv[k & 3] % (uint32_t)params.N);

        int current_spin = device.spin[j + replica_shift];
        int suggested_spin = suggest_spin_swap(current_spin);

#if defined(MCPA_BW_NEIGHBOR_TABLE)
        struct neiborsIndexes n_i = device.equilibration_neighbors[j];
#else
        struct neiborsIndexes n_i = SLF(j, params);
#endif

        struct neiborsValues n = SVLF(device, n_i, replica_shift);
        int current_local_energy = local_energy(current_spin, n);
        int suggested_local_energy = -current_local_energy;

        int suggested_energy = replica_energy + suggested_local_energy - current_local_energy;

        if ((!params.heat && (suggested_energy < U)) || (params.heat && (suggested_energy > U))) {
            replica_energy = suggested_energy;
            device.spin[j + replica_shift] = suggested_spin;
            flip_count++;
        }
    }
#endif

    device.E[r] = replica_energy;
    device.replica_statistics[r].flip_count = flip_count;
    curand_states[r] = rng_state;
}

DECLSPEC void equilibrate(void* curand_states, struct mainMemoryPointers device, struct Params params, int U) {
#if defined(MCPA_BW_NEIGHBOR_TABLE)
    attach_equilibration_neighbors(device, params);
#endif
    equilibrate_kernel << < params.blocks, params.threads >> > ((curandStatePhilox4_32_10_t*)curand_states, device, params, U);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


DECLSPEC void swap(int* A, int i, int j) {
    int temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}

DECLSPEC void quicksort(struct mainMemoryPointers host, int left, int right, int direction) {
    int Min = (left + right) / 2;
    int i = left;
    int j = right;
    double pivot = direction * host.E[host.O[Min]];

    while (left < j || i < right)
    {
        while (direction * host.E[host.O[i]] > pivot)
            i++;
        while (direction * host.E[host.O[j]] < pivot)
            j--;

        if (i <= j) {
            swap(host.O, i, j);
            i++;
            j--;
        }
        else {
            if (left < j)
                quicksort(host, left, j, direction);
            if (i < right)
                quicksort(host, i, right, direction);
            return;
        }
    }
}

DECLSPEC double prepare_resample_arrays(struct mainMemoryPointers host, struct Params params, int* U,
                                        int* culled_replica_number) {
    quicksort(host, 0, params.R - 1, 1 - 2 * params.heat); //Sorts O by energy

    int nCull = 0;
    //fprintf(e2file, "%f %i\n", 1.0 * (*U) / D_base, E[O[0]]);

    //update energy seiling to the highest available energy
    int U_old = *U;
    int U_new;

    for (int i = 0; i < params.R; i++) {
        U_new = host.E[host.O[i]];
        if ((!params.heat && U_new < U_old) || (params.heat && U_new > U_old)) {
            *U = U_new;
            break;
        }
    }

    if (*U == U_old) {
        if (culled_replica_number != nullptr)
            *culled_replica_number = params.R;
        return 1; // out of replicas
    }

    while ((!params.heat && host.E[host.O[nCull]] >= *U) || (params.heat && host.E[host.O[nCull]] <= *U)) {
        nCull++;
        if (nCull == params.R) {
            break;
        }
    }
    // culling fraction
    double X = nCull;
    X /= params.R;
    if (culled_replica_number != nullptr)
        *culled_replica_number = nCull;
    //printf("U: %d %d %d\n", U_new, U_old, *U);
    printf("Culling factor:\t%f\n", X);
    fflush(stdout);

    for (int i = 0; i < params.R; i++)
        host.update[i] = i;
    if (nCull < params.R) {
        for (int i = 0; i < nCull; i++) {
            // random selection of unculled replica
            const uint32_t draw = next_resampling_random();
            int r = static_cast<int>(draw % static_cast<uint32_t>(params.R - nCull))
                  + nCull;
            host.update[host.O[i]] = host.O[r];
            host.replica_family[host.O[i]] = host.replica_family[host.O[r]];
        }
    }

    return X;
}

DECLSPEC double calc_family_avg_sq_size(struct mainMemoryPointers host, struct Params params, int U) {
    (void)U;
    const FamilyMetrics metrics = calc_family_metrics(host.replica_family, params.R);
    printf("RhoT:\t%f\n", metrics.simpson_concentration);
    return metrics.simpson_concentration;
}

DECLSPEC struct FamilyMetrics calc_family_metrics(const int* family_ids, int R) {
    FamilyMetrics result{};
    if (family_ids == nullptr || R <= 0) {
        result.max_fraction = std::numeric_limits<double>::quiet_NaN();
        result.shannon_entropy = std::numeric_limits<double>::quiet_NaN();
        result.effective_shannon = std::numeric_limits<double>::quiet_NaN();
        result.simpson_concentration = std::numeric_limits<double>::quiet_NaN();
        return result;
    }

    std::vector<int> histogram(static_cast<size_t>(R), 0);
    for (int i = 0; i < R; ++i) {
        const int family = family_ids[i];
        if (family >= 0 && family < R) ++histogram[family];
    }

    double shannon = 0.0;
    double phi = 0.0;
    for (const int size : histogram) {
        if (size == 0) continue;
        ++result.count;
        result.max_size = std::max(result.max_size, size);
        const double fraction = static_cast<double>(size) / static_cast<double>(R);
        shannon -= fraction * std::log(fraction);
        phi += fraction * fraction;
    }
    result.max_fraction = static_cast<double>(result.max_size) / static_cast<double>(R);
    result.shannon_entropy = shannon;
    result.effective_shannon = std::exp(shannon);
    result.simpson_concentration = phi;
    return result;
}

DECLSPEC void print_main_data(struct Files files, int U, double X, double rho_t,
                              int culled_replica_number, double equilibrate_seconds,
                              const struct FamilyMetrics& family_metrics,
                              const struct RunGpuMetadata* run_gpu_metadata,
                              int equilibrate_ceiling, int equilibrate_nsteps,
                              double population_acceptance_ratio,
                              const char* nsteps_policy) {
    // Preserve the three legacy columns and append lossless culling metadata.
    fprintf(files.main_file,
            "%f\t%f\t%f\t%d\t%.17g\t%.9g"
            "\t%d\t%d\t%.17g\t%.17g\t%.17g",
            1.0 * U, X, rho_t, culled_replica_number, X, equilibrate_seconds,
            family_metrics.count, family_metrics.max_size,
            family_metrics.max_fraction, family_metrics.shannon_entropy,
            family_metrics.effective_shannon);
    if (run_gpu_metadata) {
        fprintf(files.main_file,
                "\t%s\t%.1f\t%llu\t%llu\t%llu\t%d\t%d",
                run_gpu_metadata->name,
                run_gpu_metadata->compute_capability,
                static_cast<unsigned long long>(run_gpu_metadata->total_memory_bytes),
                static_cast<unsigned long long>(
                    run_gpu_metadata->free_memory_before_setup_bytes),
                static_cast<unsigned long long>(
                    run_gpu_metadata->free_memory_after_setup_bytes),
                run_gpu_metadata->cuda_driver_version,
                run_gpu_metadata->cuda_runtime_version);
    } else {
        fprintf(files.main_file, "\tNA\tNA\tNA\tNA\tNA\tNA\tNA");
    }
    fprintf(files.main_file, "\t%d\t%d\t%.17g\t%s\n",
            equilibrate_ceiling, equilibrate_nsteps,
            population_acceptance_ratio,
            nsteps_policy ? nsteps_policy : "fixed");
    fflush(files.main_file);
};

__global__ void update_replicas_kernel(struct mainMemoryPointers device, struct Params params) {
    /*---------------------------------------------------------------------------------------------
        Updates the population after the resampling step (done on cpu) by replacing indicated
        replicas by the proper other replica
    -----------------------------------------------------------------------------------------------*/
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    long long replica_shift = (long long)r * params.N;
    int source_r = device.update[r];
    long long source_replica_shift = (long long)source_r * params.N;
    if (source_r != r) {
        for (int j = 0; j < params.N; j++) {
            device.spin[j + replica_shift] = device.spin[j + source_replica_shift];
        }
        device.E[r] = device.E[device.update[r]];
    }
}

DECLSPEC void update_replicas(struct mainMemoryPointers device, struct Params params) {
    update_replicas_kernel << < params.blocks, params.threads >> > (device, params);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void initialize_fourier_phases_kernel(struct mainMemoryPointers device,
                                                 struct Params params) {
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= params.N) return;

    const int x = j % params.L;
    const int y = j / params.L;
    const float theta = 2.0f * static_cast<float>(M_PI) / static_cast<float>(params.L);
    const int coordinate[3] = {x, y, x - y};
    for (int q = 0; q < 3; ++q) {
        const float phase = theta * static_cast<float>(coordinate[q]);
        float sine = 0.0f;
        float cosine = 0.0f;
        sincosf(phase, &sine, &cosine);
        device.fourier_phase_cos[static_cast<size_t>(q) * params.N + j] = cosine;
        device.fourier_phase_sin[static_cast<size_t>(q) * params.N + j] = sine;
    }
}

DECLSPEC void initialize_fourier_phases(struct mainMemoryPointers device,
                                        struct Params params) {
    constexpr int threads = 256;
    const int blocks = (params.N + threads - 1) / threads;
    initialize_fourier_phases_kernel<<<blocks, threads>>>(device, params);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void calc_replica_statistics_kernel(struct mainMemoryPointers device, struct Params params, int U) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    long long replica_shift = (long long)r * params.N;

    for (int sublattice_index = 0; sublattice_index < 3; sublattice_index++) {
        device.replica_statistics[r].magnetization[sublattice_index] = 0;
        device.replica_statistics[r].polarization[sublattice_index] = 0;
        device.replica_statistics[r].order_structure_factor[sublattice_index] = 0.0;
    }

    if (device.E[r] == U) {
        double fourier_real[3][3] = {};
        double fourier_imag[3][3] = {};
        for (int j = 0; j < params.N; j++) {
            int current_spin = device.spin[j + replica_shift];

            int x = j % params.L;
            int y = j / params.L;
            int sublattice_index = (x + y) % 3;

            device.replica_statistics[r].magnetization[sublattice_index] += current_spin;

            struct neiborsIndexes n_i = SLF(j, params);
            struct neiborsValues n = SVLF(device, n_i, replica_shift);

            device.replica_statistics[r].polarization[sublattice_index] += current_spin * (n.up + n.left + n.diag_right);

            for (int q = 0; q < 3; ++q) {
                const size_t phase_index = static_cast<size_t>(q) * params.N + j;
                fourier_real[sublattice_index][q] +=
                    static_cast<double>(current_spin) * device.fourier_phase_cos[phase_index];
                fourier_imag[sublattice_index][q] +=
                    static_cast<double>(current_spin) * device.fourier_phase_sin[phase_index];
            }
        }

        for (int q = 0; q < 3; ++q) {
            double mode_norm_sq = 0.0;
            for (int sublattice_index = 0; sublattice_index < 3; ++sublattice_index) {
                const double real = fourier_real[sublattice_index][q];
                const double imag = fourier_imag[sublattice_index][q];
                mode_norm_sq += real * real + imag * imag;
            }
            device.replica_statistics[r].order_structure_factor[q] =
                mode_norm_sq / static_cast<double>(params.N);
        }
    }
};

DECLSPEC void calc_replica_statistics(struct mainMemoryPointers device, struct Params params, int U) {
    //cudaMemset(device.replica_statistics, 0, params.replicaStatisticsByteSize);
    calc_replica_statistics_kernel << < params.blocks, params.threads >> > (device, params, U);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
};

namespace {

double clustered_variance(const std::vector<int>& counts,
                          const std::vector<double>& sums,
                          int family_count, int sample_count, double mean) {
    if (family_count < 2 || sample_count <= 0)
        return std::numeric_limits<double>::quiet_NaN();
    double residual_sq_sum = 0.0;
    for (size_t family = 0; family < counts.size(); ++family) {
        if (counts[family] == 0) continue;
        const double residual = sums[family] - mean * static_cast<double>(counts[family]);
        residual_sq_sum += residual * residual;
    }
    const double correction = static_cast<double>(family_count)
                            / static_cast<double>(family_count - 1);
    const double denominator = static_cast<double>(sample_count) * sample_count;
    return correction * residual_sq_sum / denominator;
}

double clustered_covariance(const std::vector<int>& counts,
                            const std::vector<double>& sums_a,
                            const std::vector<double>& sums_b,
                            int family_count, int sample_count,
                            double mean_a, double mean_b) {
    if (family_count < 2 || sample_count <= 0)
        return std::numeric_limits<double>::quiet_NaN();
    double residual_product_sum = 0.0;
    for (size_t family = 0; family < counts.size(); ++family) {
        if (counts[family] == 0) continue;
        const double residual_a = sums_a[family]
                                - mean_a * static_cast<double>(counts[family]);
        const double residual_b = sums_b[family]
                                - mean_b * static_cast<double>(counts[family]);
        residual_product_sum += residual_a * residual_b;
    }
    const double correction = static_cast<double>(family_count)
                            / static_cast<double>(family_count - 1);
    const double denominator = static_cast<double>(sample_count) * sample_count;
    return correction * residual_product_sum / denominator;
}

} // namespace

DECLSPEC struct AggregateDiagnostics calc_aggregate_diagnostics(
    struct mainMemoryPointers host, struct Params params, int U,
    const int* measured_replica_family) {
    AggregateDiagnostics result{};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    result.family_max_fraction_at_E = nan;
    result.m_s_mean = nan;
    result.p_s_mean = nan;
    for (double& value : result.order_sf_k) value = nan;
    result.family_var_mean_m_s = nan;
    result.family_var_mean_p_s = nan;
    result.family_var_mean_order_sf0 = nan;
    result.family_var_mean_order_sf_kmin = nan;
    result.family_cov_mean_order_sf0_kmin = nan;

    if (params.R <= 0 || params.N <= 0 || host.E == nullptr
        || host.replica_statistics == nullptr) return result;
    if (measured_replica_family == nullptr)
        measured_replica_family = host.replica_family;
    if (measured_replica_family == nullptr) return result;

    std::vector<int> family_sizes(static_cast<size_t>(params.R), 0);
    std::vector<double> family_m_s(static_cast<size_t>(params.R), 0.0);
    std::vector<double> family_p_s(static_cast<size_t>(params.R), 0.0);
    std::vector<double> family_sf0(static_cast<size_t>(params.R), 0.0);
    std::vector<double> family_sfkmin(static_cast<size_t>(params.R), 0.0);

    int sample_count = 0;
    int max_family_size = 0;
    double sum_m_s = 0.0;
    double sum_p_s = 0.0;
    double sum_sf0 = 0.0;
    double sum_sfkmin = 0.0;
    double sum_sfk[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < params.R; ++i) {
        if (host.E[i] != U) continue;
        const int family = measured_replica_family[i];
        if (family < 0 || family >= params.R) continue;

        const replicaStatistics& statistics = host.replica_statistics[i];
        double magnetization_norm_sq = 0.0;
        double polarization_norm_sq = 0.0;
        for (int component = 0; component < 3; ++component) {
            const double m = static_cast<double>(statistics.magnetization[component]);
            const double p = static_cast<double>(statistics.polarization[component]);
            magnetization_norm_sq += m * m;
            polarization_norm_sq += p * p;
        }
        const double m_s = std::sqrt(3.0 * magnetization_norm_sq)
                         / static_cast<double>(params.N);
        const double p_s = std::sqrt(polarization_norm_sq)
                         / (std::sqrt(3.0) * static_cast<double>(params.N));
        const double sf0 = magnetization_norm_sq / static_cast<double>(params.N);
        double sfkmin = 0.0;
        for (int q = 0; q < 3; ++q) {
            sum_sfk[q] += statistics.order_structure_factor[q];
            sfkmin += statistics.order_structure_factor[q];
        }
        sfkmin /= 3.0;

        ++sample_count;
        ++family_sizes[family];
        family_m_s[family] += m_s;
        family_p_s[family] += p_s;
        family_sf0[family] += sf0;
        family_sfkmin[family] += sfkmin;
        sum_m_s += m_s;
        sum_p_s += p_s;
        sum_sf0 += sf0;
        sum_sfkmin += sfkmin;
    }

    if (sample_count == 0) return result;
    for (const int size : family_sizes) {
        if (size == 0) continue;
        ++result.family_count_at_E;
        max_family_size = std::max(max_family_size, size);
    }
    result.family_max_fraction_at_E = static_cast<double>(max_family_size)
                                    / static_cast<double>(sample_count);
    result.m_s_mean = sum_m_s / sample_count;
    result.p_s_mean = sum_p_s / sample_count;
    const double mean_sf0 = sum_sf0 / sample_count;
    const double mean_sfkmin = sum_sfkmin / sample_count;
    for (int q = 0; q < 3; ++q)
        result.order_sf_k[q] = sum_sfk[q] / sample_count;

    result.family_var_mean_m_s = clustered_variance(
        family_sizes, family_m_s, result.family_count_at_E, sample_count,
        result.m_s_mean);
    result.family_var_mean_p_s = clustered_variance(
        family_sizes, family_p_s, result.family_count_at_E, sample_count,
        result.p_s_mean);
    result.family_var_mean_order_sf0 = clustered_variance(
        family_sizes, family_sf0, result.family_count_at_E, sample_count,
        mean_sf0);
    result.family_var_mean_order_sf_kmin = clustered_variance(
        family_sizes, family_sfkmin, result.family_count_at_E, sample_count,
        mean_sfkmin);
    result.family_cov_mean_order_sf0_kmin = clustered_covariance(
        family_sizes, family_sf0, family_sfkmin, result.family_count_at_E,
        sample_count, mean_sf0, mean_sfkmin);
    return result;
}

DECLSPEC void initialize_print(struct Files files) {
    fprintf(files.main_file,
            "E\tculling_factor\treplica_family_avg_sq"
            "\tculled_replica_number\tculling_factor_full_precision"
            "\tequilibrate_seconds"
            "\tfamily_count\tfamily_max_size\tfamily_max_fraction"
            "\tfamily_shannon_entropy\tfamily_effective_shannon"
            "\tgpu_name\tgpu_compute_capability\tgpu_total_memory_bytes"
            "\tgpu_free_memory_before_setup_bytes"
            "\tgpu_free_memory_after_setup_bytes"
            "\tcuda_driver_version\tcuda_runtime_version"
            "\tequilibrate_ceiling\tequilibrate_nsteps"
            "\tpopulation_acceptance_ratio\tnsteps_policy\n");
    fflush(files.main_file);

    fprintf(files.agg_stats_file,
            "E\tculled_replica_number\tflip_rate\tm_a\tm_b\tm_c\tp_a\tp_b\tp_c\tempty"
            "\tm_a_sq\tm_b_sq\tm_c_sq\tp_a_sq\tp_b_sq\tp_c_sq"
            "\tbranch_ppp\tbranch_pmm\tbranch_mpm\tbranch_mmp\tbranch_other"
            "\tfamily_count_at_E\tfamily_max_fraction_at_E"
            "\tm_s_mean\tp_s_mean\torder_sf_k1\torder_sf_k2\torder_sf_k3"
            "\tfamily_var_mean_m_s\tfamily_var_mean_p_s"
            "\tfamily_var_mean_order_sf0\tfamily_var_mean_order_sf_kmin"
            "\tfamily_cov_mean_order_sf0_kmin\n");
    fflush(files.agg_stats_file);

    fprintf(files.detailed_stats_file,
            "E\tflip_count\tm_a\tm_b\tm_c\tp_a\tp_b\tp_c\tempty"
            "\tdetail_schema_version\trun_seed\theat"
            "\treplica_id\tfamily_id\tsample_rank\tselection_hash"
            "\tshell_population\tsample_count\tinclusion_probability"
            "\tdetailed_limit\tsampling_policy"
            "\tm_s\tp_s\torder_sf_0\torder_sf_k1\torder_sf_k2"
            "\torder_sf_k3\torder_sf_kmean\n");
    fflush(files.detailed_stats_file);
};

DECLSPEC uint64_t detailed_sample_hash(int seed, bool heat, int U, int replica_id) {
    // SplitMix64 finalizer with fixed input mixing.  Unlike std::hash, these
    // values are reproducible across compilers and machines.
    uint64_t value = static_cast<uint32_t>(seed);
    value ^= static_cast<uint64_t>(heat ? 1U : 0U)
           * UINT64_C(0xd6e8feb86659fd93);
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(U))
           * UINT64_C(0x9e3779b97f4a7c15);
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(replica_id))
           * UINT64_C(0xbf58476d1ce4e5b9);
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

DECLSPEC int print_detailed_stats(struct mainMemoryPointers host, struct Params params,
                                 struct Files files, int U, int limit,
                                 const int* measured_replica_family) {
    struct Candidate {
        uint64_t hash;
        int replica_id;
    };
    std::vector<Candidate> selected;
    for (int replica_id = 0; replica_id < params.R; ++replica_id) {
        if (host.E[replica_id] != U) continue;
        selected.push_back(Candidate{
            detailed_sample_hash(params.seed, params.heat, U, replica_id), replica_id});
    }
    const int total_matching_replicas = static_cast<int>(selected.size());
    if (limit == 0 || selected.empty()) {
        fflush(files.detailed_stats_file);
        return 0;
    }

    const auto less = [](const Candidate& left, const Candidate& right) {
        if (left.hash != right.hash) return left.hash < right.hash;
        return left.replica_id < right.replica_id;
    };
    if (limit > 0 && static_cast<size_t>(limit) < selected.size()) {
        std::nth_element(selected.begin(), selected.begin() + limit,
                         selected.end(), less);
        selected.resize(static_cast<size_t>(limit));
    }
    std::sort(selected.begin(), selected.end(), less);

    const double sqrt_three = std::sqrt(3.0);
    const int sample_count = static_cast<int>(selected.size());
    const double inclusion_probability = static_cast<double>(sample_count)
                                       / static_cast<double>(total_matching_replicas);
    for (size_t rank = 0; rank < selected.size(); ++rank) {
        const int replica_id = selected[rank].replica_id;
        const replicaStatistics& statistics = host.replica_statistics[replica_id];
        double magnetization_norm_sq = 0.0;
        double polarization_norm_sq = 0.0;
        for (int component = 0; component < 3; ++component) {
            const double m = static_cast<double>(statistics.magnetization[component]);
            const double p = static_cast<double>(statistics.polarization[component]);
            magnetization_norm_sq += m * m;
            polarization_norm_sq += p * p;
        }
        const double m_s = sqrt_three * std::sqrt(magnetization_norm_sq)
                         / static_cast<double>(params.N);
        const double p_s = std::sqrt(polarization_norm_sq)
                         / (sqrt_three * static_cast<double>(params.N));
        const double order_sf_0 = magnetization_norm_sq
                                / static_cast<double>(params.N);
        const double order_sf_kmean =
            (statistics.order_structure_factor[0]
             + statistics.order_structure_factor[1]
             + statistics.order_structure_factor[2]) / 3.0;
        const int family_id = measured_replica_family
                            ? measured_replica_family[replica_id] : -1;

        // Keep the historical nine leading fields, including its empty
        // placeholder, and append the explicit sampling/evidence contract.
        fprintf(files.detailed_stats_file, "%f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t",
                1.0 * U, statistics.flip_count,
                statistics.magnetization[0], statistics.magnetization[1],
                statistics.magnetization[2], statistics.polarization[0],
                statistics.polarization[1], statistics.polarization[2]);
        fprintf(files.detailed_stats_file,
                "\t2\t%d\t%d\t%d\t%d\t%zu\t0x%016llx\t%d\t%d\t%.17g\t%d"
                "\tlowest_splitmix64_seed_direction_energy_replica"
                "\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",
                params.seed, static_cast<int>(params.heat), replica_id, family_id, rank,
                static_cast<unsigned long long>(selected[rank].hash),
                total_matching_replicas, sample_count, inclusion_probability, limit,
                m_s, p_s, order_sf_0,
                statistics.order_structure_factor[0],
                statistics.order_structure_factor[1],
                statistics.order_structure_factor[2], order_sf_kmean);
    }
    fflush(files.detailed_stats_file);
    return static_cast<int>(selected.size());
}


DECLSPEC void print_agg_stats(struct mainMemoryPointers host, struct Params params,
                             struct Files files, int U,
                             const int* measured_replica_family) {
    int culled_replica_number = 0;
    // Production R and N make every one of these sums capable of exceeding
    // INT_MAX, even though the per-replica statistics themselves fit in int.
    int64_t sum_flip_count = 0;
    int64_t sum_magnetization[3] = { 0, 0, 0 };
    int64_t sum_polarization[3] = { 0, 0, 0 };
    int64_t sum_magnetization_sq[3] = { 0, 0, 0 };
    int64_t sum_polarization_sq[3] = { 0, 0, 0 };
    int64_t branch_count[5] = { 0, 0, 0, 0, 0 };

    for (int i = 0; i < params.R; i++) {
        if (host.E[i] == U) {
            culled_replica_number++;
            sum_flip_count += static_cast<int64_t>(host.replica_statistics[i].flip_count);
            for (int j = 0; j < 3; j++) {
                const int64_t m = static_cast<int64_t>(host.replica_statistics[i].magnetization[j]);
                const int64_t p = static_cast<int64_t>(host.replica_statistics[i].polarization[j]);
                sum_magnetization[j] += (m < 0) ? -m : m;
                sum_polarization[j] += (p < 0) ? -p : p;
                sum_magnetization_sq[j] += m * m;
                sum_polarization_sq[j] += p * p;
            }

            const int ma = host.replica_statistics[i].magnetization[0];
            const int mb = host.replica_statistics[i].magnetization[1];
            const int mc = host.replica_statistics[i].magnetization[2];
            if (ma > 0 && mb > 0 && mc > 0)      branch_count[0]++;
            else if (ma > 0 && mb < 0 && mc < 0) branch_count[1]++;
            else if (ma < 0 && mb > 0 && mc < 0) branch_count[2]++;
            else if (ma < 0 && mb < 0 && mc > 0) branch_count[3]++;
            else                                  branch_count[4]++;
        }
    }

    fprintf(files.agg_stats_file, "%f\t", 1.0 * U);
    fprintf(files.agg_stats_file, "%d\t", culled_replica_number);
    const double denominator = static_cast<double>(culled_replica_number);
    fprintf(files.agg_stats_file, "%f\t", static_cast<double>(sum_flip_count) / denominator);
    for (int j = 0; j < 3; j++)
        fprintf(files.agg_stats_file, "%f\t", static_cast<double>(sum_magnetization[j]) / denominator);
    for (int j = 0; j < 3; j++)
        fprintf(files.agg_stats_file, "%f\t", static_cast<double>(sum_polarization[j]) / denominator);
    // Keep the historical empty placeholder empty, then append new columns.
    fprintf(files.agg_stats_file, "\t");
    for (int j = 0; j < 3; j++)
        fprintf(files.agg_stats_file, "%.17g\t", static_cast<double>(sum_magnetization_sq[j]) / denominator);
    for (int j = 0; j < 3; j++)
        fprintf(files.agg_stats_file, "%.17g\t", static_cast<double>(sum_polarization_sq[j]) / denominator);
    fprintf(files.agg_stats_file, "%lld\t%lld\t%lld\t%lld\t%lld",
            static_cast<long long>(branch_count[0]), static_cast<long long>(branch_count[1]),
            static_cast<long long>(branch_count[2]), static_cast<long long>(branch_count[3]),
            static_cast<long long>(branch_count[4]));

    const AggregateDiagnostics diagnostics = calc_aggregate_diagnostics(
        host, params, U, measured_replica_family);
    fprintf(files.agg_stats_file,
            "\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g"
            "\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",
            diagnostics.family_count_at_E,
            diagnostics.family_max_fraction_at_E,
            diagnostics.m_s_mean,
            diagnostics.p_s_mean,
            diagnostics.order_sf_k[0], diagnostics.order_sf_k[1],
            diagnostics.order_sf_k[2],
            diagnostics.family_var_mean_m_s,
            diagnostics.family_var_mean_p_s,
            diagnostics.family_var_mean_order_sf0,
            diagnostics.family_var_mean_order_sf_kmin,
            diagnostics.family_cov_mean_order_sf0_kmin);

    fflush(files.agg_stats_file);

};
