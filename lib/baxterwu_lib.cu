#include "baxterwu_lib.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

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
    int arrayIndex = r * params.N;
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


// hardcoded spin suggestion for init
__device__ int suggest_spin(curandStatePhilox4_32_10_t* curand_states, int r) {
    return (2 * (curand(&curand_states[r]) % 2)) - 1;
};

__global__ void initialize_population_kernel(curandStatePhilox4_32_10_t* curand_states, struct mainMemoryPointers device, struct Params params, initializePopulationMode mode, int s_a, int s_b, int s_c) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int replica_shift = r * params.N;

    for (int k = 0; k < params.N; k++) {
        int arrayIndex = replica_shift + k;
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

__device__ struct neiborsValues SVLF(struct mainMemoryPointers device, struct neiborsIndexes n_i, int replica_shift) {
    struct neiborsValues result;
    result.right = device.spin[n_i.right + replica_shift];
    result.down = device.spin[n_i.down + replica_shift];
    result.left = device.spin[n_i.left + replica_shift];
    result.up = device.spin[n_i.up + replica_shift];
    result.diag_left = device.spin[n_i.diag_left + replica_shift];
    result.diag_right = device.spin[n_i.diag_right + replica_shift];

    return result;
}

__device__ int local_energy(int currentSpin, struct neiborsValues n) {
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
    int sum = 0;
    int replica_shift = r * params.N;

    for (int j = 0; j < params.N; j++) {
        int currentSpin = device.spin[j + replica_shift];
        struct neiborsIndexes n_i = SLF(j, params);

        struct neiborsValues n = SVLF(device, n_i, replica_shift); // we look into r replica and j spin
        int le = local_energy(currentSpin, n);
        sum += le;
    }

    device.E[r] = sum / 3; // local energy calcs parkets with overlap 2
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

__global__ void equilibrate_kernel(curandStatePhilox4_32_10_t* curand_states, struct mainMemoryPointers device, struct Params params, int U) { //, enum equlibrateMode equlibrate_mode
    /*---------------------------------------------------------------------------------------------
        Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
        population
    ---------------------------------------------------------------------------------------------*/
    //long long int kernel_start = clock();
    //int SLF_spend = 0;

    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int replica_shift = r * params.N;
    device.replica_statistics[r].flip_count = 0;

    //for (int k = 0; k < 1; k++)
    for (int k = 0; k < params.N * params.nSteps; k++)
    {
        int j = curand(&curand_states[r]) % params.N;

        int current_spin = device.spin[j + replica_shift];
        int suggested_spin = suggest_spin_swap(current_spin);

        struct neiborsIndexes n_i = SLF(j, params);
        struct neiborsValues n = SVLF(device, n_i, replica_shift);

        int current_local_energy = local_energy(current_spin, n);
        int suggested_local_energy = -current_local_energy;
        //int suggested_local_energy = local_energy(suggested_spin, n);

        int suggested_energy = device.E[r] + suggested_local_energy - current_local_energy;

        if ((!params.heat && (suggested_energy < U)) || (params.heat && (suggested_energy > U))) {
            device.E[r] = suggested_energy;
            device.spin[j + replica_shift] = suggested_spin;
            device.replica_statistics[r].flip_count++;
        }
    }

    //int kernel_end = clock();
    //kernel_timers[r] += kernel_end - kernel_start;
    //if (r == 0)
    //    printf("kernel %d: %d\n", r, (kernel_end - kernel_start));
    //atomicAdd(&kernel_timers[1], (float)SLF_spend);
}

DECLSPEC void equilibrate(void* curand_states, struct mainMemoryPointers device, struct Params params, int U) {
    equilibrate_kernel << < params.blocks, params.threads >> > ((curandStatePhilox4_32_10_t*)curand_states, device, params, U);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void swap(int* A, int i, int j) {
    int temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}

void quicksort(struct mainMemoryPointers host, int left, int right, int direction) {
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

DECLSPEC double prepare_resample_arrays(struct mainMemoryPointers host, struct Params params, int* U) {
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
    //printf("U: %d %d %d\n", U_new, U_old, *U);
    printf("Culling factor:\t%f\n", X);
    fflush(stdout);

    for (int i = 0; i < params.R; i++)
        host.update[i] = i;
    if (nCull < params.R) {
        for (int i = 0; i < nCull; i++) {
            // random selection of unculled replica
            int r = (rand() % (params.R - nCull)) + nCull; // different random number generator for resampling
            host.update[host.O[i]] = host.O[r];
            host.replica_family[host.O[i]] = host.replica_family[host.O[r]];
        }
    }

    return X;
}

DECLSPEC double calc_family_avg_sq_size(struct mainMemoryPointers host, struct Params params, int U) {
    // histogram of family sizes
    int* famHist = (int*)calloc(params.R, sizeof(int));

    for (int i = 0; i < params.R; i++) {
        famHist[host.replica_family[i]]++;
    }
    double rho_t = 0.0;
    for (int i = 0; i < params.R; i++) {
        rho_t += famHist[i] * famHist[i];
    }
    rho_t /= params.R;
    rho_t /= params.R;
    printf("RhoT:\t%f\n", rho_t);

    free(famHist);

    return rho_t;
}

DECLSPEC void print_main_data(struct Files files, int U, double X, double rho_t) {
    fprintf(files.main_file, "%f\t%f\t%f\n", 1.0 * U, X, rho_t);
    fflush(files.main_file);
};

__global__ void update_replicas_kernel(struct mainMemoryPointers device, struct Params params) {
    /*---------------------------------------------------------------------------------------------
        Updates the population after the resampling step (done on cpu) by replacing indicated
        replicas by the proper other replica
    -----------------------------------------------------------------------------------------------*/
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int replica_shift = r * params.N;
    int source_r = device.update[r];
    int source_replica_shift = source_r * params.N;
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

__global__ void calc_replica_statistics_kernel(struct mainMemoryPointers device, struct Params params, int U) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int replica_shift = r * params.N;

    for (int sublattice_index = 0; sublattice_index < 3; sublattice_index++) {
        device.replica_statistics[r].magnetization[sublattice_index] = 0;
        device.replica_statistics[r].polarization[sublattice_index] = 0;
    }

    if (device.E[r] == U) {
        for (int j = 0; j < params.N; j++) {
            int current_spin = device.spin[j + replica_shift];

            int x = j % params.L;
            int y = j / params.L;
            int sublattice_index = (x + y) % 3;

            device.replica_statistics[r].magnetization[sublattice_index] += current_spin;

            struct neiborsIndexes n_i = SLF(j, params);
            struct neiborsValues n = SVLF(device, n_i, replica_shift);

            device.replica_statistics[r].polarization[sublattice_index] += current_spin * (n.up + n.left + n.diag_right);
        }
    }
};

DECLSPEC void calc_replica_statistics(struct mainMemoryPointers device, struct Params params, int U) {
    //cudaMemset(device.replica_statistics, 0, params.replicaStatisticsByteSize);
    calc_replica_statistics_kernel << < params.blocks, params.threads >> > (device, params, U);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
};

DECLSPEC void initialize_print(struct Files files) {
    fprintf(files.main_file, "E\tculling_factor\treplica_family_avg_sq\n");
    fflush(files.main_file);

    fprintf(files.agg_stats_file, "E\tculled_replica_number\tflip_rate\tm_a\tm_b\tm_c\tp_a\tp_b\tp_c\tempty\n");
    fflush(files.agg_stats_file);

    fprintf(files.detailed_stats_file, "E\tflip_count\tm_a\tm_b\tm_c\tp_a\tp_b\tp_c\tempty\n");
    fflush(files.detailed_stats_file);
};

DECLSPEC void print_detailed_stats(struct mainMemoryPointers host, struct Params params, struct Files files, int U, int limit) {
    for (int i = 0; (i < params.R) && (i < limit); i++) {
        if (host.E[i] == U) {
            fprintf(files.detailed_stats_file, "%f\t", 1.0 * U);
            fprintf(files.detailed_stats_file, "%d\t", host.replica_statistics[i].flip_count);
            for (int j = 0; j < 3; j++)
                fprintf(files.detailed_stats_file, "%d\t", host.replica_statistics[i].magnetization[j]);
            for (int j = 0; j < 3; j++)
                fprintf(files.detailed_stats_file, "%d\t", host.replica_statistics[i].polarization[j]);
            fprintf(files.detailed_stats_file, "\n");
        }
    }
    fflush(files.detailed_stats_file);
}


DECLSPEC void print_agg_stats(struct mainMemoryPointers host, struct Params params, struct Files files, int U) {
    int culled_replica_number = 0;
    int sum_flip_count = 0;
    int sum_magnetization[3] = { 0, 0, 0 };
    int sum_polarization[3] = { 0, 0, 0 };

    for (int i = 0; i < params.R; i++) {
        if (host.E[i] == U) {
            culled_replica_number++;
            sum_flip_count += host.replica_statistics[i].flip_count;
            for (int j = 0; j < 3; j++)
                sum_magnetization[j] += abs(host.replica_statistics[i].magnetization[j]);
            for (int j = 0; j < 3; j++)
                sum_polarization[j] += abs(host.replica_statistics[i].polarization[j]);
        }
    }

    fprintf(files.agg_stats_file, "%f\t", 1.0 * U);
    fprintf(files.agg_stats_file, "%d\t", culled_replica_number);
    fprintf(files.agg_stats_file, "%f\t", 1.0 * sum_flip_count / culled_replica_number);
    for (int j = 0; j < 3; j++)
        fprintf(files.agg_stats_file, "%f\t", 1.0 * sum_magnetization[j] / culled_replica_number);
    for (int j = 0; j < 3; j++)
        fprintf(files.agg_stats_file, "%f\t", 1.0 * sum_polarization[j] / culled_replica_number);
    fprintf(files.agg_stats_file, "\n");

    fflush(files.agg_stats_file);

};



