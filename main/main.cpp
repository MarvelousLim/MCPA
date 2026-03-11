#pragma once

#include "baxterwu_lib.h"
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }


int main(int argc, char* argv[]) {
    clock_t global_start = clock();
    clock_t equilibrate_ticks = 0;
    

    statisticsMode statistics_mode = detailed;
    equlibrateMode equlibrate_mode = normal;
    initializePopulationMode initialize_population_mode = random;

    Params params;
    params.seed = atoi(argv[1]); 
    params.L = atoi(argv[2]);
    params.N = params.L * params.L;
    params.blocks = atoi(argv[3]);
    params.threads = atoi(argv[4]);
    params.R = params.blocks * params.threads;
    params.nSteps = atoi(argv[5]);
    params.fullLatticeByteSize = params.R * params.N * sizeof(int);
    params.singleIntRowByteSize = params.R * sizeof(int);
    params.replicaStatisticsByteSize = params.R * sizeof(replicaStatistics);
    params.heat = (bool)atoi(argv[6]);

    //long long int* host_kernel_timers, * device_kernel_timers;

    //host_kernel_timers = (long long int*)malloc(params.R * sizeof(long long int));
    //CUDA_CHECK(cudaMallocManaged((void**)&device_kernel_timers, params.R * sizeof(long long int)));
    //cudaMemset(device_kernel_timers, 0, params.R * sizeof(long long int));

    srand(params.seed);

    char s[150];
    const char* prefix = "2DBaxterWu";
    const char* heating = params.heat ? "Heating" : "";

    struct Files files;
    sprintf(s, "C:/Users/MarvelousNote3/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_main.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    files.main_file = fopen(s, "w");
    //printf(s);
    sprintf(s, "C:/Users/MarvelousNote3/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_agg_stats.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    files.agg_stats_file = fopen(s, "w");
    //printf(s);
    sprintf(s, "C:/Users/MarvelousNote3/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_detailed_stats.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    files.detailed_stats_file = fopen(s, "w");
    //printf(s);


    mainMemoryPointers host, device;
    // Allocate space on host
    //host.spin = (int*)malloc(params.fullLatticeByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.spin, params.fullLatticeByteSize));

    host.E = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.E, params.singleIntRowByteSize));

    host.replica_statistics = (replicaStatistics*)malloc(params.replicaStatisticsByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.replica_statistics, params.replicaStatisticsByteSize));

    host.O = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.O, params.singleIntRowByteSize));

    host.update = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.update, params.singleIntRowByteSize));

    host.replica_family = (int*)malloc(params.singleIntRowByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.replica_family, params.singleIntRowByteSize));

    // Init Philox
    void* curand_states = setup_curand_states(params);


    initialize_population(curand_states, device, params, initialize_population_mode);
    initialize_update_arrays(host, params);
    initialize_print(files);

    calc_device_energy(device, params);

    int upper_energy = 1;
    int lower_energy = -2 * params.N - 1;

	int U = (params.heat ? lower_energy : upper_energy);	// U is energy ceiling

    int i = 0;
    while ((U >= lower_energy && !params.heat) || (U <= upper_energy && params.heat)) {
        printf("U:\t%f between %d and %d; nSteps: %d;\n", 1.0 * U, upper_energy, lower_energy, params.nSteps);

        clock_t equilibrate_time_start = clock();

        equilibrate(curand_states, device, params, U, equlibrate_mode);

        clock_t equilibrate_time_end = clock();
        equilibrate_ticks += equilibrate_time_end - equilibrate_time_start;

        copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);

        double X = prepare_resample_arrays(host, params, &U); //
        if (X == 1 || ++i > 10) {
            printf("ended with no replicas\n");
            break;
        }

        // not U has been lowered to next values - we collect our statistics now, but stritly before updates on replicas
        double rho_t = calc_family_avg_sq_size(host, params, U);
        print_main_data(files, U, X, rho_t);

        calc_replica_statistics(device, params, U);
        copyDeviceToHost(host.replica_statistics, device.replica_statistics, params.replicaStatisticsByteSize);
        print_agg_stats(host, params, files, U);
        print_detailed_stats(host, params, files, U);

        // here goes actuall resample
        copyHostToDevice(device.update, host.update, params.singleIntRowByteSize);
        update_replicas(device, params);

    }



    fclose(files.main_file);
    fclose(files.agg_stats_file);
    fclose(files.detailed_stats_file);


    //free(host.spin);
    CUDA_CHECK(cudaFree(device.spin));

    free(host.E);
    CUDA_CHECK(cudaFree(device.E));

    free(host.replica_statistics);
    CUDA_CHECK(cudaFree(device.replica_statistics));

    free(host.O);
    CUDA_CHECK(cudaFree(device.O));

    free(host.update);
    CUDA_CHECK(cudaFree(device.update));

    free(host.replica_family);
    CUDA_CHECK(cudaFree(device.replica_family));

    CUDA_CHECK(cudaFree(curand_states));


    clock_t global_end = clock();
    double global_time_spend = (double)(global_end - global_start) / CLOCKS_PER_SEC;
    double equilibrate_time_spend = (double)equilibrate_ticks / CLOCKS_PER_SEC;
    printf("spend %.2fs, among them %.2fs (%.2f%%) spend on equilibrate\n", global_time_spend, equilibrate_time_spend, 100.0 * equilibrate_time_spend / global_time_spend);

    //copyDeviceToHost(host_kernel_timers, device_kernel_timers, params.R * sizeof(long long int));

    //double kernel_time_spend = (double)host_kernel_timers[0] / (CLOCKS_PER_SEC * params.R);
    //printf("among them %.2f s (%.2f)%% spend on equilibrate inside kernel\n", kernel_time_spend, 100.0 * kernel_time_spend / global_time_spend);
    //
    //double slf_time_spend = (double)host_kernel_timers[1] / (CLOCKS_PER_SEC * params.R);
    //printf("among them %.2f s (%.2f)%% spend on slf inside kernel\n", slf_time_spend, 100.0 * slf_time_spend / global_time_spend);

    //free(host_kernel_timers);
    //CUDA_CHECK(cudaFree(device_kernel_timers));

    return 0;
}


