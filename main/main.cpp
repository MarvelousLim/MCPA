#pragma once

#include "baxterwu_lib.h"
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

#define OPEN_FILE_CHECK(fp, filename) do { \
    (fp) = fopen((filename), "w"); \
    if ((fp) == NULL) { \
        fprintf(stderr, "ERROR: Could not open file: %s\n", (filename)); \
    } else { \
        printf("SUCCESS: Opened %s\n", (filename)); \
    } \
} while(0)

#define FREE_HOST_DEVICE(host_ptr, dev_ptr) do { \
    free(host_ptr); \
    CUDA_CHECK(cudaFree(dev_ptr)); \
} while(0)


int main(int argc, char* argv[]) {
    clock_t global_start = clock();
    clock_t equilibrate_ticks = 0;
    clock_t prepare_resample_ticks = 0;
    clock_t family_avg_ticks = 0;
    clock_t replica_stats_ticks = 0;
    clock_t update_replicas_ticks = 0;
    clock_t copy_d2h_ticks = 0;
    clock_t copy_h2d_ticks = 0;

    statisticsMode statistics_mode = detailed;
    equlibrateMode equlibrate_mode = normal;
    initializePopulationMode initialize_population_mode = random_pop;

    Params params;
    params.seed = atoi(argv[1]); 
    params.L = atoi(argv[2]);
    params.N = params.L * params.L;
    params.blocks = atoi(argv[3]);
    params.threads = atoi(argv[4]);
    params.R = params.blocks * params.threads;
    params.nSteps = atoi(argv[5]);
    params.fullLatticeByteSize = (size_t)params.R * (size_t)params.N * sizeof(int);
    params.singleIntRowByteSize = (size_t)params.R * sizeof(int);
    params.replicaStatisticsByteSize = (size_t)params.R * sizeof(replicaStatistics);
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

    // Open output files
    sprintf(s, "C:/Users/marvelouslim/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_main.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    OPEN_FILE_CHECK(files.main_file, s);

    sprintf(s, "C:/Users/marvelouslim/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_agg_stats.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    OPEN_FILE_CHECK(files.agg_stats_file, s);

    sprintf(s, "C:/Users/marvelouslim/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_detailed_stats.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    OPEN_FILE_CHECK(files.detailed_stats_file, s);

    mainMemoryPointers host, device;
    // Allocate space on host
    host.spin = (int*)malloc(params.fullLatticeByteSize);
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

    int upper_energy = 2 * params.N + 2;
    int lower_energy = -2 * params.N - 2;

	int U = (params.heat ? lower_energy : upper_energy);	// U is energy ceiling

    //int i = 0;
    int no_replicas_try_again_counter = 0;
    int break_flg = 0;

    while ((U >= lower_energy && !params.heat) || (U <= upper_energy && params.heat)) {
        printf("U:\t%f between %d and %d; nSteps: %d;\n", 1.0 * U, upper_energy, lower_energy, params.nSteps);

        // Equilibrate
        clock_t t0 = clock();
        equilibrate(curand_states, device, params, U);
        clock_t t1 = clock();
        equilibrate_ticks += t1 - t0;

        // Copy energy to host
        clock_t t2 = clock();
        copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);
        clock_t t3 = clock();
        copy_d2h_ticks += t3 - t2;

        // Prepare resample arrays
        clock_t t4 = clock();
        double X = prepare_resample_arrays(host, params, &U);
        clock_t t5 = clock();
        prepare_resample_ticks += t5 - t4;

        if (X == 1) {
            if (no_replicas_try_again_counter < 10) {
                printf("try again number %d at U=%d\n", no_replicas_try_again_counter, U);
                //if X == 1 then U stays the same as in prepare_resample_arrays function, and we try again
                no_replicas_try_again_counter++;
                continue;
            }
            else {
                printf("ended with no replicas\n");
                break_flg = 1;
            }
        }

        // Family average size
        clock_t t6 = clock();
        double rho_t = calc_family_avg_sq_size(host, params, U);
        clock_t t7 = clock();
        family_avg_ticks += t7 - t6;

        print_main_data(files, U, X, rho_t);

        // Replica statistics
        clock_t t10 = clock();
        calc_replica_statistics(device, params, U);
        clock_t t11 = clock();
        replica_stats_ticks += t11 - t10;

        // Copy statistics to host
        clock_t t12 = clock();
        copyDeviceToHost(host.replica_statistics, device.replica_statistics, params.replicaStatisticsByteSize);
        clock_t t13 = clock();
        copy_d2h_ticks += t13 - t12;

        print_agg_stats(host, params, files, U);
        print_detailed_stats(host, params, files, U, 100);

        //copyDeviceToHost(host.spin, device.spin, params.R * sizeof(int));
        //print_replica_row(host.E, params, 1);
        //print_spin_sample(host.spin, 0, params);

        if (break_flg)
            break;

        // here goes actual resample - copy update array and update replicas
        clock_t t14 = clock();
        copyHostToDevice(device.update, host.update, params.singleIntRowByteSize);
        clock_t t15 = clock();
        copy_h2d_ticks += t15 - t14;

        clock_t t16 = clock();
        update_replicas(device, params);
        clock_t t17 = clock();
        update_replicas_ticks += t17 - t16;
    }



    fclose(files.main_file);
    fclose(files.agg_stats_file);
    fclose(files.detailed_stats_file);

    FREE_HOST_DEVICE(host.spin, device.spin);
    FREE_HOST_DEVICE(host.E, device.E);
    FREE_HOST_DEVICE(host.replica_statistics, device.replica_statistics);
    FREE_HOST_DEVICE(host.O, device.O);
    FREE_HOST_DEVICE(host.update, device.update);
    FREE_HOST_DEVICE(host.replica_family, device.replica_family);
    CUDA_CHECK(cudaFree(curand_states));


    clock_t global_end = clock();
    double global_time_spend = (double)(global_end - global_start) / CLOCKS_PER_SEC;
    double equilibrate_time_spend = (double)equilibrate_ticks / CLOCKS_PER_SEC;
    double prepare_resample_time_spend = (double)prepare_resample_ticks / CLOCKS_PER_SEC;
    double family_avg_time_spend = (double)family_avg_ticks / CLOCKS_PER_SEC;
    double replica_stats_time_spend = (double)replica_stats_ticks / CLOCKS_PER_SEC;
    double update_replicas_time_spend = (double)update_replicas_ticks / CLOCKS_PER_SEC;
    double copy_d2h_time_spend = (double)copy_d2h_ticks / CLOCKS_PER_SEC;
    double copy_h2d_time_spend = (double)copy_h2d_ticks / CLOCKS_PER_SEC;

    printf("\n=== TIMING SUMMARY ===\n");
    printf("Total time:        %.2fs\n", global_time_spend);
    printf("Equilibrate:       %.2fs (%.1f%%)\n", equilibrate_time_spend, 100.0 * equilibrate_time_spend / global_time_spend);
    printf("Prepare resample:  %.2fs (%.1f%%)\n", prepare_resample_time_spend, 100.0 * prepare_resample_time_spend / global_time_spend);
    printf("Family avg:       %.2fs (%.1f%%)\n", family_avg_time_spend, 100.0 * family_avg_time_spend / global_time_spend);
    printf("Replica stats:      %.2fs (%.1f%%)\n", replica_stats_time_spend, 100.0 * replica_stats_time_spend / global_time_spend);
    printf("Update replicas:  %.2fs (%.1f%%)\n", update_replicas_time_spend, 100.0 * update_replicas_time_spend / global_time_spend);
    printf("Copy D->H:        %.2fs (%.1f%%)\n", copy_d2h_time_spend, 100.0 * copy_d2h_time_spend / global_time_spend);
    printf("Copy H->D:        %.2fs (%.1f%%)\n", copy_h2d_time_spend, 100.0 * copy_h2d_time_spend / global_time_spend);
    printf("====================\n");

    return 0;
}


