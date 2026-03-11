#pragma once


#include "baxterwu_lib.h"
//#include <cuda.h>
#include <cuda_runtime.h>
//#include <curand_kernel.h>
#include <iostream>

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }


int main()
{
    testMode test_mode = calc_replica_statistics_test;
    statisticsMode statistics_mode = detailed;
    equlibrateMode equlibrate_mode = single_step;

    if (test_mode == between_test){
        std::cout << "between section" << std::endl;
        std::cout << "0 <= 1 <= 2: " << between(1, 0, 2) << std::endl;
        std::cout << "2 >= 1 >= 0: " << between(1, 2, 0) << std::endl;
        std::cout << "0 <= -1 <= 2: " << between(-1, 0, 2) << std::endl;
    }

    Params params;
    params.L = 10;
    params.N = params.L * params.L;
    params.seed = 1;
    params.blocks = 1;
    params.threads = 1;
    params.R = params.blocks * params.threads;
    params.nSteps = 1;
    params.fullLatticeByteSize = params.R * params.N * sizeof(int);
    params.singleIntRowByteSize = params.R * sizeof(int);
    params.replicaStatisticsByteSize = params.R * sizeof(replicaStatistics);
    params.heat = false;

    //int U;

	char s[150];
	const char* prefix = "test";
    const char* heating = params.heat ? "Heating" : "";

    struct Files files;
	sprintf(s, "C:/Users/MarvelousNote3/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_main.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
	files.main_file = fopen(s, "w");
    printf(s);
    sprintf(s, "C:/Users/MarvelousNote3/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_agg_stats.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    files.agg_stats_file = fopen(s, "w");
    printf(s);
    sprintf(s, "C:/Users/MarvelousNote3/Yandex.Disk/ASAV/Analytics/datasets/%s/2DBaxterWu%s_N%d_R%d_nSteps%d_run%d_detailed_stats.txt", prefix, heating, params.N, params.R, params.nSteps, params.seed);
    files.detailed_stats_file = fopen(s, "w");
    printf(s);

    
    if (test_mode == params_test){
        std::cout << "params: "
            << params.L << ' '
            << params.N << ' '
            << params.R << ' '
            << params.seed << ' '
            << params.blocks << ' '
            << params.threads << ' '
            << params.nSteps << ' '
            << params.fullLatticeByteSize << ' '
            << params.singleIntRowByteSize << ' '
            << std::endl;
    }

    struct neiborsIndexes neibors_indexes_test = SLF(0, params);

    if (test_mode == ALL or test_mode == slf){
        std::cout << "neibors_indexes section" << std::endl;
        std::cout << "neibors_indexes_test: "
            << neibors_indexes_test.right << ' '
            << neibors_indexes_test.left << ' '
            << neibors_indexes_test.up << ' '
            << neibors_indexes_test.down << ' '
            << neibors_indexes_test.diag_right << ' '
            << neibors_indexes_test.diag_left << std::endl;
    }


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

    // setup lattics test
    if (test_mode == ALL or test_mode == lattice_setup) {
        {
            std::cout << "deviceSpin random test" << std::endl;

            initialize_population(curand_states, device, params, random, 0, -1, 1);

            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);

            print_spin_sample(host.spin, 0, params);

            calc_device_energy(device, params);

            copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);

            print_replica_row(host.E, params);

        }


        {
            std::cout << "deviceSpin 0 0 0 test" << std::endl;

            initialize_population(curand_states, device, params, by_sublattice, 0, 0, 0);

            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);

            print_spin_sample(host.spin, 0, params);

            calc_device_energy(device, params);

            copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);

            print_replica_row(host.E, params);

        }
    }


    //SLF tests
    if (test_mode == ALL or test_mode == slf) {
        {
            std::cout << "SLF test: middle" << std::endl;

            initialize_population(curand_states, device, params, by_sublattice, 0, 0, 0);

            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);

            neibors_indexes_test = SLF(45, params);

            host.spin[neibors_indexes_test.right] = 3;
            host.spin[neibors_indexes_test.left] = 6;
            host.spin[neibors_indexes_test.up] = 2;
            host.spin[neibors_indexes_test.down] = 5;
            host.spin[neibors_indexes_test.diag_left] = 1;
            host.spin[neibors_indexes_test.diag_right] = 4;


            print_spin_sample(host.spin, 0, params);

        }


        {
            std::cout << "SLF test: angle" << std::endl;

            initialize_population(curand_states, device, params, by_sublattice, 0, 0, 0);

            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);

            neibors_indexes_test = SLF(0, params);

            host.spin[neibors_indexes_test.right] = 3;
            host.spin[neibors_indexes_test.left] = 6;
            host.spin[neibors_indexes_test.up] = 2;
            host.spin[neibors_indexes_test.down] = 5;
            host.spin[neibors_indexes_test.diag_left] = 1;
            host.spin[neibors_indexes_test.diag_right] = 4;


            print_spin_sample(host.spin, 0, params);

        }

        {
            std::cout << "SLF test: angle" << std::endl;

            initialize_population(curand_states, device, params, by_sublattice, 0, 0, 0);

            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);

            neibors_indexes_test = SLF(99, params);

            host.spin[neibors_indexes_test.right] = 3;
            host.spin[neibors_indexes_test.left] = 6;
            host.spin[neibors_indexes_test.up] = 2;
            host.spin[neibors_indexes_test.down] = 5;
            host.spin[neibors_indexes_test.diag_left] = 1;
            host.spin[neibors_indexes_test.diag_right] = 4;


            print_spin_sample(host.spin, 0, params);

        }

        {
            std::cout << "SLF test: angle" << std::endl;

            initialize_population(curand_states, device, params, by_sublattice, 0, 0, 0);

            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);

            neibors_indexes_test = SLF(10, params);

            host.spin[neibors_indexes_test.right] = 3;
            host.spin[neibors_indexes_test.left] = 6;
            host.spin[neibors_indexes_test.up] = 2;
            host.spin[neibors_indexes_test.down] = 5;
            host.spin[neibors_indexes_test.diag_left] = 1;
            host.spin[neibors_indexes_test.diag_right] = 4;


            print_spin_sample(host.spin, 0, params);

        }

        {
            std::cout << "SLF test: angle" << std::endl;

            initialize_population(curand_states, device, params, by_sublattice, 0, 0, 0);

            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);

            neibors_indexes_test = SLF(90, params);

            host.spin[neibors_indexes_test.right] = 3;
            host.spin[neibors_indexes_test.left] = 6;
            host.spin[neibors_indexes_test.up] = 2;
            host.spin[neibors_indexes_test.down] = 5;
            host.spin[neibors_indexes_test.diag_left] = 1;
            host.spin[neibors_indexes_test.diag_right] = 4;


            print_spin_sample(host.spin, 0, params);

        }
    }


    // Local Energy (changes) tests
    if (test_mode == ALL or test_mode == local_energy_test) {
        std::cout << "local energy test" << std::endl;
        initialize_population(curand_states, device, params, by_sublattice, 1, 1, 1);
        calc_device_energy(device, params);


        for (int i = 0; i < 10; i++) {
            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);
            copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);

            print_spin_sample(host.spin, 0, params);
            print_replica_row(host.E, params);

            equilibrate(curand_states, device, params, 1000, equlibrate_mode); // single step
        }

    }


    if (test_mode == ALL or test_mode == resample_test) {
        std::cout << "replica statistics test" << std::endl;
        initialize_population(curand_states, device, params, by_sublattice, 1, 1, 1);
        calc_device_energy(device, params);

        for (int i = 0; i < 10; i++) {
            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);
            copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);

            equilibrate(curand_states, device, params, 1000, equlibrate_mode); // single step

            print_spin_sample(host.spin, 0, params);
            print_replica_row(host.E, params);

            //prepare_resample_arrays(host, params, files, &U);
        }

    }


    if (test_mode == ALL or test_mode == calc_replica_statistics_test) {
        std::cout << "replica statistics test" << std::endl;
        initialize_population(curand_states, device, params, by_sublattice, 1, -1, 0);
        calc_device_energy(device, params);


        for (int i = 0; i < 10; i++) {
            copyDeviceToHost(host.spin, device.spin, params.fullLatticeByteSize);
            copyDeviceToHost(host.E, device.E, params.singleIntRowByteSize);

            print_spin_sample(host.spin, 0, params);
            print_replica_row(host.E, params);

            equilibrate(curand_states, device, params, 1000, equlibrate_mode); // single step
        
            calc_replica_statistics(device, params, 0);
            copyDeviceToHost(host.replica_statistics, device.replica_statistics, params.replicaStatisticsByteSize);

            print_detailed_stats(host, params, files, 0);
            print_agg_stats(host, params, files, 0);

        }

    }

    
    fclose(files.main_file);
    fclose(files.agg_stats_file);
    fclose(files.detailed_stats_file);


    free(host.spin);
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

    return 0;
}
