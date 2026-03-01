#pragma once


#include "baxterwu_lib.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }


int main()
{
    std::cout << "between section" << std::endl;
    std::cout << "0 <= 1 <= 2: " << between(1, 0, 2) << std::endl;
    std::cout << "2 >= 1 >= 0: " << between(1, 2, 0) << std::endl;
    std::cout << "0 <= -1 <= 2: " << between(-1, 0, 2) << std::endl;

    std::cout << "neibors_indexes section" << std::endl;
    struct neibors_indexes neibors_indexes_test = SLF(0, 10, 100);
    std::cout << "neibors_indexes_test: "
        << neibors_indexes_test.right << ' '
        << neibors_indexes_test.down << ' '
        << neibors_indexes_test.diag << std::endl;

    std::cout << "deviceSpin test" << std::endl;
    Params params;
    params.L = 10;
    params.N = params.L * params.L;
    params.seed = 0;
    params.blocks = 1;
    params.threads = 1;
    params.R = params.blocks * params.threads;
    params.nSteps = 1;
    params.fullLatticeByteSize = params.R * params.N * sizeof(int);
    params.heat = false;

    std::cout << "params: "
        << params.L << ' '
        << params.N << ' '
        << params.R << ' '
        << params.seed << ' '
        << params.blocks << ' '
        << params.threads << ' '
        << params.nSteps << ' '
        << params.fullLatticeByteSize << ' ' << std::endl;


    mainMemoryPointers host, device;
	// Allocate space on host
	host.spin = (int*)malloc(params.fullLatticeByteSize);
    CUDA_CHECK(cudaMalloc((void**)&device.spin, params.fullLatticeByteSize));


    free(host.spin);
    CUDA_CHECK(cudaFree(device.spin));
    return 0;
}
