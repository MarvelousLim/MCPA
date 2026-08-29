#pragma once

#include "ising1d_runtime.h"

#include <curand_kernel.h>

__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, int seed);
__global__ void initializePopulation(curandStatePhilox4_32_10_t* state,
                                     char* spins, int N, int q);
__global__ void deviceEnergy(char* spins, int* energy, int L, int N);
__global__ void equilibrate(curandStatePhilox4_32_10_t* state,
                            char* spins, int* energy,
                            int L, int N, int R, int q,
                            int nSteps, int U, int* flip_counts);
IsingResampleResult resample(int* energy, int* order, int* update,
                             int* family, int R, int* U, bool heat);
__global__ void updateReplicas(char* spins, int* energy, int* update, int N);
