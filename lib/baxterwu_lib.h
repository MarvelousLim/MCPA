#pragma once


#ifdef DLL_EXPORT
    #define DECLSPEC __declspec(dllexport)
#else
    #define DECLSPEC __declspec(dllimport)
#endif


#define GPU_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#else
// Forward declarations for MSVC
    struct curandStatePhilox4_32_10_t;
    enum cudaError;
#endif


// data types
struct neibors_indexes {
    int right;
    int down;
    int diag;
};
struct neibors {
    int right;
    int down;
    int diag;
};
struct Params {
    int L;
    int N;
    int R;
    int seed;
    int blocks;
    int threads;
    int nSteps;
    size_t fullLatticeByteSize;
    bool heat;
};
struct mainMemoryPointers {
    int* spin;
    int* E;
};
enum initializePopulationMode {RANDOM, BY_SUBLATTICE};



// functions
DECLSPEC inline void gpuAssert(cudaError code, const char* file, int line, bool abort = true);
DECLSPEC bool between(float x, float a, float b);
DECLSPEC void printSpinSample(int* s, int L, int N, int r);

DECLSPEC struct neibors_indexes SLF(int j, int L, int N);


#ifdef __CUDACC__
    DECLSPEC __device__ int suggestSpin(curandStatePhilox4_32_10_t* state, int r);
    DECLSPEC __global__ void initializePopulation(curandStatePhilox4_32_10_t* state, int* s, int N, initializePopulationMode mode, int s_a = 0, int s_b = 0, int s_c = 0);
#endif




