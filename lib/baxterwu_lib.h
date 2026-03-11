#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef DLL_EXPORT
    #define DECLSPEC __declspec(dllexport)
#else
    #define DECLSPEC __declspec(dllimport)
#endif

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#else
// Forward declarations for MSVC
    enum cudaError;
#endif


// data types
struct neiborsIndexes {
    int left;
    int right;
    int up;
    int down;
    int diag_left;
    int diag_right;
};
struct neiborsValues {
    int left;
    int right;
    int up;
    int down;
    int diag_left;
    int diag_right;
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
    size_t singleIntRowByteSize;
    size_t replicaStatisticsByteSize;
    bool heat;
};
struct replicaStatistics {
    int magnetization[3];
    int polarization[3];
    int flip_count;
};

struct mainMemoryPointers {
    int* spin;
    int* E; // too important to put into struct
    // update pointers
    int* O; // sort order for culling and resampling; we sort replicas in place with O and only then relocate em
    int* update; // update indexes
    int* replica_family;
    //
    struct replicaStatistics* replica_statistics;
};
struct Files {
    FILE* main_file; // for culling factor and rho_t
    FILE* agg_stats_file;
    FILE* detailed_stats_file;
};
enum initializePopulationMode {random, by_sublattice};
enum testMode { ALL, between_test, params_test, slf, lattice_setup, local_energy_test, resample_test, calc_replica_statistics_test };
enum statisticsMode { aggregated = 0, detailed = 1, spin_samples = 2 }; // detailed includes aggregated and spin_samples includes both
enum equlibrateMode { single_step, normal };


// functions
DECLSPEC void gpu_assert(cudaError code, const char* file, int line, bool abort = true);
DECLSPEC bool between(float x, float a, float b);
DECLSPEC void print_spin_sample(int* s, int r, struct Params params);
DECLSPEC void print_replica_row(int* e, struct Params params);
DECLSPEC struct neiborsIndexes SLF(int j, struct Params params);
DECLSPEC void* setup_curand_states(struct Params params);
DECLSPEC void initialize_population(void* curand_states, struct mainMemoryPointers device, struct Params params, initializePopulationMode mode, int s_a = 0, int s_b = 0, int s_c = 0);
DECLSPEC void initialize_update_arrays(struct mainMemoryPointers host, struct Params params);
DECLSPEC void copyHostToDevice(void* dst, void* src, size_t size);
DECLSPEC void copyDeviceToHost(void* dst, void* src, size_t size);
DECLSPEC void calc_device_energy(struct mainMemoryPointers device, struct Params params);
DECLSPEC void equilibrate(void* curand_states, struct mainMemoryPointers device, struct Params params, int U, enum equlibrateMode equlibrate_mode);
DECLSPEC void calc_replica_statistics(struct mainMemoryPointers device, struct Params params, int U);
DECLSPEC double prepare_resample_arrays(struct mainMemoryPointers host, struct Params params, int* U);
DECLSPEC double calc_family_avg_sq_size(struct mainMemoryPointers host, struct Params params, int U);
DECLSPEC void initialize_print(struct Files files);
DECLSPEC void print_main_data(struct Files files, int U, double X, double rho_t);
DECLSPEC void print_detailed_stats(struct mainMemoryPointers host, struct Params params, struct Files files, int U);
DECLSPEC void print_agg_stats(struct mainMemoryPointers host, struct Params params, struct Files files, int U);
DECLSPEC void update_replicas(struct mainMemoryPointers device, struct Params params);

