#pragma once

#include "mem_estimate.h"

#include <stddef.h>
#include <stdint.h>

static inline bool potts_supported_q(int q) {
    return q == 2 || q == 3 || q == 4;
}

/* Strict ceiling/floor walks must begin outside the complete spectrum. */
static inline int potts_cooling_start_U() {
    return 1;
}

static inline int potts_heating_start_U(int N) {
    return -2 * N - 1;
}

#ifdef __CUDACC__
#define POTTS_HOST_DEVICE __host__ __device__
#else
#define POTTS_HOST_DEVICE
#endif

struct PottsParams {
    int seed;
    int L;
    int N;
    int blocks;
    int threads;
    int R;
    int nSteps;
    int q;
    bool heat;
    size_t fullLatticeByteSize;
    size_t singleIntRowByteSize;
};

struct PottsMemoryPointers {
    char* spin;
    int* E;
    int* update;
    uint64_t* accepted_flips;
};

struct PottsResamplingRngState {
    uint64_t state;
    uint64_t stream;
};

enum PottsResampleStatus {
    POTTS_RESAMPLE_OK = 0,
    POTTS_RESAMPLE_NO_NEXT_SHELL = 1,
    POTTS_RESAMPLE_TERMINAL_FULL_CULL = 2,
};

struct PottsResampleResult {
    PottsResampleStatus status;
    int old_U;
    int new_U;
    int n_cull;
    double culling_fraction;
};

struct neighborsIndexes {
    int up;
    int down;
    int left;
    int right;
};

struct neighborsValues {
    char up;
    char down;
    char left;
    char right;
};

POTTS_HOST_DEVICE struct neighborsIndexes SLF(int j, int L, int N);
POTTS_HOST_DEVICE int local_energy(char currentSpin, struct neighborsValues neighbors);
void swap_order(int* order, int i, int j);
void quicksort(int* E, int* order, int left, int right, int direction);

/* Potts host: E, update, family, order; device: E, update + Philox. */
struct MemEstimate estimate_potts_setup_memory(int N, int R);

void gpu_assert(int code, const char* file, int line, bool abort = true);
void* setup_curand_states(struct PottsParams params);
void initialize_resampling_rng(int seed);
struct PottsResamplingRngState get_resampling_rng_state();
void set_resampling_rng_state(struct PottsResamplingRngState state);
void initialize_population(void* curand_states, struct PottsMemoryPointers device,
                           struct PottsParams params);
void calc_device_energy(struct PottsMemoryPointers device, struct PottsParams params);
void equilibrate(void* curand_states, struct PottsMemoryPointers device,
                 struct PottsParams params, int U);
void update_replicas(struct PottsMemoryPointers device, struct PottsParams params);

PottsResampleResult resample(int* E, int* order, int* update,
                             int* replica_family, int R, int* U, bool heat);
