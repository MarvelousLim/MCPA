#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>

#ifdef __CUDACC__
    #define BC_HOST_DEVICE __device__ __host__
#else
    #define BC_HOST_DEVICE
#endif

#ifdef _WIN32
    #ifdef DLL_EXPORT
        #define DECLSPEC __declspec(dllexport)
    #else
        #define DECLSPEC __declspec(dllimport)
    #endif
#else
    #define DECLSPEC
#endif

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

/* Outward slack on int stop band ±(2N + |D|N) */
#define BC_ENERGY_STOP_BUFFER 4

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#endif


/* Internal int energy: D_denum*e_j + D_num*e_delta (single source of truth). */
static inline BC_HOST_DEVICE int bc_energy_int(int e_j, int e_delta,
                                                int D_num, int D_denum) {
    return D_denum * e_j + D_num * e_delta;
}

/* Human-readable E = e_j + (D_num/D_denum)*e_delta — output only. */
static inline BC_HOST_DEVICE double energy_physical(int e_j, int e_delta,
                                                     int D_num, int D_denum) {
    return (double)e_j + (double)D_num / (double)D_denum * (double)e_delta;
}

/* Uniform proposal over {sigma' : sigma' != sigma_old}, spin in {-1,0,+1}. */
static inline BC_HOST_DEVICE int bc_propose_spin(int sigma_old, unsigned int u) {
    int idx_old = sigma_old + 1;
    int step    = 1 + (int)(u & 1u);
    return (idx_old + step) % 3 - 1;
}

/* Accepted MC steps as percent of N*nSteps proposals in one equilibrate call. */
static inline double bc_acceptance_pct(int flip_count, int N, int nSteps) {
    return 100.0 * (double)flip_count / (double)((long long)N * nSteps);
}


/* ── Geometry ───────────────────────────────────────────────────────────────── */
struct neighborsIndexes {
    int left;
    int right;
    int up;
    int down;
};
struct neighborsValues {
    int left;
    int right;
    int up;
    int down;
};


/* ── Model parameters ────────────────────────────────────────────────────────── */
struct Params {
    int   L;
    int   N;
    int   R;
    int   seed;
    int   blocks;
    int   threads;
    int   nSteps;
    int   D_num;       /* crystal field Δ = D_num / D_denum */
    int   D_denum;
    bool  heat;        /* false = cooling (ceiling), true = heating (floor) */
    int   U_stop_cool;
    int   U_stop_heat;
    size_t fullLatticeByteSize;
    size_t singleIntRowByteSize;
    size_t replicaStatisticsByteSize;
};


/* ── Per-replica statistics (field-mixing / detailed_stats output) ──────────── */
struct replicaStatistics {
    int flip_count;   /* accepted moves in last equilibrate block */
    int e_j;          /* bond energy part: E_J = -Σ_{<ij>} σ_i σ_j  (already /2) */
    int e_delta;      /* crystal field:    E_Δ = Σ_i σ_i² */
    int m;            /* magnetisation:   M = Σ_i σ_i */
};

/* Portable host-side parent-selection RNG.  Its complete state is small enough
 * to checkpoint; the resampling law remains uniform-with-replacement modulo
 * the survivor count. */
struct ResamplingRngState {
    uint64_t state;
    uint64_t stream;
};


/* ── Memory layout ───────────────────────────────────────────────────────────── */
struct mainMemoryPointers {
    int*   spin;              /* σ ∈ {-1,0,+1}, layout [replica][site] */
    int*   e_j;               /* per-replica bond energy component */
    int*   e_delta;           /* per-replica crystal-field component */
    int*   O;                 /* sort permutation */
    int*   update;            /* resample map after culling */
    int*   replica_family;    /* genealogy tracker (rho_t) */
    struct replicaStatistics* replica_statistics;
};


/* ── Output files ────────────────────────────────────────────────────────────── */
struct Files {
    FILE* main_file;           /* E  culling_factor  rho_t */
    FILE* agg_stats_file;      /* E  n_culled  flip_rate  e_j  e_delta  m */
    FILE* detailed_stats_file; /* E  flip_rate  e_j  e_delta  m */
};

struct GpuMetadata {
    char name[256];
    int compute_major;
    int compute_minor;
    int cuda_runtime_version;
    int cuda_driver_version;
    uint64_t total_memory_bytes;
    uint64_t free_memory_before_setup_bytes;
    uint64_t free_memory_after_setup_bytes;
};


static inline int bc_replica_energy(const struct mainMemoryPointers* host, int i,
                                     const struct Params* params) {
    return bc_energy_int(host->e_j[i], host->e_delta[i],
                         params->D_num, params->D_denum);
}


enum initializePopulationMode { random_pop };
enum statisticsMode { aggregated = 0, detailed = 1 };


DECLSPEC void gpu_assert(int code, const char* file, int line, bool abort = true);

DECLSPEC BC_HOST_DEVICE struct neighborsIndexes SLF(int j, struct Params params);

/* Pure local model terms are host-callable so CPU tests can compare them with
 * an independent square-lattice oracle.  CUDA uses the same definitions. */
DECLSPEC BC_HOST_DEVICE int local_energy_j(int sigma, struct neighborsValues nv);
DECLSPEC BC_HOST_DEVICE int local_energy_delta(int sigma);

DECLSPEC void* setup_curand_states(struct Params params);
DECLSPEC size_t curand_states_byte_size(struct Params params);

DECLSPEC void initialize_population(void* curand_states, struct mainMemoryPointers device,
                                     struct Params params);
DECLSPEC void initialize_update_arrays(struct mainMemoryPointers host, struct Params params);
DECLSPEC void initialize_resampling_rng(int seed);
DECLSPEC struct ResamplingRngState get_resampling_rng_state();
DECLSPEC void set_resampling_rng_state(struct ResamplingRngState state);

DECLSPEC void copyHostToDevice(void* dst, void* src, size_t size);
DECLSPEC void copyDeviceToHost(void* dst, void* src, size_t size);

DECLSPEC int  parse_D_from_string(const char* s, int* D_num, int* D_denum);
DECLSPEC int  check_bc_energy_overflow(struct Params* params);
DECLSPEC void compute_U_stop(struct Params* params);
DECLSPEC void format_D_for_path(int D_num, int D_denum, char* buf, size_t buf_size);
DECLSPEC double energy_physical_at_U(struct mainMemoryPointers host, struct Params params,
                                    int U);

DECLSPEC void calc_device_energy(struct mainMemoryPointers device, struct Params params);
DECLSPEC void equilibrate(void* curand_states, struct mainMemoryPointers device,
                          struct Params params, int U);
DECLSPEC void calc_replica_statistics(struct mainMemoryPointers device, struct Params params,
                                       int U);
DECLSPEC void update_replicas(struct mainMemoryPointers device, struct Params params);

DECLSPEC void swap_order(int* O, int i, int j);
DECLSPEC void quicksort(struct mainMemoryPointers host, struct Params params,
                        int left, int right, int direction);
DECLSPEC double prepare_resample_arrays(struct mainMemoryPointers host, struct Params params,
                                         int* U, int* n_culled_exact = nullptr);
DECLSPEC double calc_family_concentration(const int* family_ids, int R);
DECLSPEC double calc_family_avg_sq_size(struct mainMemoryPointers host, struct Params params);

DECLSPEC void initialize_print(struct Files files);
DECLSPEC void print_main_data(struct Files files, double E_phys, double X, double rho_t,
                              int U, int D_num, int D_denum, int n_culled_exact,
                              double equilibrate_seconds,
                              double pre_family_concentration,
                              double post_family_concentration,
                              const struct GpuMetadata* metadata);
DECLSPEC void print_detailed_stats(struct mainMemoryPointers host, struct Params params,
                                    struct Files files, int U,
                                    const int* measured_family_ids, int detail_cap);
DECLSPEC void print_agg_stats(struct mainMemoryPointers host, struct Params params,
                               struct Files files, int U,
                               const int* measured_family_ids);
