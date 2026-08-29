#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

// Platform-specific export/import macros
#ifdef _WIN32
    // Windows
    #ifdef DLL_EXPORT
        #define DECLSPEC __declspec(dllexport)
    #else
        #define DECLSPEC __declspec(dllimport)
    #endif
#else
    // Linux/Unix/Mac - no declspec needed for executables
    #define DECLSPEC  // Empty on Linux - we're building an executable, not a shared lib
#endif

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
    #define MCPA_HOST_DEVICE __host__ __device__
#else
    #define MCPA_HOST_DEVICE
#endif


// data types
// Function declarations (Added)
// Returns the local energy (negative sum of interactions)
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
    // Branch-invariant trace structure factors at reciprocal indices
    // (1,0), (0,1), and (1,-1), divided by N.
    double order_structure_factor[3];
};

struct FamilyMetrics {
    int count;
    int max_size;
    double max_fraction;
    double shannon_entropy;
    double effective_shannon;
    double simpson_concentration;
};

// Explicit host-side RNG for replica-parent selection.  Unlike libc rand(),
// this complete state is portable and can be included in a checkpoint.
struct ResamplingRngState {
    uint64_t state;
    uint64_t stream;
};

struct RunGpuMetadata {
    char name[256];
    double compute_capability;
    uint64_t total_memory_bytes;
    uint64_t free_memory_before_setup_bytes;
    uint64_t free_memory_after_setup_bytes;
    int cuda_driver_version;
    int cuda_runtime_version;
};

struct AggregateDiagnostics {
    int family_count_at_E;
    double family_max_fraction_at_E;
    double m_s_mean;
    double p_s_mean;
    double order_sf_k[3];
    double family_var_mean_m_s;
    double family_var_mean_p_s;
    double family_var_mean_order_sf0;
    double family_var_mean_order_sf_kmin;
    double family_cov_mean_order_sf0_kmin;
};

struct mainMemoryPointers {
    int* spin;
    int* E; // too important to put into struct
#if defined(MCPA_BW_NEIGHBOR_TABLE)
    // Candidate-only, read-only table populated from SLF once per lattice.
    struct neiborsIndexes* equilibration_neighbors;
#endif
    // update pointers
    int* O; // sort order for culling and resampling; we sort replicas in place with O and only then relocate em
    int* update; // update indexes
    int* replica_family;
    // Device-only, read-only phase tables. Layout is [q * N + site], q=0,1,2.
    float* fourier_phase_cos;
    float* fourier_phase_sin;
    //
    struct replicaStatistics* replica_statistics;
};
struct Files {
    FILE* main_file; // for culling factor and rho_t
    FILE* agg_stats_file;
    FILE* detailed_stats_file;
};
enum initializePopulationMode { random_pop, by_sublattice, strips };
enum testMode { ALL, between_test, params_test, slf_test, lattice_setup, local_energy_test, resample_test, calc_replica_statistics_test };
enum statisticsMode { aggregated = 0, detailed = 1, spin_samples = 2 }; // detailed includes aggregated and spin_samples includes both
enum equlibrateMode { single_step, normal };


// functions
DECLSPEC void swap(int* A, int i, int j);
DECLSPEC void quicksort(struct mainMemoryPointers host, int left, int right, int direction);
DECLSPEC MCPA_HOST_DEVICE int local_energy(int currentSpin, struct neiborsValues n);
DECLSPEC void gpu_assert(int code, const char* file, int line, bool abort = true);
DECLSPEC bool between(float x, float a, float b);
DECLSPEC void print_spin_sample(int* s, int r, struct Params params);
DECLSPEC void print_replica_row(int* e, struct Params params, int limit);
// Pure geometry/math logic shared by production CUDA and host-side tests.
DECLSPEC MCPA_HOST_DEVICE struct neiborsIndexes SLF(int j, struct Params params);
DECLSPEC void* setup_curand_states(struct Params params);
DECLSPEC size_t curand_states_byte_size(struct Params params);
DECLSPEC void initialize_resampling_rng(int seed);
DECLSPEC struct ResamplingRngState get_resampling_rng_state();
DECLSPEC void set_resampling_rng_state(struct ResamplingRngState state);
DECLSPEC void initialize_population(void* curand_states, struct mainMemoryPointers device, struct Params params, initializePopulationMode mode, int s_a = 0, int s_b = 0, int s_c = 0);
DECLSPEC void initialize_update_arrays(struct mainMemoryPointers host, struct Params params);
DECLSPEC void copyHostToDevice(void* dst, void* src, size_t size);
DECLSPEC void copyDeviceToHost(void* dst, void* src, size_t size);
DECLSPEC void calc_device_energy(struct mainMemoryPointers device, struct Params params);
DECLSPEC void equilibrate(void* curand_states, struct mainMemoryPointers device, struct Params params, int U);
DECLSPEC void initialize_fourier_phases(struct mainMemoryPointers device, struct Params params);
DECLSPEC void calc_replica_statistics(struct mainMemoryPointers device, struct Params params, int U);
DECLSPEC double prepare_resample_arrays(struct mainMemoryPointers host, struct Params params, int* U,
                                        int* culled_replica_number = nullptr);
DECLSPEC double calc_family_avg_sq_size(struct mainMemoryPointers host, struct Params params, int U);
DECLSPEC struct FamilyMetrics calc_family_metrics(const int* family_ids, int R);
DECLSPEC struct AggregateDiagnostics calc_aggregate_diagnostics(
    struct mainMemoryPointers host, struct Params params, int U,
    const int* measured_replica_family);
// Fixed, platform-independent key used to select the smallest hashes among
// replicas at one energy.  The key depends only on run seed, direction,
// energy, and replica index; it never depends on an observable or family label.
DECLSPEC uint64_t detailed_sample_hash(int seed, bool heat, int U, int replica_id);
DECLSPEC void initialize_print(struct Files files);
DECLSPEC void print_main_data(struct Files files, int U, double X, double rho_t,
                              int culled_replica_number, double equilibrate_seconds,
                              const struct FamilyMetrics& family_metrics,
                              const struct RunGpuMetadata* run_gpu_metadata = nullptr,
                              int equilibrate_ceiling = 0,
                              int equilibrate_nsteps = 0,
                              double population_acceptance_ratio = NAN,
                              const char* nsteps_policy = "fixed");
DECLSPEC int print_detailed_stats(struct mainMemoryPointers host, struct Params params,
                                 struct Files files, int U, int limit,
                                 const int* measured_replica_family = nullptr);
DECLSPEC void print_agg_stats(struct mainMemoryPointers host, struct Params params,
                             struct Files files, int U,
                             const int* measured_replica_family = nullptr);
DECLSPEC void update_replicas(struct mainMemoryPointers device, struct Params params);
