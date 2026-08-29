#include "blumeCapel_lib.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

namespace {

ResamplingRngState resampling_rng{0, 1};

uint32_t next_resampling_random() {
    const uint64_t old_state = resampling_rng.state;
    resampling_rng.state = old_state * 6364136223846793005ULL
                         + (resampling_rng.stream | 1ULL);
    const uint32_t xorshifted = static_cast<uint32_t>(
        ((old_state >> 18U) ^ old_state) >> 27U);
    const uint32_t rotation = static_cast<uint32_t>(old_state >> 59U);
    return (xorshifted >> rotation)
         | (xorshifted << ((-rotation) & 31U));
}

} // namespace

DECLSPEC void gpu_assert(int code, const char* file, int line, bool abort) {
    cudaError_t err = (cudaError_t)code;
    if (err != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(code);
    }
}


// ── Geometry: square-lattice spin-lookup function ─────────────────────────────
// Square lattice with 4 neighbours (left, right, up, down), periodic BC.
DECLSPEC __host__ __device__ struct neighborsIndexes SLF(int j, struct Params params) {
    struct neighborsIndexes ni;
    int L = params.L;
    int x = j % L;
    int y = j / L;
    ni.left  = (x - 1 + L) % L + y * L;
    ni.right = (x + 1)     % L + y * L;
    ni.up    = x + ((y - 1 + L) % L) * L;
    ni.down  = x + ((y + 1)     % L) * L;
    return ni;
}

// Fetch neighbour spin values from device array.
__device__ __host__ struct neighborsValues SVLF(
        struct mainMemoryPointers mem, struct neighborsIndexes ni, long long rep_shift) {
    struct neighborsValues nv;
    nv.left  = mem.spin[ni.left  + rep_shift];
    nv.right = mem.spin[ni.right + rep_shift];
    nv.up    = mem.spin[ni.up    + rep_shift];
    nv.down  = mem.spin[ni.down  + rep_shift];
    return nv;
}


// ── Blume-Capel local energies ────────────────────────────────────────────────
// H = -J Σ_{<ij>} σ_i σ_j + Δ Σ_i σ_i²   with J=1.
//
// LOCAL BOND energy contribution for spin at site i (counts all 4 bonds touching i;
// each bond is counted twice in the total, so calc_device_energy divides by 2).
__device__ __host__ int local_energy_j(int sigma, struct neighborsValues nv) {
    return -sigma * (nv.left + nv.right + nv.up + nv.down);
}

// LOCAL CRYSTAL-FIELD energy for a single spin (σ²; no double-counting issue).
__device__ __host__ int local_energy_delta(int sigma) {
    return sigma * sigma;   /* 0 if sigma==0, 1 if sigma==±1 */
}


// ── D parsing (argv string → rational D_num/D_denum, never atof) ──────────────

static int gcd_int(int a, int b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { int t = b; b = a % b; a = t; }
    return a ? a : 1;
}

static void strip_trailing_zeros(char* frac) {
    int len = (int)strlen(frac);
    while (len > 0 && frac[len - 1] == '0') {
        frac[len - 1] = '\0';
        len--;
    }
}

DECLSPEC int parse_D_from_string(const char* s, int* D_num, int* D_denum) {
    if (!s || !D_num || !D_denum) return -1;

    const char* p = s;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0') return -1;

    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') p++;

    long long num = 0;
    long long den = 1;

    if (*p < '0' || *p > '9') return -1;
    while (*p >= '0' && *p <= '9') {
        num = num * 10 + (*p - '0');
        p++;
    }

    if (*p == '.') {
        p++;
        if (*p < '0' || *p > '9') return -1;
        while (*p >= '0' && *p <= '9') {
            num = num * 10 + (*p - '0');
            den *= 10;
            p++;
        }
    }

    while (*p == ' ' || *p == '\t') p++;
    if (*p != '\0') return -1;

    num *= sign;

    while (den > 1 && num % 10 == 0) {
        num /= 10;
        den /= 10;
    }

    int g = gcd_int((int)(num < 0 ? -num : num), (int)den);
    num /= g;
    den /= g;

    if (den == 0 || den > INT_MAX) return -1;
    if (num > INT_MAX || num < INT_MIN) return -1;

    *D_num   = (int)num;
    *D_denum = (int)den;
    return 0;
}

DECLSPEC void format_D_for_path(int D_num, int D_denum, char* buf, size_t buf_size) {
    if (D_denum == 1) {
        snprintf(buf, buf_size, "%d", D_num);
        return;
    }
    int sign = (D_num < 0) ? -1 : 1;
    int an   = (D_num < 0) ? -D_num : D_num;
    int whole = an / D_denum;
    int rem   = an % D_denum;

    if (rem == 0) {
        snprintf(buf, buf_size, "%d", sign * whole);
        return;
    }

    char frac[32];
    int flen = 0;
    int r = rem;
    int d = D_denum;
    while (r != 0 && flen < (int)sizeof(frac) - 1) {
        r *= 10;
        frac[flen++] = (char)('0' + r / d);
        r %= d;
    }
    frac[flen] = '\0';
    strip_trailing_zeros(frac);

    if (sign < 0)
        snprintf(buf, buf_size, "-%d.%s", whole, frac);
    else
        snprintf(buf, buf_size, "%d.%s", whole, frac);
}

// ── Loop exit bounds and overflow check (long long used here only) ────────────

DECLSPEC int check_bc_energy_overflow(struct Params* p) {
    long long N = p->N;
    long long Dd = p->D_denum;
    long long Dn = p->D_num;
    long long abs_Dn = (Dn < 0) ? -Dn : Dn;

    if (p->D_denum == 0) {
        fprintf(stderr, "D_denum must not be zero\n");
        return -1;
    }

    if (Dd * 2 * N > INT_MAX || abs_Dn * N > INT_MAX) {
        fprintf(stderr, "Energy product overflow: D_denum*2N or |D_num|*N exceeds INT_MAX\n");
        return -1;
    }

    long long corners_ej[] = { -2 * N, 2 * N, -2 * N, 2 * N };
    long long corners_ed[] = { 0, 0, N, N };
    long long e_min = (long long)INT_MAX;
    long long e_max = (long long)INT_MIN;
    for (int c = 0; c < 4; c++) {
        long long e = Dd * corners_ej[c] + Dn * corners_ed[c];
        if (e < e_min) e_min = e;
        if (e > e_max) e_max = e;
    }
    if (e_min < INT_MIN || e_max > INT_MAX) {
        fprintf(stderr, "Corner energy out of int32 range\n");
        return -1;
    }

    long long stop_mag = 2 * Dd * N + abs_Dn * N;
    if (stop_mag + BC_ENERGY_STOP_BUFFER > INT_MAX) {
        fprintf(stderr, "stop_mag + buffer overflow\n");
        return -1;
    }
    return 0;
}

DECLSPEC void compute_U_stop(struct Params* params) {
    int abs_D_num = params->D_num >= 0 ? params->D_num : -params->D_num;
    int stop_mag  = 2 * params->D_denum * params->N + abs_D_num * params->N;
    params->U_stop_cool = -stop_mag - BC_ENERGY_STOP_BUFFER;
    params->U_stop_heat =  stop_mag + BC_ENERGY_STOP_BUFFER;
}

DECLSPEC double energy_physical_at_U(struct mainMemoryPointers host, struct Params params,
                                     int U) {
    for (int i = 0; i < params.R; i++) {
        if (bc_replica_energy(&host, i, &params) == U)
            return energy_physical(host.e_j[i], host.e_delta[i],
                                   params.D_num, params.D_denum);
    }
    /* U = D_denum*e_j + D_num*e_delta  =>  H = U/D_denum = energy_physical(0, 1, U, D_denum) */
    return energy_physical(0, 1, U, params.D_denum);
}


// ── RNG setup (Philox4-32, same as BaxterWu) ──────────────────────────────────
__global__ void setup_curand_kernel(curandStatePhilox4_32_10_t* state, int seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, state + id);
}

DECLSPEC void* setup_curand_states(struct Params params) {
    curandStatePhilox4_32_10_t* states = nullptr;
    printf("Allocating %d Philox random states...\n", params.R);
    CUDA_CHECK(cudaMalloc((void**)&states, params.R * sizeof(curandStatePhilox4_32_10_t)));
    setup_curand_kernel<<<params.blocks, params.threads>>>(states, params.seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    return (void*)states;
}

DECLSPEC size_t curand_states_byte_size(struct Params params) {
    return static_cast<size_t>(params.R) * sizeof(curandStatePhilox4_32_10_t);
}


// ── Memory helpers ─────────────────────────────────────────────────────────────
DECLSPEC void copyHostToDevice(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaDeviceSynchronize());
}
DECLSPEC void copyDeviceToHost(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ── Population initialisation ─────────────────────────────────────────────────
// Random spin configuration: σ ∈ {-1, 0, +1} uniformly.
__global__ void initialize_population_kernel(
        curandStatePhilox4_32_10_t* curand_states,
        struct mainMemoryPointers device, struct Params params) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    long long rep_shift = (long long)r * params.N;
    for (int k = 0; k < params.N; k++) {
        /* Map uniform random integer mod 3 to {-1, 0, +1} */
        int val = (int)(curand(&curand_states[r]) % 3) - 1;
        device.spin[rep_shift + k] = val;
    }
}

DECLSPEC void initialize_population(void* curand_states, struct mainMemoryPointers device,
                                     struct Params params) {
    initialize_population_kernel<<<params.blocks, params.threads>>>(
        (curandStatePhilox4_32_10_t*)curand_states, device, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}

DECLSPEC void initialize_update_arrays(struct mainMemoryPointers host, struct Params params) {
    for (int i = 0; i < params.R; i++) {
        host.O[i] = i;
        host.replica_family[i] = i;
    }
}

DECLSPEC void initialize_resampling_rng(int seed) {
    const uint64_t unsigned_seed = static_cast<uint64_t>(static_cast<uint32_t>(seed));
    resampling_rng.state = 0;
    resampling_rng.stream = (unsigned_seed << 1U) | 1U;
    (void)next_resampling_random();
    resampling_rng.state += unsigned_seed ^ 0x9e3779b97f4a7c15ULL;
    (void)next_resampling_random();
}

DECLSPEC struct ResamplingRngState get_resampling_rng_state() {
    return resampling_rng;
}

DECLSPEC void set_resampling_rng_state(struct ResamplingRngState state) {
    state.stream |= 1ULL;
    resampling_rng = state;
}


// ── Energy calculation ────────────────────────────────────────────────────────
// Per replica: e_j = (Σ local bond terms)/2, e_delta = Σ σ_i².
// Comparisons use bc_energy_int(e_j, e_delta, D_num, D_denum).
__global__ void calc_device_energy_kernel(struct mainMemoryPointers device,
                                          struct Params params) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    long long rep_shift = (long long)r * params.N;
    int sum_j = 0, sum_delta = 0;
    for (int j = 0; j < params.N; j++) {
        int sigma = device.spin[j + rep_shift];
        struct neighborsIndexes ni = SLF(j, params);
        struct neighborsValues  nv = SVLF(device, ni, rep_shift);
        sum_j     += local_energy_j(sigma, nv);
        sum_delta += local_energy_delta(sigma);
    }
    /* Each bond counted twice in the loop → divide by 2 */
    device.e_j[r]     = sum_j / 2;
    device.e_delta[r] = sum_delta;
}

DECLSPEC void calc_device_energy(struct mainMemoryPointers device, struct Params params) {
    calc_device_energy_kernel<<<params.blocks, params.threads>>>(device, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}


// ── Equilibration (microcanonical MC sweep) ───────────────────────────────────
// Each thread handles one replica.  Proposes σ' ≠ σ via bc_propose_spin;
// accepts only if bc_energy_int(new) satisfies ceiling (cooling) or floor (heating).
__global__ void equilibrate_kernel(curandStatePhilox4_32_10_t* curand_states,
                                    struct mainMemoryPointers device,
                                    struct Params params, int U) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    long long rep_shift = (long long)r * params.N;
    int flip_count = 0;

    for (int k = 0; k < params.N * params.nSteps; k++) {
        int j = (int)(curand(&curand_states[r]) % params.N);
        int sigma_old = device.spin[j + rep_shift];
        int sigma_new = bc_propose_spin(sigma_old, curand(&curand_states[r]));

        struct neighborsIndexes ni = SLF(j, params);
        struct neighborsValues  nv = SVLF(device, ni, rep_shift);

        int dEj     = local_energy_j(sigma_new, nv) - local_energy_j(sigma_old, nv);
        int dEdelta = local_energy_delta(sigma_new) - local_energy_delta(sigma_old);
        int ej_new  = device.e_j[r] + dEj;
        int ed_new  = device.e_delta[r] + dEdelta;
        int e_new   = bc_energy_int(ej_new, ed_new, params.D_num, params.D_denum);

        bool accept = params.heat ? (e_new > U) : (e_new < U);
        if (accept) {
            device.e_j[r] = ej_new;
            device.e_delta[r] = ed_new;
            device.spin[j + rep_shift] = sigma_new;
            flip_count++;
        }
    }
    device.replica_statistics[r].flip_count = flip_count;
}

DECLSPEC void equilibrate(void* curand_states, struct mainMemoryPointers device,
                           struct Params params, int U) {
    equilibrate_kernel<<<params.blocks, params.threads>>>(
        (curandStatePhilox4_32_10_t*)curand_states, device, params, U);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}


// ── Per-replica statistics ────────────────────────────────────────────────────
// Called after equilibration at culled level U.  Field-mixing uses:
//   n   = e_delta / N,  eps = e_j / N,  Q = (n - s*eps) / (1 - r*s)
__global__ void calc_replica_statistics_kernel(struct mainMemoryPointers device,
                                                struct Params params, int U) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    long long rep_shift = (long long)r * params.N;

    device.replica_statistics[r].e_j     = 0;
    device.replica_statistics[r].e_delta = 0;
    device.replica_statistics[r].m       = 0;

    int e = bc_energy_int(device.e_j[r], device.e_delta[r], params.D_num, params.D_denum);
    if (e == U) {
        int sum_m = 0;
        for (int j = 0; j < params.N; j++)
            sum_m += device.spin[j + rep_shift];
        device.replica_statistics[r].e_j     = device.e_j[r];
        device.replica_statistics[r].e_delta = device.e_delta[r];
        device.replica_statistics[r].m       = sum_m;
    }
}

DECLSPEC void calc_replica_statistics(struct mainMemoryPointers device, struct Params params,
                                       int U) {
    calc_replica_statistics_kernel<<<params.blocks, params.threads>>>(device, params, U);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}


// ── Replica update after resampling ──────────────────────────────────────────
__global__ void update_replicas_kernel(struct mainMemoryPointers device, struct Params params) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= params.R) return;
    int src = device.update[r];
    if (src != r) {
        device.e_j[r]     = device.e_j[src];
        device.e_delta[r] = device.e_delta[src];
        long long rep_shift     = (long long)r   * params.N;
        long long src_rep_shift = (long long)src * params.N;
        for (int j = 0; j < params.N; j++)
            device.spin[j + rep_shift] = device.spin[j + src_rep_shift];
    }
}

DECLSPEC void update_replicas(struct mainMemoryPointers device, struct Params params) {
    update_replicas_kernel<<<params.blocks, params.threads>>>(device, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}


// ── Sorting and resampling (CPU) ──────────────────────────────────────────────
// Quicksort on bc_energy_int via permutation O; int U (no float tolerance).
DECLSPEC void swap_order(int* O, int i, int j) {
    int tmp = O[i]; O[i] = O[j]; O[j] = tmp;
}

DECLSPEC void quicksort(struct mainMemoryPointers host, struct Params params,
                        int left, int right, int direction) {
    int mid = (left + right) / 2;
    int i = left, j = right;
    int pivot = direction * bc_replica_energy(&host, host.O[mid], &params);
    while (left < j || i < right) {
        while (direction * bc_replica_energy(&host, host.O[i], &params) > pivot) i++;
        while (direction * bc_replica_energy(&host, host.O[j], &params) < pivot) j--;
        if (i <= j) {
            swap_order(host.O, i, j);
            i++; j--;
        } else {
            if (left < j) quicksort(host, params, left, j, direction);
            if (i < right) quicksort(host, params, i, right, direction);
            return;
        }
    }
}

DECLSPEC double prepare_resample_arrays(struct mainMemoryPointers host, struct Params params,
                                         int* U, int* n_culled_exact) {
    int direction = params.heat ? -1 : 1;   /* cooling: descend; heating: ascend */
    quicksort(host, params, 0, params.R - 1, direction);

    int nCull = 0;
    int U_old = *U;
    int U_new = U_old;

    for (int i = 0; i < params.R; i++) {
        U_new = bc_replica_energy(&host, host.O[i], &params);
        if ((!params.heat && U_new < U_old) || (params.heat && U_new > U_old)) {
            *U = U_new;
            break;
        }
    }
    if (*U == U_old) {
        if (n_culled_exact) *n_culled_exact = params.R;
        return 1.0;
    }

    while (nCull < params.R &&
           ((!params.heat && bc_replica_energy(&host, host.O[nCull], &params) >= *U) ||
            ( params.heat && bc_replica_energy(&host, host.O[nCull], &params) <= *U))) {
        nCull++;
    }
    double X = (double)nCull / params.R;
    if (n_culled_exact) *n_culled_exact = nCull;
    printf("Culling factor:\t%f\n", X);
    fflush(stdout);

    for (int i = 0; i < params.R; i++) host.update[i] = i;
    if (nCull < params.R) {
        for (int i = 0; i < nCull; i++) {
            const uint32_t draw = next_resampling_random();
            int src = static_cast<int>(draw % static_cast<uint32_t>(params.R - nCull))
                    + nCull;
            host.update[host.O[i]] = host.O[src];
            host.replica_family[host.O[i]] = host.replica_family[host.O[src]];
        }
    }
    return X;
}

DECLSPEC double calc_family_concentration(const int* family_ids, int R) {
    int* hist = (int*)calloc(R, sizeof(int));
    for (int i = 0; i < R; i++) hist[family_ids[i]]++;
    double rho = 0.0;
    for (int i = 0; i < R; i++) rho += (double)hist[i] * hist[i];
    rho /= (double)R * R;
    free(hist);
    return rho;
}

DECLSPEC double calc_family_avg_sq_size(struct mainMemoryPointers host, struct Params params) {
    const double rho = calc_family_concentration(host.replica_family, params.R);
    printf("RhoT:\t%f\n", rho);
    return rho;
}


// ── Output ─────────────────────────────────────────────────────────────────────
// flip_rate columns = bc_acceptance_pct (0–100% of N*nSteps proposals accepted)
DECLSPEC void initialize_print(struct Files files) {
    /* Legacy columns stay first; exact shell/culling identity is appended. */
    fprintf(files.main_file,
            "E\tculling_factor\treplica_family_avg_sq\tU_scaled\tD_num\tD_denum"
            "\tn_culled_exact\tculling_factor_exact\tequilibrate_seconds"
            "\tpre_resample_family_concentration"
            "\tpost_resample_family_concentration\tgpu_name"
            "\tgpu_compute_capability\tcuda_runtime_version"
            "\ttotal_memory_bytes\tfree_memory_before_setup_bytes"
            "\tfree_memory_after_setup_bytes\tcuda_driver_version\n");
    fflush(files.main_file);
    /* agg_stats: aggregated per-energy-level stats */
    fprintf(files.agg_stats_file,
            "E\tn_culled\tflip_rate\te_j\te_delta\tm\tU_scaled\tD_num\tD_denum"
            "\tmatching_family_concentration\tmatching_unique_families\n");
    fflush(files.agg_stats_file);
    /* detailed_stats: per-replica stats at each culled level — for field mixing */
    fprintf(files.detailed_stats_file,
            "E\tflip_rate\te_j\te_delta\tm\tU_scaled\tD_num\tD_denum"
            "\tfamily_id\ttotal_matching_replicas\tsampling_policy\n");
    fflush(files.detailed_stats_file);
}

DECLSPEC void print_main_data(struct Files files, double E_phys, double X, double rho_t,
                              int U, int D_num, int D_denum, int n_culled_exact,
                              double equilibrate_seconds,
                              double pre_family_concentration,
                              double post_family_concentration,
                              const struct GpuMetadata* metadata) {
    fprintf(files.main_file,
            "%.6f\t%f\t%f\t%d\t%d\t%d\t%d\t%.17g\t%.9f\t%.17g\t%.17g",
            E_phys, X, rho_t, U, D_num, D_denum, n_culled_exact, X,
            equilibrate_seconds, pre_family_concentration,
            post_family_concentration);
    if (metadata) {
        fprintf(files.main_file,
                "\t%s\t%d.%d\t%d\t%llu\t%llu\t%llu\t%d\n",
                metadata->name,
                metadata->compute_major, metadata->compute_minor,
                metadata->cuda_runtime_version,
                static_cast<unsigned long long>(metadata->total_memory_bytes),
                static_cast<unsigned long long>(
                    metadata->free_memory_before_setup_bytes),
                static_cast<unsigned long long>(
                    metadata->free_memory_after_setup_bytes),
                metadata->cuda_driver_version);
    } else {
        fprintf(files.main_file, "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n");
    }
    fflush(files.main_file);
}

DECLSPEC void print_detailed_stats(struct mainMemoryPointers host, struct Params params,
                                    struct Files files, int U,
                                    const int* measured_family_ids, int detail_cap) {
    if (detail_cap == 0) return;
    double E_phys = energy_physical_at_U(host, params, U);
    int total_matching = 0;
    for (int i = 0; i < params.R; ++i)
        if (bc_replica_energy(&host, i, &params) == U) total_matching++;
    const char* policy = detail_cap < 0 ? "all_matching" : "prefix_first_N";
    int printed = 0;
    for (int i = 0; i < params.R
                    && (detail_cap < 0 || printed < detail_cap); i++) {
        if (bc_replica_energy(&host, i, &params) == U) {
            fprintf(files.detailed_stats_file,
                    "%.6f\t%.4f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n",
                    E_phys,
                    bc_acceptance_pct(host.replica_statistics[i].flip_count,
                                      params.N, params.nSteps),
                    host.replica_statistics[i].e_j,
                    host.replica_statistics[i].e_delta,
                    host.replica_statistics[i].m,
                    U, params.D_num, params.D_denum, measured_family_ids[i],
                    total_matching, policy);
            printed++;
        }
    }
    fflush(files.detailed_stats_file);
}

DECLSPEC void print_agg_stats(struct mainMemoryPointers host, struct Params params,
                               struct Files files, int U,
                               const int* measured_family_ids) {
    int n_at_U = 0;
    int unique_families = 0;
    double sum_flip_pct = 0.0;
    int64_t sum_ej = 0, sum_ed = 0, sum_m = 0;
    int* family_hist = (int*)calloc(params.R, sizeof(int));
    for (int i = 0; i < params.R; i++) {
        if (bc_replica_energy(&host, i, &params) == U) {
            n_at_U++;
            if (family_hist[measured_family_ids[i]]++ == 0) unique_families++;
            sum_flip_pct += bc_acceptance_pct(host.replica_statistics[i].flip_count,
                                              params.N, params.nSteps);
            sum_ej += host.replica_statistics[i].e_j;
            sum_ed += host.replica_statistics[i].e_delta;
            sum_m  += host.replica_statistics[i].m;
        }
    }
    if (n_at_U == 0) {
        free(family_hist);
        return;
    }
    double family_concentration = 0.0;
    for (int family = 0; family < params.R; ++family)
        family_concentration += (double)family_hist[family] * family_hist[family];
    family_concentration /= (double)n_at_U * n_at_U;
    free(family_hist);
    double E_phys = energy_physical_at_U(host, params, U);
    fprintf(files.agg_stats_file,
            "%.6f\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%.17g\t%d\n",
            E_phys, n_at_U,
            sum_flip_pct / n_at_U,
            (double)sum_ej / n_at_U,
            (double)sum_ed / n_at_U,
            (double)sum_m  / n_at_U,
            U, params.D_num, params.D_denum, family_concentration,
            unique_families);
    fflush(files.agg_stats_file);
}
