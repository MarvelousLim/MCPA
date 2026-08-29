#include "mem_estimate.h"

#include <cuda_runtime.h>
#include <stdio.h>

static void format_bytes(char* buffer, size_t buffer_size, size_t bytes) {
    double value = (double)bytes;
    if (bytes >= (1ull << 30))
        snprintf(buffer, buffer_size, "%6.2f GiB", value / (1ull << 30));
    else if (bytes >= (1ull << 20))
        snprintf(buffer, buffer_size, "%6.2f MiB", value / (1ull << 20));
    else if (bytes >= (1ull << 10))
        snprintf(buffer, buffer_size, "%6.2f KiB", value / (1ull << 10));
    else
        snprintf(buffer, buffer_size, "%6zu B", bytes);
}

void mem_estimate_query_gpu(struct MemEstimate* estimate) {
    cudaError_t error = cudaMemGetInfo(&estimate->gpu_free, &estimate->gpu_total);
    if (error != cudaSuccess) {
        estimate->gpu_free = 0;
        estimate->gpu_total = 0;
        cudaGetLastError();
    }

    int device_id = 0;
    cudaDeviceProp properties;
    if (cudaGetDevice(&device_id) == cudaSuccess
            && cudaGetDeviceProperties(&properties, device_id) == cudaSuccess)
        snprintf(estimate->gpu_name, sizeof(estimate->gpu_name), "%s", properties.name);
    else
        snprintf(estimate->gpu_name, sizeof(estimate->gpu_name), "(unknown)");
}

struct MemEstimate estimate_setup_memory(int N, int R, size_t spin_bytes,
                                         size_t host_aux_bytes,
                                         size_t device_aux_bytes,
                                         size_t device_rng_bytes) {
    struct MemEstimate result = {};
    size_t lattice = (size_t)R * (size_t)N * spin_bytes;

    result.host_lattice = lattice;
    result.host_aux = host_aux_bytes;
    result.host_total = result.host_lattice + result.host_aux;

    result.device_lattice = lattice;
    result.device_aux = device_aux_bytes;
    result.device_rng = device_rng_bytes;
    result.device_total = result.device_lattice + result.device_aux + result.device_rng;
    result.grand_total = result.host_total + result.device_total;

    mem_estimate_query_gpu(&result);
    return result;
}

int report_setup_memory(const char* model, int L, int N, int R,
                        size_t spin_bytes, const struct MemEstimate* estimate) {
    char value[32];
    char total[32];
    const char* title = model ? model : "MCPA";

    printf("\n========== %s memory estimate ==========\n", title);
    printf("  L=%d  N=%d  R=%d  spin=%zu B  rng=%zu B/replica\n",
           L, N, R, spin_bytes,
           R > 0 ? estimate->device_rng / (size_t)R : 0);
    printf("  GPU: %s\n", estimate->gpu_name);

    format_bytes(value, sizeof(value), estimate->gpu_free);
    format_bytes(total, sizeof(total), estimate->gpu_total);
    double already_used = estimate->gpu_total
        ? 100.0 * (double)(estimate->gpu_total - estimate->gpu_free)
          / (double)estimate->gpu_total
        : 0.0;
    printf("  VRAM free/total: %s / %s  (%.1f%% already used)\n",
           value, total, already_used);
    printf("  ------------------------------------------------\n");

    format_bytes(value, sizeof(value), estimate->host_lattice);
    printf("  Host lattice         %s\n", value);
    format_bytes(value, sizeof(value), estimate->host_aux);
    printf("  Host auxiliary       %s\n", value);
    format_bytes(value, sizeof(value), estimate->host_total);
    printf("  Host total           %s\n", value);

    format_bytes(value, sizeof(value), estimate->device_lattice);
    printf("  Device lattice       %s\n", value);
    format_bytes(value, sizeof(value), estimate->device_aux);
    printf("  Device auxiliary     %s\n", value);
    format_bytes(value, sizeof(value), estimate->device_rng);
    printf("  Device RNG           %s\n", value);
    format_bytes(value, sizeof(value), estimate->device_total);
    printf("  Device total         %s\n", value);

    format_bytes(value, sizeof(value), estimate->grand_total);
    printf("  ------------------------------------------------\n");
    printf("  Grand total (H+D)    %s\n", value);

    double of_free = estimate->gpu_free
        ? 100.0 * (double)estimate->device_total / (double)estimate->gpu_free : 0.0;
    double of_total = estimate->gpu_total
        ? 100.0 * (double)estimate->device_total / (double)estimate->gpu_total : 0.0;
    printf("  Device need / free   %.1f%%\n", of_free);
    printf("  Device need / total  %.1f%%\n", of_total);

    int insufficient = estimate->gpu_free
                       && estimate->device_total > estimate->gpu_free;
    if (insufficient)
        printf("  WARNING: device setup exceeds free VRAM\n");
    else if (!estimate->gpu_free)
        printf("  WARNING: CUDA could not report available VRAM\n");
    else
        printf("  OK: device setup fits in free VRAM\n");
    printf("=================================================\n\n");
    fflush(stdout);
    fflush(stderr);
    return insufficient;
}
