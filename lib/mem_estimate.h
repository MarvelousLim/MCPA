#pragma once

#include <stddef.h>

/* Shared MCPA setup-memory report — copy to other model branches as-is. */
struct MemEstimate {
    size_t host_lattice;   /* R * N * sizeof(spin) */
    size_t host_aux;       /* per-replica host rows (E, update, family, ...) */
    size_t host_total;
    size_t device_lattice; /* R * N * sizeof(spin) */
    size_t device_aux;     /* per-replica device rows (E, update, stats, ...) */
    size_t device_rng;     /* Philox / curand states */
    size_t device_total;
    size_t grand_total;    /* host + device */
    size_t gpu_free;
    size_t gpu_total;
    char gpu_name[256];
};

void mem_estimate_query_gpu(struct MemEstimate* estimate);
struct MemEstimate estimate_setup_memory(int N, int R, size_t spin_bytes,
                                         size_t host_aux_bytes,
                                         size_t device_aux_bytes,
                                         size_t device_rng_bytes);
int report_setup_memory(const char* model, int L, int N, int R,
                        size_t spin_bytes, const struct MemEstimate* estimate);
