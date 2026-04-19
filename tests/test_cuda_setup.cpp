#include "baxterwu_lib.h"
#include "test_cuda_setup.h"
#include "test_common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

extern int g_test_passing;
extern int g_test_failed;
extern struct Params mock_params;
extern void setup_mock_params(int L, int R, int N, bool heat);

#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cout << COLOR_RED "[FAIL] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_failed++; \
    } else { \
        std::cout << COLOR_GREEN "[PASS] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_passing++; \
    } \
    std::cout.flush();

void test_setup_curand_states() {
    std::cout << "\n--- Testing setup_curand_states() ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);
    mock_params.seed = 12345;

    void* curand_states = setup_curand_states(mock_params);
    ASSERT_TRUE(curand_states != nullptr, "Returns non-null pointer");

    cudaFree(curand_states);
    ASSERT_TRUE(cudaGetLastError() == cudaSuccess, "CUDA memory freed successfully");
}

void test_initialize_population_random() {
    std::cout << "\n--- Testing initialize_population(random) ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);
    mock_params.seed = 42;

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};

    host.spin = (int*)malloc(N_test * R_test * sizeof(int));
    host.E = (int*)malloc(R_test * sizeof(int));
    host.O = (int*)malloc(R_test * sizeof(int));
    host.update = (int*)malloc(R_test * sizeof(int));
    host.replica_family = (int*)malloc(R_test * sizeof(int));
    host.replica_statistics = (struct replicaStatistics*)malloc(R_test * sizeof(struct replicaStatistics));

    cudaMalloc(&device.spin, N_test * R_test * sizeof(int));
    cudaMalloc(&device.E, R_test * sizeof(int));
    cudaMalloc(&device.O, R_test * sizeof(int));
    cudaMalloc(&device.update, R_test * sizeof(int));
    cudaMalloc(&device.replica_family, R_test * sizeof(int));
    cudaMalloc(&device.replica_statistics, R_test * sizeof(struct replicaStatistics));

    void* curand_states = setup_curand_states(mock_params);
    initialize_population(curand_states, device, mock_params, random_pop);

    int* h_spin = (int*)malloc(N_test * R_test * sizeof(int));
    cudaMemcpy(h_spin, device.spin, N_test * R_test * sizeof(int), cudaMemcpyDeviceToHost);

    int plus_count = 0;
    int minus_count = 0;
    for (int i = 0; i < N_test * R_test; i++) {
        if (h_spin[i] == 1) plus_count++;
        else if (h_spin[i] == -1) minus_count++;
    }

    ASSERT_TRUE(plus_count + minus_count == N_test * R_test, "All spins are either +1 or -1");
    ASSERT_TRUE(plus_count > 0 && minus_count > 0, "Random population contains both spin types");

    free(h_spin);
    free(host.spin);
    free(host.E);
    free(host.O);
    free(host.update);
    free(host.replica_family);
    free(host.replica_statistics);
    cudaFree(device.spin);
    cudaFree(device.E);
    cudaFree(device.O);
    cudaFree(device.update);
    cudaFree(device.replica_family);
    cudaFree(device.replica_statistics);
    cudaFree(curand_states);
}

void test_initialize_population_sublattice() {
    std::cout << "\n--- Testing initialize_population(by_sublattice) ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};

    host.spin = (int*)malloc(N_test * R_test * sizeof(int));
    host.E = (int*)malloc(R_test * sizeof(int));
    host.O = (int*)malloc(R_test * sizeof(int));
    host.update = (int*)malloc(R_test * sizeof(int));
    host.replica_family = (int*)malloc(R_test * sizeof(int));
    host.replica_statistics = (struct replicaStatistics*)malloc(R_test * sizeof(struct replicaStatistics));

    cudaMalloc(&device.spin, N_test * R_test * sizeof(int));
    cudaMalloc(&device.E, R_test * sizeof(int));
    cudaMalloc(&device.O, R_test * sizeof(int));
    cudaMalloc(&device.update, R_test * sizeof(int));
    cudaMalloc(&device.replica_family, R_test * sizeof(int));
    cudaMalloc(&device.replica_statistics, R_test * sizeof(struct replicaStatistics));

    void* curand_states = setup_curand_states(mock_params);
    initialize_population(curand_states, device, mock_params, by_sublattice, 1, -1, 0);

    int* h_spin = (int*)malloc(N_test * R_test * sizeof(int));
    cudaMemcpy(h_spin, device.spin, N_test * R_test * sizeof(int), cudaMemcpyDeviceToHost);

    bool valid_sublattice = true;
    for (int r = 0; r < R_test && valid_sublattice; r++) {
        for (int j = 0; j < N_test; j++) {
            int spin = h_spin[r * N_test + j];
            int mod = j % 3;
            if (mod == 0 && spin != 1) valid_sublattice = false;
            else if (mod == 1 && spin != -1) valid_sublattice = false;
            else if (mod == 2 && spin != 0) valid_sublattice = false;
        }
    }

    ASSERT_TRUE(valid_sublattice, "Sublattice pattern matches (s_a=1, s_b=-1, s_c=0)");

    free(h_spin);
    free(host.spin);
    free(host.E);
    free(host.O);
    free(host.update);
    free(host.replica_family);
    free(host.replica_statistics);
    cudaFree(device.spin);
    cudaFree(device.E);
    cudaFree(device.O);
    cudaFree(device.update);
    cudaFree(device.replica_family);
    cudaFree(device.replica_statistics);
    cudaFree(curand_states);
}

void test_initialize_population_strips() {
    std::cout << "\n--- Testing initialize_population(strips) ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};

    host.spin = (int*)malloc(N_test * R_test * sizeof(int));
    host.E = (int*)malloc(R_test * sizeof(int));
    host.O = (int*)malloc(R_test * sizeof(int));
    host.update = (int*)malloc(R_test * sizeof(int));
    host.replica_family = (int*)malloc(R_test * sizeof(int));
    host.replica_statistics = (struct replicaStatistics*)malloc(R_test * sizeof(struct replicaStatistics));

    cudaMalloc(&device.spin, N_test * R_test * sizeof(int));
    cudaMalloc(&device.E, R_test * sizeof(int));
    cudaMalloc(&device.O, R_test * sizeof(int));
    cudaMalloc(&device.update, R_test * sizeof(int));
    cudaMalloc(&device.replica_family, R_test * sizeof(int));
    cudaMalloc(&device.replica_statistics, R_test * sizeof(struct replicaStatistics));

    void* curand_states = setup_curand_states(mock_params);
    initialize_population(curand_states, device, mock_params, strips, 1, -1, 0);

    int* h_spin = (int*)malloc(N_test * R_test * sizeof(int));
    cudaMemcpy(h_spin, device.spin, N_test * R_test * sizeof(int), cudaMemcpyDeviceToHost);

    bool valid_strips = true;
    for (int r = 0; r < R_test && valid_strips; r++) {
        for (int j = 0; j < N_test; j++) {
            int x = j % L_test;
            int y = j / L_test;
            int spin = h_spin[r * N_test + j];
            int mod = (x + y) % 3;
            if (mod == 0 && spin != 1) valid_strips = false;
            else if (mod == 1 && spin != -1) valid_strips = false;
            else if (mod == 2 && spin != 0) valid_strips = false;
        }
    }

    ASSERT_TRUE(valid_strips, "Strip pattern matches (s_a=1, s_b=-1, s_c=0)");

    free(h_spin);
    free(host.spin);
    free(host.E);
    free(host.O);
    free(host.update);
    free(host.replica_family);
    free(host.replica_statistics);
    cudaFree(device.spin);
    cudaFree(device.E);
    cudaFree(device.O);
    cudaFree(device.update);
    cudaFree(device.replica_family);
    cudaFree(device.replica_statistics);
    cudaFree(curand_states);
}

void test_initialize_update_arrays() {
    std::cout << "\n--- Testing initialize_update_arrays() ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);

    struct mainMemoryPointers host = {};
    host.O = (int*)malloc(R_test * sizeof(int));
    host.replica_family = (int*)malloc(R_test * sizeof(int));

    initialize_update_arrays(host, mock_params);

    bool O_correct = true;
    bool family_correct = true;
    for (int i = 0; i < R_test; i++) {
        if (host.O[i] != i) O_correct = false;
        if (host.replica_family[i] != i) family_correct = false;
    }

    ASSERT_TRUE(O_correct, "O array initialized to [0, 1, 2, ...]");
    ASSERT_TRUE(family_correct, "replica_family initialized to [0, 1, 2, ...]");

    free(host.O);
    free(host.replica_family);
}