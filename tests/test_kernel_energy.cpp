#include "baxterwu_lib.h"
#include "test_kernel_energy.h"
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

static void setup_test_memory(struct mainMemoryPointers* host, struct mainMemoryPointers* device, int N, int R) {
    host->spin = (int*)malloc(N * R * sizeof(int));
    host->E = (int*)malloc(R * sizeof(int));
    host->O = (int*)malloc(R * sizeof(int));
    host->update = (int*)malloc(R * sizeof(int));
    host->replica_family = (int*)malloc(R * sizeof(int));
    host->replica_statistics = (struct replicaStatistics*)malloc(R * sizeof(struct replicaStatistics));

    cudaMalloc(&device->spin, N * R * sizeof(int));
    cudaMalloc(&device->E, R * sizeof(int));
    cudaMalloc(&device->O, R * sizeof(int));
    cudaMalloc(&device->update, R * sizeof(int));
    cudaMalloc(&device->replica_family, R * sizeof(int));
    cudaMalloc(&device->replica_statistics, R * sizeof(struct replicaStatistics));

    cudaMemset(device->spin, 0, N * R * sizeof(int));
    cudaMemset(device->E, 0, R * sizeof(int));
    cudaMemset(device->O, 0, R * sizeof(int));
    cudaMemset(device->update, 0, R * sizeof(int));
    cudaMemset(device->replica_family, 0, R * sizeof(int));
    cudaMemset(device->replica_statistics, 0, R * sizeof(struct replicaStatistics));
}

static void cleanup_test_memory(struct mainMemoryPointers* host, struct mainMemoryPointers* device) {
    free(host->spin);
    free(host->E);
    free(host->O);
    free(host->update);
    free(host->replica_family);
    free(host->replica_statistics);

    cudaFree(device->spin);
    cudaFree(device->E);
    cudaFree(device->O);
    cudaFree(device->update);
    cudaFree(device->replica_family);
    cudaFree(device->replica_statistics);
}

void test_calc_device_energy() {
    std::cout << "\n--- Testing calc_device_energy() ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};
    setup_test_memory(&host, &device, N_test, R_test);

    for (int i = 0; i < N_test * R_test; i++) {
        host.spin[i] = 1;
    }
    cudaMemcpy(device.spin, host.spin, N_test * R_test * sizeof(int), cudaMemcpyHostToDevice);

    calc_device_energy(device, mock_params);

    int h_E;
    cudaMemcpy(&h_E, device.E, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "  Energy value: " << h_E << "\n";

    ASSERT_TRUE(h_E < 0, "All-ferromagnetic lattice has negative energy");

    cudaFree(device.spin);
    cudaFree(device.E);
    cudaFree(device.O);
    cudaFree(device.update);
    cudaFree(device.replica_family);
    cudaFree(device.replica_statistics);
    free(host.spin);
    free(host.E);
    free(host.O);
    free(host.update);
    free(host.replica_family);
    free(host.replica_statistics);
}

void test_calc_device_energy_known_state() {
    std::cout << "\n--- Testing calc_device_energy() with known state ---\n";

    const int L_test = 2;
    const int R_test = 1;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};
    setup_test_memory(&host, &device, N_test, R_test);

    host.spin[0] = 1;
    host.spin[1] = 1;
    host.spin[2] = 1;
    host.spin[3] = 1;
    cudaMemcpy(device.spin, host.spin, N_test * sizeof(int), cudaMemcpyHostToDevice);

    calc_device_energy(device, mock_params);

    int h_E;
    cudaMemcpy(&h_E, device.E, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "  2x2 all +1 energy: " << h_E << "\n";
    ASSERT_TRUE(h_E < 0, "2x2 all +1 has negative energy");

    cleanup_test_memory(&host, &device);
}

void test_equilibrate() {
    std::cout << "\n--- Testing equilibrate() ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);
    mock_params.nSteps = 1;
    mock_params.seed = 42;

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};
    setup_test_memory(&host, &device, N_test, R_test);

    for (int i = 0; i < N_test * R_test; i++) {
        host.spin[i] = 1;
    }
    cudaMemcpy(device.spin, host.spin, N_test * R_test * sizeof(int), cudaMemcpyHostToDevice);

    void* curand_states = setup_curand_states(mock_params);
    equilibrate(curand_states, device, mock_params, 1000);

    int* h_spin = (int*)malloc(N_test * R_test * sizeof(int));
    cudaMemcpy(h_spin, device.spin, N_test * R_test * sizeof(int), cudaMemcpyDeviceToHost);

    bool has_both_spins = false;
    int plus_count = 0;
    for (int i = 0; i < N_test * R_test; i++) {
        if (h_spin[i] == 1) plus_count++;
    }
    has_both_spins = (plus_count > 0 && plus_count < N_test * R_test);

    std::cout << "  Plus spins: " << plus_count << "/" << (N_test * R_test) << "\n";
    ASSERT_TRUE(has_both_spins, "Equilibration produces spin flips");

    free(h_spin);
    cudaFree(curand_states);
    cleanup_test_memory(&host, &device);
}

void test_equilibrate_energy_lowering() {
    std::cout << "\n--- Testing equilibrate() energy lowering ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);
    mock_params.heat = false;
    mock_params.nSteps = 10;
    mock_params.seed = 123;

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};
    setup_test_memory(&host, &device, N_test, R_test);

    for (int i = 0; i < N_test * R_test; i++) {
        host.spin[i] = 1;
    }
    cudaMemcpy(device.spin, host.spin, N_test * R_test * sizeof(int), cudaMemcpyHostToDevice);

    calc_device_energy(device, mock_params);
    int* h_E_before = (int*)malloc(R_test * sizeof(int));
    cudaMemcpy(h_E_before, device.E, R_test * sizeof(int), cudaMemcpyDeviceToHost);
    int initial_energy = h_E_before[0];

    void* curand_states = setup_curand_states(mock_params);
    equilibrate(curand_states, device, mock_params, initial_energy);

    calc_device_energy(device, mock_params);
    int* h_E_after = (int*)malloc(R_test * sizeof(int));
    cudaMemcpy(h_E_after, device.E, R_test * sizeof(int), cudaMemcpyDeviceToHost);
    int final_energy = h_E_after[0];

    std::cout << "  Initial energy: " << initial_energy << ", Final energy: " << final_energy << "\n";
    ASSERT_TRUE(final_energy <= initial_energy, "Energy does not increase after equilibration (heat=false)");

    free(h_E_before);
    free(h_E_after);
    cudaFree(curand_states);
    cleanup_test_memory(&host, &device);
}

void test_calc_replica_statistics() {
    std::cout << "\n--- Testing calc_replica_statistics() ---\n";

    const int L_test = 4;
    const int R_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, R_test, N_test, false);
    mock_params.seed = 42;

    struct mainMemoryPointers host = {};
    struct mainMemoryPointers device = {};
    setup_test_memory(&host, &device, N_test, R_test);

    for (int i = 0; i < N_test * R_test; i++) {
        host.spin[i] = 1;
    }
    cudaMemcpy(device.spin, host.spin, N_test * R_test * sizeof(int), cudaMemcpyHostToDevice);

    calc_device_energy(device, mock_params);

    int h_E;
    cudaMemcpy(&h_E, device.E, sizeof(int), cudaMemcpyDeviceToHost);

    calc_replica_statistics(device, mock_params, h_E);

    struct replicaStatistics* h_stats = (struct replicaStatistics*)malloc(R_test * sizeof(struct replicaStatistics));
    cudaMemcpy(h_stats, device.replica_statistics, R_test * sizeof(struct replicaStatistics), cudaMemcpyDeviceToHost);

    bool have_statistics = false;
    for (int j = 0; j < 3; j++) {
        if (h_stats[0].magnetization[j] != 0) have_statistics = true;
    }

    std::cout << "  Magnetization[0]: " << h_stats[0].magnetization[0] << "\n";
    ASSERT_TRUE(have_statistics, "Statistics computed for ferromagnetic state");

    free(h_stats);
    cleanup_test_memory(&host, &device);
}