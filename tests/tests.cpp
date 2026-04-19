#include "baxterwu_lib.h"
#include "test_common.h"
#include "test_pure_logic.h"
#include "test_slf.h"
#include "test_cuda_setup.h"
#include "test_kernel_energy.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <windows.h>   // For enabling ANSI escape sequences

// ====================================================================================
// 1. COLOR SUPPORT & GLOBAL TEST STATE
// ====================================================================================

// Enable ANSI escape sequences on Windows 10+ console
void enable_ansi_support() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
}

// ANSI color codes
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_BOLD    "\033[1m"

// Global test counters (shared with other test files via extern)
int g_test_passing = 0;
int g_test_failed = 0;

// Global mock parameters for controlled testing (shared with other test files)
struct Params mock_params;

// Function to reset the global mock parameters (shared with other test files)
void setup_mock_params(int L, int R, int N, bool heat) {
    mock_params.L = L;
    mock_params.R = R;
    mock_params.N = N;
    mock_params.heat = heat;
    mock_params.blocks = (R + L - 1) / L;  // Ceiling division to ensure at least 1 block
    if (mock_params.blocks < 1) mock_params.blocks = 1;
    mock_params.threads = L;
    mock_params.singleIntRowByteSize = R * sizeof(int);
    mock_params.fullLatticeByteSize = N * sizeof(int);
    mock_params.replicaStatisticsByteSize = R * sizeof(replicaStatistics);
}

// Custom assertion macro with colored output
#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cout << COLOR_RED "[FAIL] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_failed++; \
    } else { \
        std::cout << COLOR_GREEN "[PASS] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_passing++; \
    } \
    std::cout.flush();

// Macro to wrap and execute a test function with colored header
#define RUN_TEST(test_func) \
    do { \
        std::cout << COLOR_CYAN COLOR_BOLD "\n====================================================\n"; \
        std::cout << "STARTING TEST: " << #test_func << COLOR_RESET "\n"; \
        test_func(); \
    } while (0)

// ====================================================================================
// 2. TEST STUBS: RESAMPLE LOGIC (to be implemented later)
// ====================================================================================

void test_resample() {
    std::cout << COLOR_YELLOW "\n--- Resample Tests (stub) ---" COLOR_RESET "\n";
    // Placeholder: will implement in Phase 4
}

void test_integration_simulation() {
    std::cout << COLOR_YELLOW "\n--- Integration Simulation Test (stub) ---" COLOR_RESET "\n";
    // Placeholder: will implement in Phase 5
}

// ====================================================================================
// 3. MAIN TEST RUNNER
// ====================================================================================

int main() {
    enable_ansi_support();  // Ensure colored output works

    std::cout << COLOR_CYAN COLOR_BOLD;
    std::cout << "===========================================================\n";
    std::cout << "       STARTING BAXTERWU LIBRARY TEST SUITE\n";
    std::cout << "===========================================================" COLOR_RESET "\n";

    // Phase 1: Pure CPU Logic Tests
    RUN_TEST(test_between);
    RUN_TEST(test_local_energy);
    RUN_TEST(test_swap);
    RUN_TEST(test_quicksort);
    RUN_TEST(test_slf_lattice_find_vs_precomputed);

    // Phase 2: CUDA Setup Tests
    RUN_TEST(test_setup_curand_states);
    RUN_TEST(test_initialize_population_random);
    RUN_TEST(test_initialize_population_sublattice);
    RUN_TEST(test_initialize_population_strips);
    RUN_TEST(test_initialize_update_arrays);

    // Phase 3: Kernel Energy Tests
    RUN_TEST(test_calc_device_energy);
    RUN_TEST(test_calc_device_energy_known_state);
    RUN_TEST(test_equilibrate);
    RUN_TEST(test_equilibrate_energy_lowering);
    RUN_TEST(test_calc_replica_statistics);

    // Phase 4: Resample Tests (stub for now)
    RUN_TEST(test_resample);

    // Phase 5: Integration Test (stub)
    RUN_TEST(test_integration_simulation);

    // Summary with color based on failures
    std::cout << COLOR_CYAN COLOR_BOLD "\n================================================================\n";
    std::cout << "TEST SUMMARY:\n";
    std::cout << COLOR_GREEN "PASSING TESTS: " << g_test_passing << COLOR_RESET "\n";
    if (g_test_failed > 0)
        std::cout << COLOR_RED "FAILED TESTS:  " << g_test_failed << COLOR_RESET "\n";
    else
        std::cout << "FAILED TESTS:  " << g_test_failed << "\n";
    std::cout << COLOR_CYAN COLOR_BOLD "===========================================================" COLOR_RESET "\n";

    // Cleanup any device memory (if any tests allocated)
    // Currently no allocations, but placeholder for future.

    return g_test_failed > 0 ? 1 : 0;
}