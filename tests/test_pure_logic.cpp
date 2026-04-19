#include "baxterwu_lib.h"
#include "test_pure_logic.h"
#include "test_common.h"
#include <iostream>

// External globals from tests.cpp
extern int g_test_passing;
extern int g_test_failed;

#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cout << COLOR_RED "[FAIL] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_failed++; \
    } else { \
        std::cout << COLOR_GREEN "[PASS] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_passing++; \
    } \
    std::cout.flush();

// ----------------------------------------------------------------------
// Test: between()
// ----------------------------------------------------------------------
void test_between() {
    std::cout << "\n--- Testing between() ---\n";

    // Standard inclusive range
    ASSERT_TRUE(between(1.0f, 0.0f, 2.0f), "1.0 is between 0.0 and 2.0");
    ASSERT_TRUE(between(0.0f, 0.0f, 2.0f), "0.0 is between 0.0 and 2.0 (boundary)");
    ASSERT_TRUE(between(2.0f, 0.0f, 2.0f), "2.0 is between 0.0 and 2.0 (boundary)");
    ASSERT_TRUE(!between(3.0f, 0.0f, 2.0f), "3.0 is not between 0.0 and 2.0");
    ASSERT_TRUE(!between(-1.0f, 0.0f, 2.0f), "-1.0 is not between 0.0 and 2.0");

    // Reverse range (implementation handles this)
    ASSERT_TRUE(between(1.0f, 2.0f, 0.0f), "1.0 is between 2.0 and 0.0 (reverse)");
    ASSERT_TRUE(between(0.0f, 2.0f, 0.0f), "0.0 is between 2.0 and 0.0 (reverse boundary)");
    ASSERT_TRUE(between(2.0f, 2.0f, 0.0f), "2.0 is between 2.0 and 0.0 (reverse boundary)");

    // Edge: equal bounds
    ASSERT_TRUE(between(5.0f, 5.0f, 5.0f), "Value equals both bounds");
    ASSERT_TRUE(!between(4.9f, 5.0f, 5.0f), "Value just outside equal bounds");
}

// ----------------------------------------------------------------------
// Test: local_energy()
// ----------------------------------------------------------------------
void test_local_energy() {
    std::cout << "\n--- Testing local_energy() ---\n";

    // Case 1: All spins +1
    {
        neiborsValues n = { 1, 1, 1, 1, 1, 1 };
        int energy = local_energy(1, n);
        ASSERT_TRUE(energy == -6, "All +1 spins yields energy -6");
    }

    // Case 2: Current +1, all neighbors -1
    // Each product: (-1)*(-1)*(+1) = +1, sum over 6 triangles = 6, then -6
    {
        neiborsValues n = { -1, -1, -1, -1, -1, -1 };
        int energy = local_energy(1, n);
        ASSERT_TRUE(energy == -6, "Current +1, all neighbors -1 yields -6");
    }

    // Case 3: Current -1, all neighbors +1 -> expected +6
    {
        neiborsValues n = { 1, 1, 1, 1, 1, 1 };
        int energy = local_energy(-1, n);
        ASSERT_TRUE(energy == 6, "Current -1, all +1 neighbors yields +6");
    }

    // Case 4: Current -1, all neighbors -1 -> expected +6? Wait, let's recalc.
    // Each product: (-1)*(-1)*(-1) = -1, sum = -6, then -(-6) = +6.
    {
        neiborsValues n = { -1, -1, -1, -1, -1, -1 };
        int energy = local_energy(-1, n);
        ASSERT_TRUE(energy == 6, "All -1 spins yields +6");
    }

    // Case 5: Symmetry check (local_energy(s, n) == -local_energy(-s, n))
    {
        neiborsValues n = { 1, -1, 1, -1, 1, -1 };  // mixed
        int e_plus = local_energy(1, n);
        int e_minus = local_energy(-1, n);
        ASSERT_TRUE(e_plus == -e_minus, "Energy is antisymmetric under spin flip");
    }

    // Case 6: Mixed neighbors (already had one in original, but let's verify a specific pattern)
    // For a known configuration, we can compute manually.
    // Example: left=1, right=-1, up=1, down=-1, diag_left=1, diag_right=-1, current=1
    // Triangles (from code):
    // diag_left*up*current = 1*1*1=1
    // diag_left*left*current = 1*1*1=1
    // diag_right*down*current = -1*-1*1=1
    // diag_right*right*current = -1*-1*1=1
    // down*left*current = -1*1*1=-1
    // up*right*current = 1*-1*1=-1
    // sum = 1+1+1+1-1-1 = 2, then -2
    {
        neiborsValues n = { 1, -1, 1, -1, 1, -1 };
        int energy = local_energy(1, n);
        ASSERT_TRUE(energy == -2, "Specific mixed pattern yields -2");
    }
}

// ----------------------------------------------------------------------
// Test: swap()
// ----------------------------------------------------------------------
void test_swap() {
    std::cout << "\n--- Testing swap() ---\n";

    int arr[] = { 10, 20, 30 };
    swap(arr, 0, 2);
    ASSERT_TRUE(arr[0] == 30 && arr[2] == 10, "swap(0,2) exchanges elements");
    swap(arr, 1, 1);
    ASSERT_TRUE(arr[1] == 20, "swap(1,1) leaves element unchanged");
}

// ----------------------------------------------------------------------
// Test: quicksort()
// ----------------------------------------------------------------------
void test_quicksort() {
    std::cout << "\n--- Testing quicksort() ---\n";

    // Helper to set up a test case
    auto run_sort_test = [](int* E_arr, int* O_arr, int R, int direction, const char* desc) {
        mainMemoryPointers host;
        host.E = E_arr;
        host.O = O_arr;
        // replica_family not used, but point to something valid
        int dummy_family[100];
        host.replica_family = dummy_family;

        quicksort(host, 0, R - 1, direction);
        };

    // Test 1: Normal descending sort (R=5 with duplicates)
    {
        int E[] = { 5, 2, 8, 2, 6 };
        int O[] = { 0, 1, 2, 3, 4 };
        run_sort_test(E, O, 5, 1, "descending with duplicates");

        ASSERT_TRUE(E[O[0]] == 8, "Largest energy at first position");
        ASSERT_TRUE(E[O[1]] == 6, "Second largest at second");
        ASSERT_TRUE(E[O[2]] == 5, "Third largest at third");
        // Remaining two are 2s (order among equal energies not guaranteed)
        ASSERT_TRUE(E[O[3]] == 2 && E[O[4]] == 2, "Smallest energies at end");
    }

    // Test 2: Ascending sort
    {
        int E[] = { 5, 2, 8, 2, 6 };
        int O[] = { 0, 1, 2, 3, 4 };
        run_sort_test(E, O, 5, -1, "ascending with duplicates");

        ASSERT_TRUE(E[O[0]] == 2, "Smallest energy first");
        ASSERT_TRUE(E[O[4]] == 8, "Largest energy last");
    }

    // Test 3: Single element (R=1)
    {
        int E[] = { 42 };
        int O[] = { 0 };
        run_sort_test(E, O, 1, 1, "single element descending");
        ASSERT_TRUE(E[O[0]] == 42, "Single element unchanged");
    }

    // Test 4: Already sorted (descending)
    {
        int E[] = { 9, 7, 5, 3, 1 };
        int O[] = { 0, 1, 2, 3, 4 };
        run_sort_test(E, O, 5, 1, "already descending");
        bool sorted_desc = true;
        for (int i = 0; i < 4; i++)
            if (E[O[i]] < E[O[i + 1]]) sorted_desc = false;
        ASSERT_TRUE(sorted_desc, "Already descending remains descending");
    }

    // Test 5: Reverse sorted (ascending input, sort descending)
    {
        int E[] = { 1, 3, 5, 7, 9 };
        int O[] = { 0, 1, 2, 3, 4 };
        run_sort_test(E, O, 5, 1, "reverse sorted to descending");
        ASSERT_TRUE(E[O[0]] == 9, "Largest first after sorting reverse");
    }

    // Test 6: All equal energies (should not crash)
    {
        int E[] = { 4, 4, 4, 4, 4 };
        int O[] = { 0, 1, 2, 3, 4 };
        run_sort_test(E, O, 5, 1, "all equal energies");
        // Order may be anything, but check no out-of-bounds
        bool all_four = true;
        for (int i = 0; i < 5; i++)
            if (E[O[i]] != 4) all_four = false;
        ASSERT_TRUE(all_four, "All energies still 4");
    }
}