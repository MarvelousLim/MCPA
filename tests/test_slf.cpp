#include "baxterwu_lib.h"
#include "test_slf.h"
#include "test_common.h"

#include <iostream>

// External globals from tests.cpp
extern int g_test_passing;
extern int g_test_failed;
extern struct Params mock_params;
extern void setup_mock_params(int L, int R, int N, bool heat);

// Custom assertion macro (simplified)
#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cout << COLOR_RED "[FAIL] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_failed++; \
    } else { \
        std::cout << COLOR_GREEN "[PASS] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_passing++; \
    } \
    std::cout.flush();

/*
void test_slf_lattice_find() {
    std::cout << "\n========================================================\n";
    std::cout << "STARTING SLF (Spin Lattice Find) Tests\n";
    std::cout << "========================================================\n";

    // Use a small 4x4 lattice: L=4, R=4, N=16
    int L_test = 4;
    int N_test = L_test * L_test;
    setup_mock_params(L_test, L_test, N_test, false);

    // Helper lambda to compute linear index from (x,y)
    auto idx = [L_test](int x, int y) { return x + y * L_test; };

    // ------------------------------------------------------------------
    // Corner Tests (all four corners)
    // ------------------------------------------------------------------

    // Helper lambda to test a specific corner
    auto test_corner = [&](int x, int y, const char* name) {
        int j = idx(x, y);
        std::cout << "\n[Corner] " << name << " (" << x << ", " << y << ") -> index " << j << "\n";

        int expected_left = idx((x - 1 + L_test) % L_test, y);
        int expected_right = idx((x + 1) % L_test, y);
        int expected_up = idx(x, (y - 1 + L_test) % L_test);
        int expected_down = idx(x, (y + 1) % L_test);
        int expected_diag_left = idx((x - 1 + L_test) % L_test, (y - 1 + L_test) % L_test);
        int expected_diag_right = idx((x + 1) % L_test, (y + 1) % L_test);

        neiborsIndexes n_i = SLF(j, mock_params);

        ASSERT_TRUE(n_i.left == expected_left, "Left neighbor index");
        ASSERT_TRUE(n_i.right == expected_right, "Right neighbor index");
        ASSERT_TRUE(n_i.up == expected_up, "Up neighbor index");
        ASSERT_TRUE(n_i.down == expected_down, "Down neighbor index");
        ASSERT_TRUE(n_i.diag_left == expected_diag_left, "DiagLeft neighbor index");
        ASSERT_TRUE(n_i.diag_right == expected_diag_right, "DiagRight neighbor index");
        };

    test_corner(0, 0, "Top-Left");
    test_corner(3, 0, "Top-Right");
    test_corner(0, 3, "Bottom-Left");
    test_corner(3, 3, "Bottom-Right");

    // ------------------------------------------------------------------
    // Edge Tests (non-corner edges: one per side)
    // ------------------------------------------------------------------

    auto test_edge = [&](int x, int y, const char* name) {
        int j = idx(x, y);
        std::cout << "\n[Edge] " << name << " (" << x << ", " << y << ") -> index " << j << "\n";

        int expected_left = idx((x - 1 + L_test) % L_test, y);
        int expected_right = idx((x + 1) % L_test, y);
        int expected_up = idx(x, (y - 1 + L_test) % L_test);
        int expected_down = idx(x, (y + 1) % L_test);
        int expected_diag_left = idx((x - 1 + L_test) % L_test, (y - 1 + L_test) % L_test);
        int expected_diag_right = idx((x + 1) % L_test, (y + 1) % L_test);

        neiborsIndexes n_i = SLF(j, mock_params);

        ASSERT_TRUE(n_i.left == expected_left, "Left neighbor index");
        ASSERT_TRUE(n_i.right == expected_right, "Right neighbor index");
        ASSERT_TRUE(n_i.up == expected_up, "Up neighbor index");
        ASSERT_TRUE(n_i.down == expected_down, "Down neighbor index");
        ASSERT_TRUE(n_i.diag_left == expected_diag_left, "DiagLeft neighbor index");
        ASSERT_TRUE(n_i.diag_right == expected_diag_right, "DiagRight neighbor index");
        };

    test_edge(1, 0, "Top Edge (non-corner)");
    test_edge(3, 1, "Right Edge (non-corner)");
    test_edge(1, 3, "Bottom Edge (non-corner)");
    test_edge(0, 1, "Left Edge (non-corner)");

    // ------------------------------------------------------------------
    // Additional Interior Point (just one more for safety)
    // ------------------------------------------------------------------
    {
        int x = 2, y = 2;
        int j = idx(x, y);
        std::cout << "\n[Interior] (" << x << ", " << y << ") -> index " << j << "\n";

        int expected_left = idx((x - 1 + L_test) % L_test, y);
        int expected_right = idx((x + 1) % L_test, y);
        int expected_up = idx(x, (y - 1 + L_test) % L_test);
        int expected_down = idx(x, (y + 1) % L_test);
        int expected_diag_left = idx((x - 1 + L_test) % L_test, (y - 1 + L_test) % L_test);
        int expected_diag_right = idx((x + 1) % L_test, (y + 1) % L_test);

        neiborsIndexes n_i = SLF(j, mock_params);

        ASSERT_TRUE(n_i.left == expected_left, "Left neighbor index");
        ASSERT_TRUE(n_i.right == expected_right, "Right neighbor index");
        ASSERT_TRUE(n_i.up == expected_up, "Up neighbor index");
        ASSERT_TRUE(n_i.down == expected_down, "Down neighbor index");
        ASSERT_TRUE(n_i.diag_left == expected_diag_left, "DiagLeft neighbor index");
        ASSERT_TRUE(n_i.diag_right == expected_diag_right, "DiagRight neighbor index");
    }
}
*/

void test_slf_lattice_find_vs_precomputed() {
    std::cout << "\n========================================================\n";
    std::cout << "STARTING SLF (Spin Lattice Find) Tests\n";
    std::cout << "========================================================\n";

    const int L_test = 4;
    const int N_test = L_test * L_test;
    setup_mock_params(L_test, L_test, N_test, false);

    // Precomputed expected neighbor indices for 4x4 lattice (manually verified)
    // Format: { left, right, up, down, diag_left, diag_right }
    const int expected[16][6] = {
        // j = 0
        { 3,  1, 12,  4, 15,  5},
        // j = 1
        { 0,  2, 13,  5, 12,  6},
        // j = 2
        { 1,  3, 14,  6, 13,  7},
        // j = 3
        { 2,  0, 15,  7, 14,  4},
        // j = 4
        { 7,  5,  0,  8,  3,  9},
        // j = 5
        { 4,  6,  1,  9,  0, 10},
        // j = 6
        { 5,  7,  2, 10,  1, 11},
        // j = 7
        { 6,  4,  3, 11,  2,  8},
        // j = 8
        {11,  9,  4, 12,  7, 13},
        // j = 9
        { 8, 10,  5, 13,  4, 14},
        // j = 10
        { 9, 11,  6, 14,  5, 15},
        // j = 11
        {10,  8,  7, 15,  6, 12},
        // j = 12
        {15, 13,  8,  0, 11,  1},
        // j = 13
        {12, 14,  9,  1,  8,  2},
        // j = 14
        {13, 15, 10,  2,  9,  3},
        // j = 15
        {14, 12, 11,  3, 10,  0}
    };

    // Test all 16 sites
    for (int j = 0; j < N_test; j++) {
        neiborsIndexes n_i = SLF(j, mock_params);

        // Build a description string for this site
        char desc[100];
        snprintf(desc, sizeof(desc), "j=%d (x=%d, y=%d)", j, j % L_test, j / L_test);

        ASSERT_TRUE(n_i.left == expected[j][0], (std::string(desc) + " left neighbor").c_str());
        ASSERT_TRUE(n_i.right == expected[j][1], (std::string(desc) + " right neighbor").c_str());
        ASSERT_TRUE(n_i.up == expected[j][2], (std::string(desc) + " up neighbor").c_str());
        ASSERT_TRUE(n_i.down == expected[j][3], (std::string(desc) + " down neighbor").c_str());
        ASSERT_TRUE(n_i.diag_left == expected[j][4], (std::string(desc) + " diag_left neighbor").c_str());
        ASSERT_TRUE(n_i.diag_right == expected[j][5], (std::string(desc) + " diag_right neighbor").c_str());
    }

    std::cout << COLOR_GREEN "\nAll 16 lattice sites verified against precomputed table.\n" COLOR_RESET;
}
