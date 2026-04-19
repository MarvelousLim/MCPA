#ifndef SLF_TEST_H
#define SLF_TEST_H

// Include necessary standard headers
#include<cmath>
#include<iostream>

// --- Test Utility Prototypes ---

/**
 * @brief Resets and sets the global mock parameters for the SLF test.
 * @param L Lattice dimension.
 * @param R Row count.
 * @param N Total size.
 * @param heat Heating flag.
 */
void setup_mock_params(int L, int R, int N, bool heat);


// --- Test Functions ---

/**
 * @brief Test suite for the Spin Lattice Find (SLF) function, covering
 *        general, corner, and edge cases with periodic boundary conditions.
 */
//void test_slf_lattice_find();
void test_slf_lattice_find_vs_precomputed();

#endif // SLF_TEST_H
