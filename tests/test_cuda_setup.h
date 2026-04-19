#pragma once
#ifndef TEST_CUDA_SETUP_H
#define TEST_CUDA_SETUP_H

#include <cstddef>

void test_setup_curand_states();
void test_initialize_population_random();
void test_initialize_population_sublattice();
void test_initialize_population_strips();
void test_initialize_update_arrays();

#endif