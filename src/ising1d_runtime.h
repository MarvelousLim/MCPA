#pragma once

#include <stdint.h>

struct ResamplingRngState {
    uint64_t state;
    uint64_t stream;
};

enum class IsingResampleStatus : int {
    ok = 0,
    no_next_shell = 1,
    terminal_full_cull = 2,
};

struct IsingResampleResult {
    IsingResampleStatus status;
    int old_U;
    int new_U;
    int n_cull;
    double culling_fraction;
};

void initializeResamplingRng(int seed);
ResamplingRngState getResamplingRngState();
void setResamplingRngState(ResamplingRngState state);
