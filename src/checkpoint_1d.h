#pragma once

#include "ising1d_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr std::size_t kIsing1DOutputFileCount = 3;

struct Ising1DCheckpointIdentity {
    int N = 0;
    int R = 0;
    int n_steps = 0;
    int seed = 0;
    int detailed_cap = 100;
    std::size_t philox_bytes = 0;
};

struct Ising1DCheckpointState {
    int U = 0;
    std::uint64_t completed_shells = 0;
    std::array<std::int64_t, kIsing1DOutputFileCount> output_offsets{};
    ResamplingRngState resampling_rng{};
    std::vector<char> spins;
    std::vector<int> energies;
    std::vector<int> families;
    std::vector<int> order;
    std::vector<std::uint8_t> philox;
};

enum class Ising1DCheckpointLoadStatus {
    not_found,
    loaded,
    done,
    invalid
};

std::string ising1d_checkpoint_path(const std::string& base,
                                    const char* suffix);

bool save_ising1d_checkpoint(const std::string& base,
                             const Ising1DCheckpointIdentity& identity,
                             const Ising1DCheckpointState& state,
                             std::string* error = nullptr);

Ising1DCheckpointLoadStatus load_ising1d_checkpoint(
    const std::string& base,
    const Ising1DCheckpointIdentity& identity,
    Ising1DCheckpointState* state,
    std::string* loaded_path = nullptr,
    std::string* error = nullptr);

bool mark_ising1d_checkpoint_done(const std::string& base,
                                  std::string* error = nullptr);
