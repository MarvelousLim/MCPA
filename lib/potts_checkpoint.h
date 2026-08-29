#pragma once

#include "potts_lib.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr std::size_t kPottsLiveOutputCount = 3;

struct PottsCheckpointIdentity {
    int L = 0;
    int N = 0;
    int R = 0;
    int n_steps = 0;
    int seed = 0;
    int q = 0;
    bool heat = false;
    int detailed_cap = 100;
    std::size_t philox_bytes = 0;
};

struct PottsCheckpointState {
    int U = 0;
    std::uint64_t completed_shells = 0;
    std::array<std::int64_t, kPottsLiveOutputCount> output_offsets{};
    PottsResamplingRngState resampling_rng{};
    std::vector<char> spins;
    std::vector<int> energies;
    std::vector<int> families;
    std::vector<int> order;
    std::vector<std::uint8_t> philox;
};

enum class PottsCheckpointLoadStatus {
    not_found,
    loaded,
    done,
    invalid
};

std::string potts_checkpoint_path(const std::string& base, const char* suffix);

bool validate_potts_checkpoint_state(const PottsCheckpointIdentity& identity,
                                     const PottsCheckpointState& state,
                                     std::string* error = nullptr);

bool save_potts_checkpoint(const std::string& base,
                           const PottsCheckpointIdentity& identity,
                           const PottsCheckpointState& state,
                           std::string* error = nullptr);

PottsCheckpointLoadStatus load_potts_checkpoint(
    const std::string& base,
    const PottsCheckpointIdentity& identity,
    PottsCheckpointState* state,
    std::string* loaded_path = nullptr,
    std::string* error = nullptr);

bool mark_potts_checkpoint_done(const std::string& base,
                                std::string* error = nullptr);
