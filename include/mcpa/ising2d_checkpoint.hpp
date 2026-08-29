#pragma once

#include "mcpa/ising2d_model.hpp"
#include "mcpa/ising2d_resampling.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mcpa::ising2d {

struct CheckpointIdentity {
    int linear_size;
    int site_count;
    int replicas;
    int sweeps;
    std::uint64_t seed;
    WalkDirection direction;
    int detailed_cap;
    std::size_t philox_state_bytes;
};

struct CheckpointState {
    Energy boundary;
    std::uint64_t completed_shells;
    std::array<std::uint64_t, 3> output_offsets;
    ResamplingRngState resampling_rng;
    std::vector<Spin> spins;
    std::vector<Energy> energies;
    std::vector<FamilyId> families;
    std::vector<ReplicaIndex> order;
    std::vector<unsigned char> philox;
};

struct CheckpointFiles {
    std::filesystem::path current;
    std::filesystem::path previous;
    std::filesystem::path temporary;
    std::filesystem::path done;
};

enum class CheckpointSource {
    none,
    current,
    previous,
    temporary,
};

struct CheckpointLoadResult {
    bool found = false;
    bool had_candidates = false;
    CheckpointSource source = CheckpointSource::none;
    CheckpointState state{};
    std::string diagnostics;
};

[[nodiscard]] CheckpointFiles make_checkpoint_files(
    const std::filesystem::path& directory, const std::string& basename);

// Throws std::invalid_argument when identity or state is not physically and
// structurally valid. Energy validation is a complete H=-sum bond recompute.
void validate_checkpoint_state(const CheckpointIdentity& identity,
                               const CheckpointState& state);

// Writes and fsyncs temporary, rotates current to previous, promotes temporary,
// then fsyncs the directory. Current and previous generations are retained.
void save_checkpoint(const CheckpointFiles& files,
                     const CheckpointIdentity& identity,
                     const CheckpointState& state);

// Valid candidates among current, previous, and temporary are semantically
// checked; the greatest completed-shell count wins, with current preferred on
// ties. CRC or semantic failure in one generation does not hide another.
[[nodiscard]] CheckpointLoadResult load_checkpoint(
    const CheckpointFiles& files, const CheckpointIdentity& identity);

// A present marker with a different run identity is an error, not a fresh run.
[[nodiscard]] bool checkpoint_is_done(const CheckpointFiles& files,
                                      const CheckpointIdentity& identity);
void mark_checkpoint_done(const CheckpointFiles& files,
                          const CheckpointIdentity& identity);

} // namespace mcpa::ising2d
