#pragma once

#include "mcpa/ising2d_model.hpp"

#include <cstdint>

namespace mcpa::ising2d {

using ReplicaIndex = std::int32_t;
using FamilyId = std::int32_t;

struct ResamplingRngState {
    std::uint64_t state;
    std::uint64_t stream;
};

// Complete, checkpoint-ready PCG32 state. No libc RNG state is used.
[[nodiscard]] ResamplingRngState seed_resampling_rng(std::uint64_t seed) noexcept;
[[nodiscard]] std::uint32_t next_resampling_random(
    ResamplingRngState& state) noexcept;

enum class ResampleStatus {
    ok,
    no_next_shell,
    terminal_full_cull,
};

[[nodiscard]] const char* resample_status_name(ResampleStatus status) noexcept;

// Non-owning host arrays. resample_strict rewrites order and parents, and
// relabels only culled destination families when a surviving parent exists.
struct HostResamplingView {
    const Energy* energies;
    ReplicaIndex* order;
    ReplicaIndex* parents;
    FamilyId* families;
    int replicas;
};

struct ResampleResult {
    ResampleStatus status;
    Energy previous_boundary;
    Energy boundary;
    std::uint64_t culled;
    double culling_fraction;
};

// Cooling sorts high-to-low and advances to max(E < old U). Heating sorts
// low-to-high and advances to min(E > old U). At the selected shell, cooling
// culls E >= U and heating culls E <= U. Throws on malformed host memory.
[[nodiscard]] ResampleResult resample_strict(
    HostResamplingView host, Energy boundary, WalkDirection direction,
    ResamplingRngState& rng);

struct FamilyMetrics {
    int count;
    int max_size;
    double max_fraction;
    double shannon_entropy;
    double effective_shannon;
    double simpson_concentration;
};

[[nodiscard]] FamilyMetrics calculate_family_metrics(
    const FamilyId* families, int replicas);

} // namespace mcpa::ising2d
