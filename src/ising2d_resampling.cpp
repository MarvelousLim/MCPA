#include "mcpa/ising2d_resampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mcpa::ising2d {
namespace {

constexpr std::uint64_t pcg_multiplier = 6364136223846793005ULL;
constexpr std::uint64_t seed_mixer = 0x9e3779b97f4a7c15ULL;

void validate_host(HostResamplingView host) {
    if (host.replicas <= 0 || host.energies == nullptr || host.order == nullptr
        || host.parents == nullptr || host.families == nullptr)
        throw std::invalid_argument("2D Ising resampling requires complete nonempty host arrays");
    for (int replica = 0; replica < host.replicas; ++replica) {
        if (host.families[replica] < 0
            || host.families[replica] >= host.replicas)
            throw std::invalid_argument("2D Ising family id is outside [0,R)");
    }
}

} // namespace

std::uint32_t next_resampling_random(ResamplingRngState& rng) noexcept {
    const std::uint64_t old_state = rng.state;
    rng.state = old_state * pcg_multiplier + (rng.stream | 1ULL);
    const auto xorshifted = static_cast<std::uint32_t>(
        ((old_state >> 18U) ^ old_state) >> 27U);
    const auto rotation = static_cast<std::uint32_t>(old_state >> 59U);
    return (xorshifted >> rotation)
         | (xorshifted << ((0U - rotation) & 31U));
}

ResamplingRngState seed_resampling_rng(std::uint64_t seed) noexcept {
    ResamplingRngState rng{0, (seed << 1U) | 1U};
    (void)next_resampling_random(rng);
    rng.state += seed ^ seed_mixer;
    (void)next_resampling_random(rng);
    return rng;
}

const char* resample_status_name(ResampleStatus status) noexcept {
    switch (status) {
        case ResampleStatus::ok: return "ok";
        case ResampleStatus::no_next_shell: return "no_next_shell";
        case ResampleStatus::terminal_full_cull: return "terminal_full_cull";
    }
    return "unknown";
}

ResampleResult resample_strict(HostResamplingView host, Energy boundary,
                               WalkDirection direction,
                               ResamplingRngState& rng) {
    validate_host(host);
    std::iota(host.order, host.order + host.replicas, ReplicaIndex{0});
    std::iota(host.parents, host.parents + host.replicas, ReplicaIndex{0});

    std::stable_sort(host.order, host.order + host.replicas,
        [&](ReplicaIndex left, ReplicaIndex right) {
            if (direction == WalkDirection::cooling)
                return host.energies[left] > host.energies[right];
            return host.energies[left] < host.energies[right];
        });

    bool found = false;
    Energy next_boundary = boundary;
    for (int rank = 0; rank < host.replicas; ++rank) {
        const Energy candidate = host.energies[host.order[rank]];
        if (strict_constraint(candidate, boundary, direction)) {
            next_boundary = candidate;
            found = true;
            break;
        }
    }
    if (!found) {
        return ResampleResult{ResampleStatus::no_next_shell, boundary, boundary,
                              0, 0.0};
    }

    int culled = 0;
    while (culled < host.replicas) {
        const Energy energy = host.energies[host.order[culled]];
        const bool remove = direction == WalkDirection::cooling
                                ? energy >= next_boundary
                                : energy <= next_boundary;
        if (!remove) break;
        ++culled;
    }

    const double fraction = static_cast<double>(culled)
                          / static_cast<double>(host.replicas);
    if (culled == host.replicas) {
        return ResampleResult{ResampleStatus::terminal_full_cull, boundary,
                              next_boundary,
                              static_cast<std::uint64_t>(culled), fraction};
    }

    const auto survivor_count = static_cast<std::uint32_t>(host.replicas - culled);
    for (int rank = 0; rank < culled; ++rank) {
        const std::uint32_t draw = next_resampling_random(rng);
        const int parent_rank = culled
                              + static_cast<int>(draw % survivor_count);
        const ReplicaIndex destination = host.order[rank];
        const ReplicaIndex parent = host.order[parent_rank];
        host.parents[destination] = parent;
        host.families[destination] = host.families[parent];
    }
    return ResampleResult{ResampleStatus::ok, boundary, next_boundary,
                          static_cast<std::uint64_t>(culled), fraction};
}

FamilyMetrics calculate_family_metrics(const FamilyId* families, int replicas) {
    if (families == nullptr || replicas <= 0)
        throw std::invalid_argument("2D Ising genealogy requires families and R > 0");
    std::vector<int> histogram(static_cast<std::size_t>(replicas), 0);
    for (int replica = 0; replica < replicas; ++replica) {
        const FamilyId family = families[replica];
        if (family < 0 || family >= replicas)
            throw std::invalid_argument("2D Ising family id is outside [0,R)");
        ++histogram[family];
    }

    FamilyMetrics result{};
    for (const int size : histogram) {
        if (size == 0) continue;
        ++result.count;
        result.max_size = std::max(result.max_size, size);
        const double fraction = static_cast<double>(size) / replicas;
        result.shannon_entropy -= fraction * std::log(fraction);
        result.simpson_concentration += fraction * fraction;
    }
    result.max_fraction = static_cast<double>(result.max_size) / replicas;
    result.effective_shannon = std::exp(result.shannon_entropy);
    return result;
}

} // namespace mcpa::ising2d
