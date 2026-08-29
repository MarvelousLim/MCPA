#include "test_harness.hpp"

#include "mcpa/ising2d_resampling.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace {

using namespace mcpa::ising2d;

template <std::size_t R>
struct ResamplingFixture {
    std::array<Energy, R> energies{};
    std::array<ReplicaIndex, R> order{};
    std::array<ReplicaIndex, R> parents{};
    std::array<FamilyId, R> families{};

    ResamplingFixture() {
        for (std::size_t replica = 0; replica < R; ++replica)
            families[replica] = static_cast<FamilyId>(replica);
    }

    HostResamplingView view() {
        return HostResamplingView{energies.data(), order.data(), parents.data(),
                                  families.data(), static_cast<int>(R)};
    }
};

} // namespace

ISING_TEST_CASE("Resampling PCG state replays parents and families exactly") {
    ResamplingFixture<7> first;
    first.energies = {{12, 8, 8, 4, 0, -4, -8}};
    ResamplingRngState rng = seed_resampling_rng(3141592653ULL);
    const ResamplingRngState initial = rng;
    const ResampleResult first_result = resample_strict(
        first.view(), 13, WalkDirection::cooling, rng);
    ISING_REQUIRE(first_result.status == ResampleStatus::ok);
    ISING_REQUIRE(first_result.boundary == 12);
    ISING_REQUIRE(first_result.culled == 1);
    const auto expected_parents = first.parents;
    const auto expected_families = first.families;
    const ResamplingRngState expected_rng = rng;

    ResamplingFixture<7> replay;
    replay.energies = first.energies;
    rng = initial;
    const ResampleResult replay_result = resample_strict(
        replay.view(), 13, WalkDirection::cooling, rng);
    ISING_REQUIRE(replay_result.status == ResampleStatus::ok);
    ISING_REQUIRE(replay.parents == expected_parents);
    ISING_REQUIRE(replay.families == expected_families);
    ISING_REQUIRE(rng.state == expected_rng.state);
    ISING_REQUIRE(rng.stream == expected_rng.stream);
    const ReplicaIndex destination = replay.order[0];
    ISING_REQUIRE(replay.energies[replay.parents[destination]]
                  < replay_result.boundary);
    ISING_REQUIRE(replay.families[destination]
                  == replay.families[replay.parents[destination]]);
}

ISING_TEST_CASE("Strict resampling distinguishes terminal and no-next-shell states") {
    for (const WalkDirection direction : {WalkDirection::cooling,
                                          WalkDirection::heating}) {
        ResamplingFixture<4> fixture;
        fixture.energies = direction == WalkDirection::cooling
                               ? std::array<Energy, 4>{{-18, -18, -18, -18}}
                               : std::array<Energy, 4>{{6, 6, 6, 6}};
        ResamplingRngState rng = seed_resampling_rng(41);
        const ResamplingRngState before = rng;
        const Energy outside = direction == WalkDirection::cooling ? -17 : 5;
        const ResampleResult terminal = resample_strict(
            fixture.view(), outside, direction, rng);
        ISING_REQUIRE(terminal.status == ResampleStatus::terminal_full_cull);
        ISING_REQUIRE(terminal.culled == 4);
        ISING_REQUIRE(terminal.culling_fraction == 1.0);
        ISING_REQUIRE(rng.state == before.state);
        ISING_REQUIRE(rng.stream == before.stream);
        for (int replica = 0; replica < 4; ++replica) {
            ISING_REQUIRE(fixture.parents[replica] == replica);
            ISING_REQUIRE(fixture.families[replica] == replica);
        }

        const ResampleResult no_next = resample_strict(
            fixture.view(), terminal.boundary, direction, rng);
        ISING_REQUIRE(no_next.status == ResampleStatus::no_next_shell);
        ISING_REQUIRE(no_next.boundary == terminal.boundary);
        ISING_REQUIRE(no_next.culled == 0);
    }
}

ISING_TEST_CASE("Int64 resampling shells do not narrow or overflow") {
    ResamplingFixture<6> cooling;
    cooling.energies = {{
        std::numeric_limits<Energy>::max() - 3,
        std::numeric_limits<Energy>::max() - 7,
        0,
        std::numeric_limits<Energy>::min() + 11,
        std::numeric_limits<Energy>::min() + 19,
        -4,
    }};
    ResamplingRngState cool_rng = seed_resampling_rng(UINT64_MAX);
    const ResampleResult cool = resample_strict(
        cooling.view(), std::numeric_limits<Energy>::max() - 1,
        WalkDirection::cooling, cool_rng);
    ISING_REQUIRE(cool.status == ResampleStatus::ok);
    ISING_REQUIRE(cool.boundary == std::numeric_limits<Energy>::max() - 3);
    ISING_REQUIRE(cool.culled == 1);

    ResamplingFixture<6> heating;
    heating.energies = cooling.energies;
    ResamplingRngState heat_rng = seed_resampling_rng(UINT64_MAX - 1);
    const ResampleResult heat = resample_strict(
        heating.view(), std::numeric_limits<Energy>::min() + 1,
        WalkDirection::heating, heat_rng);
    ISING_REQUIRE(heat.status == ResampleStatus::ok);
    ISING_REQUIRE(heat.boundary == std::numeric_limits<Energy>::min() + 11);
    ISING_REQUIRE(heat.culled == 1);

    const FamilyMetrics metrics = calculate_family_metrics(
        cooling.families.data(), static_cast<int>(cooling.families.size()));
    ISING_REQUIRE(metrics.count >= 1);
    ISING_REQUIRE(metrics.simpson_concentration > 0.0);
    ISING_REQUIRE(metrics.simpson_concentration <= 1.0);
}
