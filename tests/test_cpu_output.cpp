#include "test_harness.h"

#include "ising1d_output.h"

#include <array>
#include <cmath>

ISING_TEST_CASE("Ising shell measurements preserve pre-resampling families") {
    constexpr int N = 4;
    constexpr int R = 4;
    const std::array<char, N * R> spins{{
        1, 1, 1, 1,
        1, 1, -1, -1,
        -1, -1, -1, -1,
        1, -1, 1, -1,
    }};
    const std::array<int, R> flips{{1, 2, 3, 4}};
    const std::array<int, R> energies{{-4, 0, -4, 4}};
    const std::array<int, R> pre_families{{0, 1, 0, 3}};

    const auto measured = measure_ising_replicas(
        spins.data(), flips.data(), N, R);
    ISING_REQUIRE(measured[0].magnetization == 4);
    ISING_REQUIRE(measured[1].magnetization == 0);
    ISING_REQUIRE(measured[2].magnetization == -4);
    ISING_REQUIRE(measured[3].magnetization_squared == 0);

    const IsingShellAggregate aggregate = aggregate_ising_shell(
        measured, energies.data(), pre_families.data(), R, -4, N, 2);
    ISING_REQUIRE(aggregate.shell_replica_count == 2);
    ISING_REQUIRE(aggregate.accepted_flip_sum == 4);
    ISING_REQUIRE(aggregate.accepted_flip_mean == 2.0);
    ISING_REQUIRE(aggregate.accepted_flip_rate == 0.25);
    ISING_REQUIRE(aggregate.magnetization_sum == 0);
    ISING_REQUIRE(aggregate.absolute_magnetization_sum == 8);
    ISING_REQUIRE(aggregate.magnetization_squared_sum == 32);
    ISING_REQUIRE(aggregate.pre_resampling_family.count == 1);
    ISING_REQUIRE(aggregate.pre_resampling_family.max_size == 2);
    ISING_REQUIRE(std::abs(aggregate.pre_resampling_family.max_fraction - 1.0)
                  < 1e-15);
}
