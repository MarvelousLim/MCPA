#include <doctest/doctest.h>

#include "baxterwu_lib.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

Params diagnostic_params(int L, int R) {
    Params params{};
    params.L = L;
    params.N = L * L;
    params.R = R;
    params.blocks = 1;
    params.threads = R;
    params.replicaStatisticsByteSize = static_cast<size_t>(R) * sizeof(replicaStatistics);
    return params;
}

} // namespace

TEST_CASE("Genealogy metrics identify unique and collapsed populations") {
    const std::array<int, 4> unique{{0, 1, 2, 3}};
    const FamilyMetrics unique_metrics = calc_family_metrics(unique.data(), 4);
    CHECK(unique_metrics.count == 4);
    CHECK(unique_metrics.max_size == 1);
    CHECK(unique_metrics.max_fraction == doctest::Approx(0.25));
    CHECK(unique_metrics.shannon_entropy == doctest::Approx(std::log(4.0)));
    CHECK(unique_metrics.effective_shannon == doctest::Approx(4.0));
    CHECK(unique_metrics.simpson_concentration == doctest::Approx(0.25));

    const std::array<int, 4> collapsed{{2, 2, 2, 2}};
    const FamilyMetrics collapsed_metrics = calc_family_metrics(collapsed.data(), 4);
    CHECK(collapsed_metrics.count == 1);
    CHECK(collapsed_metrics.max_size == 4);
    CHECK(collapsed_metrics.max_fraction == doctest::Approx(1.0));
    CHECK(collapsed_metrics.shannon_entropy == doctest::Approx(0.0));
    CHECK(collapsed_metrics.effective_shannon == doctest::Approx(1.0));
    CHECK(collapsed_metrics.simpson_concentration == doctest::Approx(1.0));
}

TEST_CASE("Family diagnostics use measured pre-resampling identities") {
    constexpr int R = 4;
    constexpr int U = -60;
    Params params = diagnostic_params(6, R);
    std::array<int, R> energies{{U, U, U, U}};
    std::array<int, R> measured_families{{0, 0, 1, 1}};
    std::array<int, R> future_parent_families{{1, 1, 1, 1}};
    std::array<replicaStatistics, R> statistics{};

    for (int i = 0; i < R; ++i) {
        const int m = i < 2 ? 6 : 12;
        statistics[i].magnetization[0] = m;
        statistics[i].polarization[0] = i < 2 ? 18 : 36;
        for (int q = 0; q < 3; ++q)
            statistics[i].order_structure_factor[q] = i < 2 ? 2.0 : 6.0;
    }

    mainMemoryPointers host{};
    host.E = energies.data();
    host.replica_statistics = statistics.data();
    host.replica_family = future_parent_families.data();

    const AggregateDiagnostics measured = calc_aggregate_diagnostics(
        host, params, U, measured_families.data());
    CHECK(measured.family_count_at_E == 2);
    CHECK(measured.family_max_fraction_at_E == doctest::Approx(0.5));
    const double family_mean_difference = std::sqrt(3.0) * (12.0 - 6.0) / params.N;
    CHECK(measured.family_var_mean_m_s
          == doctest::Approx(family_mean_difference * family_mean_difference / 4.0));
    CHECK(measured.family_var_mean_order_sf_kmin == doctest::Approx(4.0));
    CHECK(measured.family_cov_mean_order_sf0_kmin > 0.0);

    const AggregateDiagnostics wrongly_relabelled = calc_aggregate_diagnostics(
        host, params, U, future_parent_families.data());
    CHECK(wrongly_relabelled.family_count_at_E == 1);
    CHECK(std::isnan(wrongly_relabelled.family_var_mean_m_s));
    CHECK(std::isnan(wrongly_relabelled.family_var_mean_order_sf_kmin));
}

TEST_CASE("Branch-invariant scalar means equal one in every ground branch") {
    constexpr int R = 4;
    constexpr int L = 6;
    constexpr int N = L * L;
    constexpr int U = -2 * N;
    Params params = diagnostic_params(L, R);
    const std::array<std::array<int, 3>, R> branches{{
        {{1, 1, 1}}, {{1, -1, -1}}, {{-1, 1, -1}}, {{-1, -1, 1}}
    }};
    std::array<int, R> energies{{U, U, U, U}};
    std::array<int, R> families{{0, 1, 2, 3}};
    std::array<replicaStatistics, R> statistics{};
    for (int r = 0; r < R; ++r) {
        for (int component = 0; component < 3; ++component) {
            statistics[r].magnetization[component] = branches[r][component] * (N / 3);
            statistics[r].polarization[component] = N;
        }
    }
    mainMemoryPointers host{};
    host.E = energies.data();
    host.replica_family = families.data();
    host.replica_statistics = statistics.data();
    const AggregateDiagnostics diagnostics = calc_aggregate_diagnostics(
        host, params, U, families.data());
    CHECK(diagnostics.m_s_mean == doctest::Approx(1.0));
    CHECK(diagnostics.p_s_mean == doctest::Approx(1.0));
}
