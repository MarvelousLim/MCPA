#include <doctest/doctest.h>

#include "baxterwu_lib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

Params make_params(int L, int R, bool heat = false) {
    Params params{};
    params.L = L;
    params.N = L * L;
    params.R = R;
    params.blocks = 1;
    params.threads = R;
    params.nSteps = 1;
    params.heat = heat;
    params.singleIntRowByteSize = static_cast<size_t>(R) * sizeof(int);
    params.fullLatticeByteSize = static_cast<size_t>(R) * params.N * sizeof(int);
    params.replicaStatisticsByteSize = static_cast<size_t>(R) * sizeof(replicaStatistics);
    return params;
}

int direct_local_energy(int spin, const neiborsValues& n) {
    return -spin * (n.diag_left * n.up + n.diag_left * n.left
                  + n.diag_right * n.down + n.diag_right * n.right
                  + n.down * n.left + n.up * n.right);
}

int cpu_energy(const std::vector<int>& spins, const Params& params) {
    int sum = 0;
    for (int j = 0; j < params.N; ++j) {
        const neiborsIndexes index = SLF(j, params);
        const neiborsValues neighbors{
            spins[index.left], spins[index.right], spins[index.up],
            spins[index.down], spins[index.diag_left], spins[index.diag_right]
        };
        sum += local_energy(spins[j], neighbors);
    }
    return sum / 3;
}

int independent_baxter_wu_energy(const std::vector<int>& spins, int L) {
    const auto spin = [&spins, L](int x, int y) {
        const int wrapped_x = (x + L) % L;
        const int wrapped_y = (y + L) % L;
        return spins[wrapped_x + wrapped_y * L];
    };
    int energy = 0;
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            energy -= spin(x, y) * spin(x + 1, y) * spin(x, y + 1);
            energy -= spin(x + 1, y + 1) * spin(x + 1, y) * spin(x, y + 1);
        }
    }
    return energy;
}

std::vector<int> exact_l3_energies() {
    constexpr int L = 3;
    constexpr int N = L * L;
    std::vector<int> energies;
    energies.reserve(1 << N);
    for (int mask = 0; mask < (1 << N); ++mask) {
        std::vector<int> spins(N);
        for (int j = 0; j < N; ++j) spins[j] = (mask & (1 << j)) ? 1 : -1;
        energies.push_back(independent_baxter_wu_energy(spins, L));
    }
    return energies;
}

void check_exact_l3_culling(bool heat) {
    std::vector<int> energies = exact_l3_energies();
    const int R = static_cast<int>(energies.size());
    std::vector<int> order(R);
    std::vector<int> update(R);
    std::vector<int> family(R);
    mainMemoryPointers host{};
    host.E = energies.data();
    host.O = order.data();
    host.update = update.data();
    host.replica_family = family.data();

    Params params = make_params(3, R, heat);
    initialize_update_arrays(host, params);
    initialize_resampling_rng(7);
    int U = heat ? -20 : 20;
    int culled = -1;
    const double fraction = prepare_resample_arrays(host, params, &U, &culled);
    CHECK(U == (heat ? -18 : 18));
    CHECK(culled == 4);
    CHECK(fraction == doctest::Approx(4.0 / 512.0));
}

} // namespace

TEST_CASE("Local energy matches the exhaustive neighbor oracle") {
    for (int mask = 0; mask < 128; ++mask) {
        const int spin = (mask & 1) ? 1 : -1;
        const neiborsValues n{
            (mask & 2) ? 1 : -1,
            (mask & 4) ? 1 : -1,
            (mask & 8) ? 1 : -1,
            (mask & 16) ? 1 : -1,
            (mask & 32) ? 1 : -1,
            (mask & 64) ? 1 : -1,
        };
        CAPTURE(mask);
        CHECK(local_energy(spin, n) == direct_local_energy(spin, n));
    }
}

TEST_CASE("Periodic neighbors are correct at every site") {
    for (const int L : {3, 6, 36}) {
        const Params params = make_params(L, 1);
        for (int j = 0; j < params.N; ++j) {
            const int x = j % L;
            const int y = j / L;
            const auto index = [L](int xx, int yy) {
                return (xx + L) % L + ((yy + L) % L) * L;
            };
            const neiborsIndexes got = SLF(j, params);
            CAPTURE(L);
            CAPTURE(j);
            CAPTURE(x);
            CAPTURE(y);
            CHECK(got.left == index(x - 1, y));
            CHECK(got.right == index(x + 1, y));
            CHECK(got.up == index(x, y - 1));
            CHECK(got.down == index(x, y + 1));
            CHECK(got.diag_left == index(x - 1, y - 1));
            CHECK(got.diag_right == index(x + 1, y + 1));
        }
    }
}

TEST_CASE("Four Baxter-Wu ground branches have energy -2N") {
    const Params params = make_params(6, 1);
    const std::array<std::array<int, 3>, 4> branches{{
        {{1, 1, 1}}, {{1, -1, -1}}, {{-1, 1, -1}}, {{-1, -1, 1}}
    }};

    for (const auto& branch : branches) {
        std::vector<int> spins(params.N);
        for (int j = 0; j < params.N; ++j) {
            const int x = j % params.L;
            const int y = j / params.L;
            spins[j] = branch[(x + y) % 3];
        }
        CAPTURE(branch[0]);
        CAPTURE(branch[1]);
        CAPTURE(branch[2]);
        CHECK(cpu_energy(spins, params) == -2 * params.N);
    }
}

TEST_CASE("Replica ordering is correct in both MCPA directions") {
    const Params params = make_params(3, 7);
    std::array<int, 7> energies{{4, -2, 8, 8, 0, -6, 2}};
    std::array<int, 7> order{};
    std::iota(order.begin(), order.end(), 0);
    mainMemoryPointers host{};
    host.E = energies.data();
    host.O = order.data();

    quicksort(host, 0, params.R - 1, 1);
    for (int i = 1; i < params.R; ++i) {
        CHECK(energies[order[i - 1]] >= energies[order[i]]);
    }

    std::iota(order.begin(), order.end(), 0);
    quicksort(host, 0, params.R - 1, -1);
    for (int i = 1; i < params.R; ++i) {
        CHECK(energies[order[i - 1]] <= energies[order[i]]);
    }
}

TEST_CASE("Resampling selects the next strict energy boundary") {
    std::array<int, 6> energies{{10, 10, 8, 8, 6, 4}};
    std::array<int, 6> order{};
    std::array<int, 6> update{};
    std::array<int, 6> family{};
    mainMemoryPointers host{};
    host.E = energies.data();
    host.O = order.data();
    host.update = update.data();
    host.replica_family = family.data();

    Params params = make_params(3, 6, false);
    initialize_update_arrays(host, params);
    initialize_resampling_rng(7);
    int U = 12;
    int culled_replica_number = -1;
    const double cooling_fraction = prepare_resample_arrays(
        host, params, &U, &culled_replica_number);
    CHECK(U == 10);
    CHECK(cooling_fraction == doctest::Approx(2.0 / 6.0));
    CHECK(culled_replica_number == 2);

    energies = {{-10, -10, -8, -8, -6, -4}};
    params.heat = true;
    initialize_update_arrays(host, params);
    initialize_resampling_rng(7);
    U = -12;
    culled_replica_number = -1;
    const double heating_fraction = prepare_resample_arrays(
        host, params, &U, &culled_replica_number);
    CHECK(U == -10);
    CHECK(heating_fraction == doctest::Approx(2.0 / 6.0));
    CHECK(culled_replica_number == 2);
}

TEST_CASE("Family-size square cannot overflow 32-bit arithmetic") {
    constexpr int R = 65536;
    Params params = make_params(3, R);
    std::vector<int> families(R, 0);
    mainMemoryPointers host{};
    host.replica_family = families.data();

    CHECK(calc_family_avg_sq_size(host, params, 0) == doctest::Approx(1.0));
}

TEST_CASE("Aggregate statistics survive sums above signed 32-bit range") {
    constexpr int R = 65536;
    constexpr int U = -64;
    Params params = make_params(6, R);
    std::vector<int> energies(R, U);
    std::vector<replicaStatistics> statistics(R);
    for (auto& value : statistics) {
        value.flip_count = 50000;
        value.magnetization[0] = 40000;
        value.magnetization[1] = -40000;
        value.magnetization[2] = -40000;
        value.polarization[0] = 50000;
        value.polarization[1] = 50000;
        value.polarization[2] = 50000;
    }

    mainMemoryPointers host{};
    host.E = energies.data();
    host.replica_statistics = statistics.data();
    Files files{};
    files.agg_stats_file = std::tmpfile();
    REQUIRE(files.agg_stats_file != nullptr);

    print_agg_stats(host, params, files, U);
    std::rewind(files.agg_stats_file);
    std::array<char, 2048> buffer{};
    REQUIRE(std::fgets(buffer.data(), static_cast<int>(buffer.size()),
                       files.agg_stats_file) != nullptr);
    std::fclose(files.agg_stats_file);

    std::vector<std::string> fields;
    std::istringstream row(buffer.data());
    for (std::string field; std::getline(row, field, '\t');) fields.push_back(field);
    REQUIRE(fields.size() == 33);
    CHECK(std::stoi(fields[1]) == R);
    CHECK(std::stod(fields[2]) == doctest::Approx(50000.0));
    for (int column = 3; column <= 5; ++column)
        CHECK(std::stod(fields[column]) == doctest::Approx(40000.0));
    for (int column = 6; column <= 8; ++column)
        CHECK(std::stod(fields[column]) == doctest::Approx(50000.0));

    long long branch_total = 0;
    for (int column = 16; column <= 20; ++column)
        branch_total += std::stoll(fields[column]);
    CHECK(branch_total == R);
    CHECK(std::stoll(fields[17]) == R); // +-- branch
}

TEST_CASE("Exact L=3 density of states contains all 512 configurations") {
    const std::vector<int> energies = exact_l3_energies();
    std::map<int, int> counts;
    for (const int energy : energies) ++counts[energy];
    const std::map<int, int> expected{
        {-18, 4}, {-6, 72}, {-2, 180}, {2, 180}, {6, 72}, {18, 4}};
    CHECK(energies.size() == 512);
    CHECK(counts == expected);
}

TEST_CASE("Exact L=3 density of states obeys g(E)=g(-E)") {
    const std::vector<int> energies = exact_l3_energies();
    std::map<int, int> counts;
    for (const int energy : energies) ++counts[energy];
    for (const auto& [energy, count] : counts) {
        CAPTURE(energy);
        CHECK(counts.at(-energy) == count);
    }
}

TEST_CASE("Cooling culling factor matches the exact L=3 value 4/512") {
    check_exact_l3_culling(false);
}

TEST_CASE("Heating culling factor matches the exact L=3 value 4/512") {
    check_exact_l3_culling(true);
}

TEST_CASE("Run GPU metadata is stored once in a rectangular main table") {
    Files files{};
    files.main_file = std::tmpfile();
    files.agg_stats_file = std::tmpfile();
    files.detailed_stats_file = std::tmpfile();
    REQUIRE(files.main_file != nullptr);
    REQUIRE(files.agg_stats_file != nullptr);
    REQUIRE(files.detailed_stats_file != nullptr);

    initialize_print(files);
    const FamilyMetrics family{4, 1, 0.25, std::log(4.0), 4.0, 0.25};
    RunGpuMetadata metadata{};
    std::snprintf(metadata.name, sizeof(metadata.name), "%s", "Test GPU");
    metadata.compute_capability = 8.9;
    metadata.total_memory_bytes = 8000;
    metadata.free_memory_before_setup_bytes = 7000;
    metadata.free_memory_after_setup_bytes = 6000;
    metadata.cuda_driver_version = 13000;
    metadata.cuda_runtime_version = 12040;
    print_main_data(files, 18, 4.0 / 512.0, 0.25, 4, 0.125, family,
                    &metadata, 22, 7, 0.125, "frozen_1_over_acceptance");
    print_main_data(files, 6, 72.0 / 512.0, 0.25, 72, 0.250, family, nullptr);

    std::rewind(files.main_file);
    std::array<char, 4096> buffer{};
    std::vector<std::vector<std::string>> rows;
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), files.main_file)) {
        std::vector<std::string> fields;
        std::istringstream row(buffer.data());
        for (std::string field; std::getline(row, field, '\t');) {
            if (!field.empty() && field.back() == '\n') field.pop_back();
            fields.push_back(field);
        }
        rows.push_back(std::move(fields));
    }
    REQUIRE(rows.size() == 3);
    REQUIRE(rows[0].size() == 22);
    REQUIRE(rows[1].size() == 22);
    REQUIRE(rows[2].size() == 22);
    CHECK(rows[0][11] == "gpu_name");
    CHECK(rows[1][11] == "Test GPU");
    CHECK(rows[1][12] == "8.9");
    CHECK(rows[1][13] == "8000");
    CHECK(rows[1][14] == "7000");
    CHECK(rows[1][15] == "6000");
    for (int column = 11; column < 18; ++column) CHECK(rows[2][column] == "NA");
    CHECK(rows[0][18] == "equilibrate_ceiling");
    CHECK(rows[0][19] == "equilibrate_nsteps");
    CHECK(rows[0][20] == "population_acceptance_ratio");
    CHECK(rows[0][21] == "nsteps_policy");
    CHECK(rows[1][18] == "22");
    CHECK(rows[1][19] == "7");
    CHECK(rows[1][20] == "0.125");
    CHECK(rows[1][21] == "frozen_1_over_acceptance");

    std::fclose(files.detailed_stats_file);
    std::fclose(files.agg_stats_file);
    std::fclose(files.main_file);
}

TEST_CASE("Detailed output is a reproducible observable-independent hash sample") {
    constexpr int R = 16;
    constexpr int limit = 5;
    constexpr int U = -6;
    Params params = make_params(3, R, false);
    params.seed = 1234;

    std::array<int, R> energies{};
    std::array<int, R> families{};
    std::array<replicaStatistics, R> statistics{};
    for (int replica = 0; replica < R; ++replica) {
        energies[replica] = U;
        families[replica] = (replica * 3) % R;
        statistics[replica].flip_count = 100 + replica;
        for (int component = 0; component < 3; ++component) {
            statistics[replica].magnetization[component] = replica + component + 1;
            statistics[replica].polarization[component] = 2 * replica + component + 1;
            statistics[replica].order_structure_factor[component] =
                0.25 * replica + component;
        }
    }

    mainMemoryPointers host{};
    host.E = energies.data();
    host.replica_family = families.data();
    host.replica_statistics = statistics.data();
    Files files{};
    files.main_file = std::tmpfile();
    files.agg_stats_file = std::tmpfile();
    files.detailed_stats_file = std::tmpfile();
    REQUIRE(files.main_file != nullptr);
    REQUIRE(files.agg_stats_file != nullptr);
    REQUIRE(files.detailed_stats_file != nullptr);
    initialize_print(files);
    CHECK(print_detailed_stats(host, params, files, U, limit, families.data()) == limit);

    std::vector<std::pair<uint64_t, int>> expected;
    for (int replica = 0; replica < R; ++replica) {
        expected.emplace_back(
            detailed_sample_hash(params.seed, params.heat, U, replica), replica);
    }
    std::sort(expected.begin(), expected.end());
    expected.resize(limit);

    std::rewind(files.detailed_stats_file);
    std::array<char, 4096> buffer{};
    REQUIRE(std::fgets(buffer.data(), static_cast<int>(buffer.size()),
                       files.detailed_stats_file) != nullptr);
    int row_count = 0;
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()),
                      files.detailed_stats_file)) {
        std::vector<std::string> fields;
        std::istringstream row(buffer.data());
        for (std::string field; std::getline(row, field, '\t');) {
            if (!field.empty() && field.back() == '\n') field.pop_back();
            fields.push_back(field);
        }
        REQUIRE(fields.size() == 28);
        REQUIRE(row_count < limit);
        const int replica = std::stoi(fields[12]);
        CHECK(replica == expected[row_count].second);
        CHECK(std::stoi(fields[13]) == families[replica]);
        CHECK(std::stoi(fields[14]) == row_count);
        CHECK(fields[15] == [&] {
            char hash[32];
            std::snprintf(hash, sizeof(hash), "0x%016llx",
                          static_cast<unsigned long long>(expected[row_count].first));
            return std::string(hash);
        }());
        CHECK(std::stoi(fields[16]) == R);
        CHECK(std::stoi(fields[17]) == limit);
        CHECK(std::stod(fields[18]) == doctest::Approx(1.0 * limit / R));
        CHECK(std::stoi(fields[19]) == limit);
        CHECK(fields[20] == "lowest_splitmix64_seed_direction_energy_replica");
        ++row_count;
    }
    CHECK(row_count == limit);

    std::fclose(files.detailed_stats_file);
    std::fclose(files.agg_stats_file);
    std::fclose(files.main_file);
}
