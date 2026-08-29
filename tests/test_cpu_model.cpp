#include "test_harness.h"

#include "blumeCapel_lib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {

using JointKey = std::tuple<int, int, int>;
using JointDos = std::map<JointKey, int>;

struct EnergyParts {
    int e_j = 0;
    int e_delta = 0;
    int magnetization = 0;
};

class ScopedStdoutSilence {
public:
    ScopedStdoutSilence() : saved_fd_(dup(fileno(stdout))), sink_(tmpfile()) {
        BC_REQUIRE(saved_fd_ >= 0);
        BC_REQUIRE(sink_ != nullptr);
        std::fflush(stdout);
        BC_REQUIRE(dup2(fileno(sink_), fileno(stdout)) >= 0);
    }

    ~ScopedStdoutSilence() {
        std::fflush(stdout);
        if (saved_fd_ >= 0) {
            dup2(saved_fd_, fileno(stdout));
            close(saved_fd_);
        }
        if (sink_) std::fclose(sink_);
    }

    ScopedStdoutSilence(const ScopedStdoutSilence&) = delete;
    ScopedStdoutSilence& operator=(const ScopedStdoutSilence&) = delete;

private:
    int saved_fd_;
    FILE* sink_;
};

Params make_params(int L, int R = 1, bool heat = false) {
    Params params{};
    params.L = L;
    params.N = L * L;
    params.R = R;
    params.heat = heat;
    params.D_num = 0;
    params.D_denum = 1;
    return params;
}

EnergyParts independent_energy(const std::vector<int>& spins, int L) {
    EnergyParts result{};
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = x + y * L;
            const int right = ((x + 1) % L) + y * L;
            const int down = x + ((y + 1) % L) * L;
            const int sigma = spins[site];
            result.e_j -= sigma * (spins[right] + spins[down]);
            result.e_delta += sigma * sigma;
            result.magnetization += sigma;
        }
    }
    return result;
}

EnergyParts production_host_energy(const std::vector<int>& spins, int L) {
    const Params params = make_params(L);
    EnergyParts result{};
    int doubled_e_j = 0;
    for (int site = 0; site < params.N; ++site) {
        const neighborsIndexes indexes = SLF(site, params);
        const neighborsValues values{
            spins[indexes.left], spins[indexes.right],
            spins[indexes.up], spins[indexes.down]};
        const int sigma = spins[site];
        doubled_e_j += local_energy_j(sigma, values);
        result.e_delta += local_energy_delta(sigma);
        result.magnetization += sigma;
    }
    result.e_j = doubled_e_j / 2;
    return result;
}

std::vector<int> decode_configuration(int code, int N) {
    std::vector<int> spins(static_cast<std::size_t>(N));
    for (int site = 0; site < N; ++site) {
        spins[site] = code % 3 - 1;
        code /= 3;
    }
    return spins;
}

JointDos enumerate_l3_joint_dos() {
    constexpr int configuration_count = 19683; // 3^9
    JointDos dos;
    for (int code = 0; code < configuration_count; ++code) {
        const std::vector<int> spins = decode_configuration(code, 9);
        const EnergyParts parts = production_host_energy(spins, 3);
        const EnergyParts oracle = independent_energy(spins, 3);
        BC_REQUIRE(parts.e_j == oracle.e_j);
        BC_REQUIRE(parts.e_delta == oracle.e_delta);
        BC_REQUIRE(parts.magnetization == oracle.magnetization);
        ++dos[JointKey{parts.e_j, parts.e_delta, parts.magnetization}];
    }
    return dos;
}

int choose(int n, int k) {
    if (k < 0 || k > n) return 0;
    int value = 1;
    for (int i = 1; i <= k; ++i) value = value * (n - k + i) / i;
    return value;
}

void require_parts(const std::vector<int>& spins, int L,
                   int e_j, int e_delta, int magnetization) {
    const EnergyParts oracle = independent_energy(spins, L);
    const EnergyParts production = production_host_energy(spins, L);
    BC_REQUIRE(oracle.e_j == e_j);
    BC_REQUIRE(oracle.e_delta == e_delta);
    BC_REQUIRE(oracle.magnetization == magnetization);
    BC_REQUIRE(production.e_j == oracle.e_j);
    BC_REQUIRE(production.e_delta == oracle.e_delta);
    BC_REQUIRE(production.magnetization == oracle.magnetization);
}

} // namespace

BC_TEST_CASE("Exact L=3 joint DOS contains all 19683 states") {
    const JointDos dos = enumerate_l3_joint_dos();
    int count = 0;
    for (const auto& entry : dos) count += entry.second;
    BC_REQUIRE(count == 19683);
}

BC_TEST_CASE("Exact L=3 joint DOS obeys spin-flip symmetry") {
    const JointDos dos = enumerate_l3_joint_dos();
    for (const auto& entry : dos) {
        const auto [e_j, e_delta, magnetization] = entry.first;
        const auto opposite = dos.find(JointKey{e_j, e_delta, -magnetization});
        BC_REQUIRE(opposite != dos.end());
        BC_REQUIRE(opposite->second == entry.second);
    }
}

BC_TEST_CASE("Fixed occupancy counts follow choose(9,k) times 2^k") {
    const JointDos dos = enumerate_l3_joint_dos();
    std::array<int, 10> counts{};
    for (const auto& entry : dos) counts[std::get<1>(entry.first)] += entry.second;
    for (int occupied = 0; occupied <= 9; ++occupied) {
        const int expected = choose(9, occupied) * (1 << occupied);
        BC_REQUIRE(counts[occupied] == expected);
    }
}

BC_TEST_CASE("Constructed configurations have exact energy parts") {
    require_parts(std::vector<int>(9, 0), 3, 0, 0, 0);
    require_parts(std::vector<int>(9, 1), 3, -18, 9, 9);
    require_parts(std::vector<int>(9, -1), 3, -18, 9, -9);

    std::vector<int> single_plus(9, 0);
    single_plus[0] = 1;
    require_parts(single_plus, 3, 0, 1, 1);

    std::vector<int> one_zero(9, 1);
    one_zero[0] = 0;
    require_parts(one_zero, 3, -14, 8, 8);

    std::vector<int> one_flipped(9, 1);
    one_flipped[0] = -1;
    require_parts(one_flipped, 3, -10, 9, 7);

    std::vector<int> checkerboard(16);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) checkerboard[x + 4 * y] = ((x + y) % 2) ? -1 : 1;
    require_parts(checkerboard, 4, 32, 16, 0);
}

BC_TEST_CASE("Rational D parsing preserves ordering and ties") {
    int numerator = 0;
    int denominator = 0;
    BC_REQUIRE(parse_D_from_string("1.96", &numerator, &denominator) == 0);
    BC_REQUIRE(numerator == 49);
    BC_REQUIRE(denominator == 25);
    BC_REQUIRE(bc_energy_int(-18, 9, numerator, denominator) == -9);
    BC_REQUIRE(bc_energy_int(0, 0, numerator, denominator) == 0);

    BC_REQUIRE(parse_D_from_string("1.50", &numerator, &denominator) == 0);
    BC_REQUIRE(numerator == 3);
    BC_REQUIRE(denominator == 2);
    BC_REQUIRE(bc_energy_int(-3, 2, numerator, denominator)
               == bc_energy_int(0, 0, numerator, denominator));
    BC_REQUIRE(bc_energy_int(-4, 2, numerator, denominator)
               < bc_energy_int(0, 0, numerator, denominator));

    BC_REQUIRE(parse_D_from_string("2.00", &numerator, &denominator) == 0);
    BC_REQUIRE(numerator == 2);
    BC_REQUIRE(denominator == 1);
    BC_REQUIRE(bc_energy_int(-18, 9, numerator, denominator)
               == bc_energy_int(0, 0, numerator, denominator));
}

BC_TEST_CASE("Spin proposals stay in the three-state domain") {
    for (int old_spin = -1; old_spin <= 1; ++old_spin) {
        std::array<bool, 3> seen{};
        for (unsigned int draw = 0; draw < 16; ++draw) {
            const int proposed = bc_propose_spin(old_spin, draw);
            BC_REQUIRE(proposed >= -1);
            BC_REQUIRE(proposed <= 1);
            BC_REQUIRE(proposed != old_spin);
            seen[static_cast<std::size_t>(proposed + 1)] = true;
        }
        BC_REQUIRE(!seen[static_cast<std::size_t>(old_spin + 1)]);
        BC_REQUIRE(std::accumulate(seen.begin(), seen.end(), 0) == 2);
    }
}

BC_TEST_CASE("Local energy changes match the independent oracle") {
    for (int old_spin = -1; old_spin <= 1; ++old_spin) {
        for (int new_spin = -1; new_spin <= 1; ++new_spin) {
            if (new_spin == old_spin) continue;
            for (int code = 0; code < 81; ++code) {
                int remainder = code;
                neighborsValues neighbors{};
                int* values[] = {&neighbors.left, &neighbors.right,
                                 &neighbors.up, &neighbors.down};
                int neighbor_sum = 0;
                for (int* value : values) {
                    *value = remainder % 3 - 1;
                    remainder /= 3;
                    neighbor_sum += *value;
                }
                const int expected_delta_j = -(new_spin - old_spin) * neighbor_sum;
                const int expected_delta_d = new_spin * new_spin - old_spin * old_spin;
                BC_REQUIRE(local_energy_j(new_spin, neighbors)
                               - local_energy_j(old_spin, neighbors)
                           == expected_delta_j);
                BC_REQUIRE(local_energy_delta(new_spin)
                               - local_energy_delta(old_spin)
                           == expected_delta_d);
            }
        }
    }
}

BC_TEST_CASE("MCPA shell selection is strict in both directions") {
    ScopedStdoutSilence silence;
    std::array<int, 4> e_j{{6, 6, 4, 2}};
    std::array<int, 4> e_delta{};
    std::array<int, 4> order{{0, 1, 2, 3}};
    std::array<int, 4> update{};
    std::array<int, 4> family{{0, 1, 2, 3}};
    mainMemoryPointers host{};
    host.e_j = e_j.data();
    host.e_delta = e_delta.data();
    host.O = order.data();
    host.update = update.data();
    host.replica_family = family.data();

    Params cooling = make_params(3, 4, false);
    int ceiling = 6;
    initialize_resampling_rng(7);
    const double cooling_x = prepare_resample_arrays(host, cooling, &ceiling);
    BC_REQUIRE(ceiling == 4);
    BC_REQUIRE(std::abs(cooling_x - 0.75) < 1e-12);

    order = {{0, 1, 2, 3}};
    family = {{0, 1, 2, 3}};
    Params heating = make_params(3, 4, true);
    int floor = 2;
    initialize_resampling_rng(7);
    const double heating_x = prepare_resample_arrays(host, heating, &floor);
    BC_REQUIRE(floor == 4);
    BC_REQUIRE(std::abs(heating_x - 0.50) < 1e-12);
}

BC_TEST_CASE("Resampling RNG state restores parent choices exactly") {
    ScopedStdoutSilence silence;
    Params params = make_params(3, 5, false);
    std::array<int, 5> e_j{{8, 6, 4, 2, 0}};
    std::array<int, 5> e_delta{};
    std::array<int, 5> order{{0, 1, 2, 3, 4}};
    std::array<int, 5> update{{0, 0, 0, 0, 0}};
    std::array<int, 5> family{{0, 1, 2, 3, 4}};
    mainMemoryPointers host{};
    host.e_j = e_j.data();
    host.e_delta = e_delta.data();
    host.O = order.data();
    host.update = update.data();
    host.replica_family = family.data();

    initialize_resampling_rng(31415);
    const ResamplingRngState initial_rng = get_resampling_rng_state();
    int ceiling = 8;
    int n_culled = 0;
    BC_REQUIRE(prepare_resample_arrays(host, params, &ceiling, &n_culled) == 0.4);
    BC_REQUIRE(ceiling == 6);
    BC_REQUIRE(n_culled == 2);
    const std::array<int, 5> expected_update = update;
    const std::array<int, 5> expected_family = family;
    const ResamplingRngState expected_rng = get_resampling_rng_state();

    order = {{0, 1, 2, 3, 4}};
    update = {{0, 0, 0, 0, 0}};
    family = {{0, 1, 2, 3, 4}};
    set_resampling_rng_state(initial_rng);
    ceiling = 8;
    BC_REQUIRE(prepare_resample_arrays(host, params, &ceiling, &n_culled) == 0.4);
    BC_REQUIRE(update == expected_update);
    BC_REQUIRE(family == expected_family);
    const ResamplingRngState replay_rng = get_resampling_rng_state();
    BC_REQUIRE(replay_rng.state == expected_rng.state);
    BC_REQUIRE(replay_rng.stream == expected_rng.stream);

    for (int sorted_slot = 0; sorted_slot < n_culled; ++sorted_slot) {
        const int destination = order[sorted_slot];
        const int source = update[destination];
        BC_REQUIRE(e_j[source] < ceiling);
        BC_REQUIRE(family[destination] == source);
    }
}

BC_TEST_CASE("All-boundary populations return full culling without out-of-bounds access") {
    ScopedStdoutSilence silence;
    std::array<int, 4> e_j{{4, 4, 4, 4}};
    std::array<int, 4> e_delta{};
    std::array<int, 4> order{{0, 1, 2, 3}};
    std::array<int, 4> update{{-1, -1, -1, -1}};
    std::array<int, 4> family{{0, 1, 2, 3}};
    mainMemoryPointers host{};
    host.e_j = e_j.data();
    host.e_delta = e_delta.data();
    host.O = order.data();
    host.update = update.data();
    host.replica_family = family.data();

    Params cooling = make_params(3, 4, false);
    int ceiling = 6;
    BC_REQUIRE(prepare_resample_arrays(host, cooling, &ceiling) == 1.0);
    BC_REQUIRE(ceiling == 4);
    for (int i = 0; i < 4; ++i) BC_REQUIRE(update[i] == i);

    order = {{0, 1, 2, 3}};
    update = {{-1, -1, -1, -1}};
    Params heating = make_params(3, 4, true);
    int floor = 2;
    BC_REQUIRE(prepare_resample_arrays(host, heating, &floor) == 1.0);
    BC_REQUIRE(floor == 4);
    for (int i = 0; i < 4; ++i) BC_REQUIRE(update[i] == i);
}

BC_TEST_CASE("Aggregate statistics remain exact beyond 32-bit sums") {
    constexpr int R = 65536;
    Params params = make_params(3, R, false);
    params.nSteps = 1;
    std::vector<int> e_j(R, 4);
    std::vector<int> e_delta(R, 0);
    std::vector<replicaStatistics> stats(R);
    std::vector<int> family(R);
    std::iota(family.begin(), family.end(), 0);
    for (replicaStatistics& value : stats) {
        value.flip_count = 5;
        value.e_j = 50000;
        value.e_delta = 40000;
        value.m = -50000;
    }
    mainMemoryPointers host{};
    host.e_j = e_j.data();
    host.e_delta = e_delta.data();
    host.replica_statistics = stats.data();
    host.replica_family = family.data();

    FILE* output = tmpfile();
    BC_REQUIRE(output != nullptr);
    Files files{};
    files.agg_stats_file = output;
    print_agg_stats(host, params, files, 4, family.data());
    std::rewind(output);

    double energy = 0.0;
    int count = 0;
    double flip_rate = 0.0, mean_e_j = 0.0, mean_e_delta = 0.0, mean_m = 0.0;
    double family_concentration = 0.0;
    int exact_u = 0, d_num = 0, d_denum = 0, unique_families = 0;
    BC_REQUIRE(std::fscanf(output,
                          "%lf\t%d\t%lf\t%lf\t%lf\t%lf\t%d\t%d\t%d\t%lf\t%d",
                          &energy, &count, &flip_rate, &mean_e_j,
                          &mean_e_delta, &mean_m,
                          &exact_u, &d_num, &d_denum, &family_concentration,
                          &unique_families) == 11);
    std::fclose(output);
    BC_REQUIRE(energy == 4.0);
    BC_REQUIRE(count == R);
    BC_REQUIRE(std::abs(flip_rate - 55.5556) < 1e-4);
    BC_REQUIRE(mean_e_j == 50000.0);
    BC_REQUIRE(mean_e_delta == 40000.0);
    BC_REQUIRE(mean_m == -50000.0);
    BC_REQUIRE(exact_u == 4);
    BC_REQUIRE(d_num == 0);
    BC_REQUIRE(d_denum == 1);
    BC_REQUIRE(std::abs(family_concentration - 1.0 / R) < 1e-12);
    BC_REQUIRE(unique_families == R);
}

BC_TEST_CASE("Output keeps legacy columns and appends exact shell identity") {
    FILE* main_file = tmpfile();
    FILE* agg_file = tmpfile();
    FILE* detail_file = tmpfile();
    BC_REQUIRE(main_file != nullptr);
    BC_REQUIRE(agg_file != nullptr);
    BC_REQUIRE(detail_file != nullptr);
    Files files{main_file, agg_file, detail_file};
    initialize_print(files);
    GpuMetadata metadata{};
    std::snprintf(metadata.name, sizeof(metadata.name), "Test GPU");
    metadata.compute_major = 8;
    metadata.compute_minor = 9;
    metadata.cuda_runtime_version = 12040;
    metadata.cuda_driver_version = 12080;
    metadata.total_memory_bytes = 1000;
    metadata.free_memory_before_setup_bytes = 900;
    metadata.free_memory_after_setup_bytes = 700;
    print_main_data(files, -0.36, 2.0 / 3.0, 0.125, -9, 49, 25, 2,
                    0.25, 0.0625, 0.125, &metadata);
    print_main_data(files, -0.40, 0.5, 0.25, -10, 49, 25, 3,
                    0.5, 0.125, 0.25, nullptr);
    std::rewind(main_file);

    char header[1024]{};
    char row[1024]{};
    char later_row[1024]{};
    BC_REQUIRE(std::fgets(header, sizeof(header), main_file) != nullptr);
    BC_REQUIRE(std::fgets(row, sizeof(row), main_file) != nullptr);
    BC_REQUIRE(std::fgets(later_row, sizeof(later_row), main_file) != nullptr);
    BC_REQUIRE(std::strstr(header,
        "E\tculling_factor\treplica_family_avg_sq\tU_scaled\tD_num\tD_denum"
        "\tn_culled_exact\tculling_factor_exact") == header);

    double energy = 0.0, legacy_x = 0.0, rho = 0.0, exact_x = 0.0;
    double seconds = 0.0, pre_family = 0.0, post_family = 0.0;
    int exact_u = 0, d_num = 0, d_denum = 0, n_culled = 0;
    BC_REQUIRE(std::sscanf(row,
                           "%lf\t%lf\t%lf\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf",
                           &energy, &legacy_x, &rho, &exact_u, &d_num,
                           &d_denum, &n_culled, &exact_x, &seconds,
                           &pre_family, &post_family) == 11);
    BC_REQUIRE(energy == -0.36);
    BC_REQUIRE(std::abs(legacy_x - 2.0 / 3.0) < 1e-6);
    BC_REQUIRE(rho == 0.125);
    BC_REQUIRE(exact_u == -9);
    BC_REQUIRE(d_num == 49);
    BC_REQUIRE(d_denum == 25);
    BC_REQUIRE(n_culled == 2);
    BC_REQUIRE(std::abs(exact_x - 2.0 / 3.0) < 1e-15);
    BC_REQUIRE(seconds == 0.25);
    BC_REQUIRE(pre_family == 0.0625);
    BC_REQUIRE(post_family == 0.125);
    const auto column_count = [](const char* line) {
        return 1 + static_cast<int>(std::count(line, line + std::strlen(line), '\t'));
    };
    BC_REQUIRE(column_count(header) == 18);
    BC_REQUIRE(column_count(row) == 18);
    BC_REQUIRE(column_count(later_row) == 18);
    BC_REQUIRE(std::strstr(
        row, "\tTest GPU\t8.9\t12040\t1000\t900\t700\t12080\n") != nullptr);
    BC_REQUIRE(std::strstr(
        later_row, "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n") != nullptr);

    std::rewind(agg_file);
    std::rewind(detail_file);
    BC_REQUIRE(std::fgets(header, sizeof(header), agg_file) != nullptr);
    BC_REQUIRE(column_count(header) == 11);
    BC_REQUIRE(std::fgets(header, sizeof(header), detail_file) != nullptr);
    BC_REQUIRE(column_count(header) == 11);

    std::fclose(main_file);
    std::fclose(agg_file);
    std::fclose(detail_file);
}

BC_TEST_CASE("Detailed cap modes preserve pre-resampling family evidence") {
    Params params = make_params(3, 5, false);
    params.nSteps = 1;
    std::array<int, 5> e_j{{4, 4, 4, 4, 4}};
    std::array<int, 5> e_delta{{0, 0, 0, 0, 0}};
    std::array<int, 5> measured_family{{4, 3, 2, 1, 0}};
    std::array<int, 5> post_family{{0, 0, 0, 0, 0}};
    std::array<replicaStatistics, 5> stats{};
    for (int i = 0; i < 5; ++i) {
        stats[i].flip_count = i;
        stats[i].e_j = 4;
        stats[i].e_delta = 0;
        stats[i].m = i - 2;
    }
    mainMemoryPointers host{};
    host.e_j = e_j.data();
    host.e_delta = e_delta.data();
    host.replica_family = post_family.data();
    host.replica_statistics = stats.data();

    const auto line_count = [](FILE* file) {
        std::rewind(file);
        int count = 0;
        char line[512];
        while (std::fgets(line, sizeof(line), file)) ++count;
        return count;
    };

    FILE* none = std::tmpfile();
    BC_REQUIRE(none != nullptr);
    Files none_files{};
    none_files.detailed_stats_file = none;
    print_detailed_stats(host, params, none_files, 4, measured_family.data(), 0);
    BC_REQUIRE(line_count(none) == 0);
    std::fclose(none);

    FILE* prefix = std::tmpfile();
    BC_REQUIRE(prefix != nullptr);
    Files prefix_files{};
    prefix_files.detailed_stats_file = prefix;
    print_detailed_stats(host, params, prefix_files, 4,
                         measured_family.data(), 2);
    BC_REQUIRE(line_count(prefix) == 2);
    std::rewind(prefix);
    for (int expected_family : {4, 3}) {
        char line[512]{};
        BC_REQUIRE(std::fgets(line, sizeof(line), prefix) != nullptr);
        double energy = 0.0, flip = 0.0;
        int ej = 0, ed = 0, m = 0, U = 0, d_num = 0, d_den = 0;
        int family = -1, total = 0;
        char policy[32]{};
        BC_REQUIRE(std::sscanf(
            line, "%lf\t%lf\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%31s",
            &energy, &flip, &ej, &ed, &m, &U, &d_num, &d_den, &family,
            &total, policy) == 11);
        BC_REQUIRE(ej == 4);
        BC_REQUIRE(ed == 0);
        BC_REQUIRE(family == expected_family);
        BC_REQUIRE(total == 5);
        BC_REQUIRE(std::strcmp(policy, "prefix_first_N") == 0);
    }
    std::fclose(prefix);

    FILE* full = std::tmpfile();
    BC_REQUIRE(full != nullptr);
    Files full_files{};
    full_files.detailed_stats_file = full;
    print_detailed_stats(host, params, full_files, 4,
                         measured_family.data(), -1);
    BC_REQUIRE(line_count(full) == 5);
    std::rewind(full);
    char line[512]{};
    while (std::fgets(line, sizeof(line), full))
        BC_REQUIRE(std::strstr(line, "\tall_matching\n") != nullptr);
    std::fclose(full);
}

BC_TEST_CASE("Periodic square-lattice neighbors are exact") {
    for (int L : {3, 4}) {
        const Params params = make_params(L);
        for (int site = 0; site < params.N; ++site) {
            const int x = site % L;
            const int y = site / L;
            const neighborsIndexes neighbors = SLF(site, params);
            BC_REQUIRE(neighbors.left == ((x - 1 + L) % L) + y * L);
            BC_REQUIRE(neighbors.right == ((x + 1) % L) + y * L);
            BC_REQUIRE(neighbors.up == x + ((y - 1 + L) % L) * L);
            BC_REQUIRE(neighbors.down == x + ((y + 1) % L) * L);
            BC_REQUIRE(neighbors.left != neighbors.right);
            BC_REQUIRE(neighbors.up != neighbors.down);
        }
    }
}
