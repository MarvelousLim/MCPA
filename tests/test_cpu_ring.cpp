#include "test_harness.h"

#include "ising1d_runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

IsingResampleResult resample(int* energies, int* order, int* update,
                             int* replica_family, int replicas,
                             int* ceiling, bool heat);

namespace {

using Count = std::int64_t;
using DensityOfStates = std::map<int, Count>;

Count choose(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    Count result = 1;
    for (int i = 1; i <= k; ++i) result = result * (n - k + i) / i;
    return result;
}

std::vector<int> decode_spins(std::uint64_t code, int N) {
    std::vector<int> spins(static_cast<std::size_t>(N));
    for (int site = 0; site < N; ++site)
        spins[site] = ((code >> site) & 1U) ? 1 : -1;
    return spins;
}

int right_neighbor(int site, int N) {
    return (site + 1) % N;
}

int left_neighbor(int site, int N) {
    return (site - 1 + N) % N;
}

int wall_count(const std::vector<int>& spins) {
    const int N = static_cast<int>(spins.size());
    int walls = 0;
    for (int site = 0; site < N; ++site)
        walls += spins[site] != spins[right_neighbor(site, N)];
    return walls;
}

int ring_energy(const std::vector<int>& spins) {
    const int N = static_cast<int>(spins.size());
    int energy = 0;
    for (int site = 0; site < N; ++site)
        energy -= spins[site] * spins[right_neighbor(site, N)];
    return energy;
}

DensityOfStates exhaustive_dos(int N) {
    DensityOfStates dos;
    const std::uint64_t state_count = std::uint64_t{1} << N;
    for (std::uint64_t code = 0; code < state_count; ++code)
        ++dos[ring_energy(decode_spins(code, N))];
    return dos;
}

DensityOfStates analytic_dos(int N) {
    DensityOfStates dos;
    for (int walls = 0; walls <= N; walls += 2)
        dos[-N + 2 * walls] = 2 * choose(N, walls);
    return dos;
}

std::pair<int, int> next_cooling_shell(std::vector<int> energies, int old_ceiling) {
    std::sort(energies.begin(), energies.end(), std::greater<int>());
    const auto next = std::find_if(energies.begin(), energies.end(),
                                   [old_ceiling](int energy) {
                                       return energy < old_ceiling;
                                   });
    if (next == energies.end()) return {old_ceiling, static_cast<int>(energies.size())};
    const int new_ceiling = *next;
    const int culled = static_cast<int>(std::count_if(
        energies.begin(), energies.end(), [new_ceiling](int energy) {
            return energy >= new_ceiling;
        }));
    return {new_ceiling, culled};
}

} // namespace

ISING_TEST_CASE("Even N=6 exhaustive DOS matches the analytic ring DOS") {
    ISING_REQUIRE(exhaustive_dos(6) == analytic_dos(6));
}

ISING_TEST_CASE("Odd N=5 exhaustive DOS matches the analytic ring DOS") {
    ISING_REQUIRE(exhaustive_dos(5) == analytic_dos(5));
}

ISING_TEST_CASE("Small-ring DOS totals 2^N and has two ground states") {
    for (int N = 3; N <= 12; ++N) {
        const DensityOfStates dos = analytic_dos(N);
        Count total = 0;
        for (const auto& level : dos) total += level.second;
        ISING_REQUIRE(total == (Count{1} << N));
        ISING_REQUIRE(dos.at(-N) == 2);
    }
}

ISING_TEST_CASE("Every periodic state has even walls and E=-N+2w") {
    for (int N = 3; N <= 9; ++N) {
        const std::uint64_t state_count = std::uint64_t{1} << N;
        for (std::uint64_t code = 0; code < state_count; ++code) {
            const std::vector<int> spins = decode_spins(code, N);
            const int walls = wall_count(spins);
            ISING_REQUIRE(walls % 2 == 0);
            ISING_REQUIRE(ring_energy(spins) == -N + 2 * walls);
        }
    }
}

ISING_TEST_CASE("Local flip delta matches a full energy recomputation") {
    for (int N = 3; N <= 8; ++N) {
        const std::uint64_t state_count = std::uint64_t{1} << N;
        for (std::uint64_t code = 0; code < state_count; ++code) {
            std::vector<int> spins = decode_spins(code, N);
            const int before = ring_energy(spins);
            for (int site = 0; site < N; ++site) {
                const int old_spin = spins[site];
                const int local_delta = 2 * old_spin
                    * (spins[left_neighbor(site, N)] + spins[right_neighbor(site, N)]);
                spins[site] = -old_spin;
                ISING_REQUIRE(ring_energy(spins) - before == local_delta);
                spins[site] = old_spin;
            }
        }
    }
}

ISING_TEST_CASE("Periodic ring neighbors wrap at every site") {
    for (int N = 3; N <= 12; ++N) {
        for (int site = 0; site < N; ++site) {
            ISING_REQUIRE(right_neighbor(site, N) == (site + 1) % N);
            ISING_REQUIRE(left_neighbor(site, N) == (site - 1 + N) % N);
            ISING_REQUIRE(right_neighbor(left_neighbor(site, N), N) == site);
            ISING_REQUIRE(left_neighbor(right_neighbor(site, N), N) == site);
        }
    }
}

ISING_TEST_CASE("Cooling selects the next strict occupied shell") {
    std::vector<int> exact_population;
    const DensityOfStates dos = analytic_dos(6);
    for (const auto& [energy, count] : dos)
        exact_population.insert(exact_population.end(), static_cast<std::size_t>(count), energy);

    const auto [first_shell, first_culled] = next_cooling_shell(exact_population, 7);
    ISING_REQUIRE(first_shell == 6);
    ISING_REQUIRE(first_culled == 2);

    const auto [next_shell, next_culled] = next_cooling_shell(exact_population, 6);
    ISING_REQUIRE(next_shell == 2);
    ISING_REQUIRE(next_culled == 32);
}

ISING_TEST_CASE("Resample reports the exact integer cooling shell") {
    int energies[] = {6, 2, -2, -6};
    int order[] = {0, 1, 2, 3};
    int update[] = {0, 1, 2, 3};
    int families[] = {0, 1, 2, 3};
    int ceiling = 7;
    const IsingResampleResult result = resample(
        energies, order, update, families, 4, &ceiling, false);
    ISING_REQUIRE(result.status == IsingResampleStatus::ok);
    ISING_REQUIRE(ceiling == 6);
    ISING_REQUIRE(result.old_U == 7);
    ISING_REQUIRE(result.new_U == 6);
    ISING_REQUIRE(result.n_cull == 1);
    ISING_REQUIRE(result.culling_fraction == 0.25);
}

ISING_TEST_CASE("Resample reports bounded unique terminal states") {
    for (bool no_next_shell : {true, false}) {
        int energies_no_next[] = {-5, -5, -5, -5};
        int energies_full_cull[] = {3, 3, 3, 3};
        int* energies = no_next_shell ? energies_no_next : energies_full_cull;
        int order[] = {0, 1, 2, 3};
        int update[] = {0, 1, 2, 3};
        int families[] = {0, 1, 2, 3};
        int ceiling = no_next_shell ? -5 : 4;
        const IsingResampleResult result = resample(
            energies, order, update, families, 4, &ceiling, false);
        ISING_REQUIRE(result.status == (no_next_shell
            ? IsingResampleStatus::no_next_shell
            : IsingResampleStatus::terminal_full_cull));
        ISING_REQUIRE(ceiling == (no_next_shell ? -5 : 3));
        ISING_REQUIRE(result.n_cull == 4);
        ISING_REQUIRE(result.culling_fraction == 1.0);
    }
}

ISING_TEST_CASE("PCG resampling state replays the exact parent map") {
    int initial_energies[] = {6, 6, 2, 2, -2, -6};
    int first_energies[6];
    int second_energies[6];
    std::copy(std::begin(initial_energies), std::end(initial_energies), first_energies);
    std::copy(std::begin(initial_energies), std::end(initial_energies), second_energies);
    int first_order[] = {0, 1, 2, 3, 4, 5};
    int second_order[] = {0, 1, 2, 3, 4, 5};
    int first_update[6] = {};
    int second_update[6] = {};
    int first_families[] = {0, 1, 2, 3, 4, 5};
    int second_families[] = {0, 1, 2, 3, 4, 5};
    int first_ceiling = 7;
    int second_ceiling = 7;
    initializeResamplingRng(90210);
    const ResamplingRngState start = getResamplingRngState();
    ISING_REQUIRE(resample(first_energies, first_order, first_update, first_families,
                           6, &first_ceiling, false).status
                  == IsingResampleStatus::ok);
    const ResamplingRngState first_end = getResamplingRngState();

    setResamplingRngState(start);
    ISING_REQUIRE(resample(second_energies, second_order, second_update, second_families,
                           6, &second_ceiling, false).status
                  == IsingResampleStatus::ok);
    const ResamplingRngState second_end = getResamplingRngState();

    ISING_REQUIRE(first_ceiling == second_ceiling);
    ISING_REQUIRE(std::equal(std::begin(first_order), std::end(first_order), second_order));
    ISING_REQUIRE(std::equal(std::begin(first_update), std::end(first_update), second_update));
    ISING_REQUIRE(std::equal(std::begin(first_families), std::end(first_families), second_families));
    ISING_REQUIRE(first_end.state == second_end.state);
    ISING_REQUIRE(first_end.stream == second_end.stream);
    ISING_REQUIRE(first_end.state != start.state);

}

ISING_TEST_CASE("PCG parent selection uses only surviving replicas") {
    int energies[] = {6, 6, 2, 2, -2, -6};
    int order[] = {0, 1, 2, 3, 4, 5};
    int update[6] = {};
    int families[] = {0, 1, 2, 3, 4, 5};
    int ceiling = 7;
    initializeResamplingRng(77);
    ISING_REQUIRE(resample(energies, order, update, families, 6, &ceiling,
                           false).status == IsingResampleStatus::ok);
    ISING_REQUIRE(ceiling == 6);
    for (int culled = 0; culled < 2; ++culled) {
        const int destination = order[culled];
        const int parent = update[destination];
        bool parent_survives = false;
        for (int survivor = 2; survivor < 6; ++survivor)
            parent_survives = parent_survives || parent == order[survivor];
        ISING_REQUIRE(parent_survives);
        ISING_REQUIRE(families[destination] == parent);
    }
}
