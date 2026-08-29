#include "test_harness.h"

#include "potts_lib.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <unistd.h>
#include <vector>

namespace {

using DensityOfStates = std::map<int, int>;

class ScopedStdoutSilence {
public:
    ScopedStdoutSilence() : saved_fd_(dup(fileno(stdout))), sink_(std::tmpfile()) {
        POTTS_REQUIRE(saved_fd_ >= 0);
        POTTS_REQUIRE(sink_ != nullptr);
        std::fflush(stdout);
        POTTS_REQUIRE(dup2(fileno(sink_), fileno(stdout)) >= 0);
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

PottsParams make_params(int L, int q, bool heat = false) {
    PottsParams params{};
    params.L = L;
    params.N = L * L;
    params.q = q;
    params.heat = heat;
    return params;
}

std::vector<char> decode_configuration(int code, int N, int q) {
    std::vector<char> spins(static_cast<std::size_t>(N));
    for (int site = 0; site < N; ++site) {
        spins[site] = static_cast<char>(code % q);
        code /= q;
    }
    return spins;
}

int integer_power(int base, int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; ++i) result *= base;
    return result;
}

int independent_energy(const std::vector<char>& spins, int L) {
    int energy = 0;
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = x + y * L;
            const int right = ((x + 1) % L) + y * L;
            const int down = x + ((y + 1) % L) * L;
            energy -= spins[site] == spins[right];
            energy -= spins[site] == spins[down];
        }
    }
    return energy;
}

int production_host_energy(const std::vector<char>& spins, int L) {
    const PottsParams params = make_params(L, 2);
    int doubled_energy = 0;
    for (int site = 0; site < params.N; ++site) {
        const neighborsIndexes indexes = SLF(site, L, params.N);
        const neighborsValues values{
            spins[indexes.up], spins[indexes.down],
            spins[indexes.left], spins[indexes.right]};
        doubled_energy += local_energy(spins[site], values);
    }
    return doubled_energy / 2;
}

DensityOfStates enumerate_dos(int L, int q, bool compare_production) {
    const int N = L * L;
    const int configuration_count = integer_power(q, N);
    DensityOfStates dos;
    for (int code = 0; code < configuration_count; ++code) {
        const std::vector<char> spins = decode_configuration(code, N, q);
        const int oracle = independent_energy(spins, L);
        if (compare_production) POTTS_REQUIRE(production_host_energy(spins, L) == oracle);
        ++dos[oracle];
    }
    return dos;
}

int independent_ising_energy(const std::vector<char>& potts, int L) {
    int energy = 0;
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = x + y * L;
            const int right = ((x + 1) % L) + y * L;
            const int down = x + ((y + 1) % L) * L;
            const int spin = potts[site] == 0 ? -1 : 1;
            const int spin_right = potts[right] == 0 ? -1 : 1;
            const int spin_down = potts[down] == 0 ? -1 : 1;
            energy -= spin * spin_right;
            energy -= spin * spin_down;
        }
    }
    return energy;
}

void check_shell(bool heat) {
    std::array<int, 4> energies = heat
        ? std::array<int, 4>{{-8, -8, -6, -4}}
        : std::array<int, 4>{{8, 8, 6, 4}};
    std::array<int, 4> order{{0, 1, 2, 3}};
    std::array<int, 4> update{};
    std::array<int, 4> family{{0, 1, 2, 3}};
    int U = heat ? -10 : 10;
    initialize_resampling_rng(7);
    const PottsResampleResult result = resample(
        energies.data(), order.data(), update.data(), family.data(), 4, &U, heat);
    POTTS_REQUIRE(result.status == POTTS_RESAMPLE_OK);
    POTTS_REQUIRE(result.old_U == (heat ? -10 : 10));
    POTTS_REQUIRE(result.new_U == (heat ? -8 : 8));
    POTTS_REQUIRE(result.n_cull == 2);
    POTTS_REQUIRE(result.culling_fraction == 0.5);
    POTTS_REQUIRE(U == (heat ? -8 : 8));
    POTTS_REQUIRE(update[0] >= 2);
    POTTS_REQUIRE(update[1] >= 2);
    POTTS_REQUIRE(update[2] == 2);
    POTTS_REQUIRE(update[3] == 3);

}

void check_terminal_shell(bool heat) {
    std::array<int, 4> energies{};
    energies.fill(heat ? -8 : 8);
    std::array<int, 4> order{{0, 1, 2, 3}};
    std::array<int, 4> update{{-1, -1, -1, -1}};
    std::array<int, 4> family{{0, 1, 2, 3}};
    int U = heat ? -10 : 10;
    const PottsResampleResult terminal = resample(
        energies.data(), order.data(), update.data(), family.data(), 4,
        &U, heat);
    POTTS_REQUIRE(terminal.status == POTTS_RESAMPLE_TERMINAL_FULL_CULL);
    POTTS_REQUIRE(terminal.n_cull == 4);
    POTTS_REQUIRE(terminal.culling_fraction == 1.0);
    POTTS_REQUIRE(U == (heat ? -8 : 8));
    const std::array<int, 4> identity{{0, 1, 2, 3}};
    POTTS_REQUIRE(update == identity);
    const PottsResampleResult no_next = resample(
        energies.data(), order.data(), update.data(), family.data(), 4,
        &U, heat);
    POTTS_REQUIRE(no_next.status == POTTS_RESAMPLE_NO_NEXT_SHELL);
    POTTS_REQUIRE(U == (heat ? -8 : 8));
}

} // namespace

POTTS_TEST_CASE("Exact L=2 DOS is correct for q=2,3,4") {
    const DensityOfStates q2{{-8, 2}, {-4, 12}, {0, 2}};
    const DensityOfStates q3{{-8, 3}, {-4, 36}, {-2, 24}, {0, 18}};
    const DensityOfStates q4{{-8, 4}, {-4, 72}, {-2, 96}, {0, 84}};
    POTTS_REQUIRE(enumerate_dos(2, 2, true) == q2);
    POTTS_REQUIRE(enumerate_dos(2, 3, true) == q3);
    POTTS_REQUIRE(enumerate_dos(2, 4, true) == q4);
}

POTTS_TEST_CASE("Exact L=3 DOS is correct for q=2,3,4") {
    const DensityOfStates q2{
        {-18, 2}, {-14, 18}, {-12, 48}, {-10, 198}, {-8, 144}, {-6, 102}};
    const DensityOfStates q3{
        {-18, 3}, {-14, 54}, {-12, 144}, {-11, 108}, {-10, 810},
        {-9, 660}, {-8, 2592}, {-7, 3240}, {-6, 3348}, {-5, 3996},
        {-4, 3240}, {-3, 936}, {-2, 540}, {0, 12}};
    const DensityOfStates q4{
        {-18, 4}, {-14, 108}, {-12, 288}, {-11, 432}, {-10, 2052},
        {-9, 2784}, {-8, 10368}, {-7, 19872}, {-6, 34884}, {-5, 48384},
        {-4, 63936}, {-3, 45360}, {-2, 26568}, {-1, 6048}, {0, 1056}};
    POTTS_REQUIRE(enumerate_dos(3, 2, false) == q2);
    POTTS_REQUIRE(enumerate_dos(3, 3, false) == q3);
    POTTS_REQUIRE(enumerate_dos(3, 4, false) == q4);
}

POTTS_TEST_CASE("Byte spins preserve the q=2,3,4 domains") {
    POTTS_REQUIRE(sizeof(char) == 1);
    for (int q : {2, 3, 4}) {
        POTTS_REQUIRE(potts_supported_q(q));
        for (int state = 0; state < q; ++state) {
            const char stored = static_cast<char>(state);
            POTTS_REQUIRE(static_cast<int>(stored) == state);
            POTTS_REQUIRE(stored >= 0);
            POTTS_REQUIRE(stored < q);
        }
    }
    POTTS_REQUIRE(!potts_supported_q(1));
    POTTS_REQUIRE(!potts_supported_q(5));
    POTTS_REQUIRE(potts_cooling_start_U() == 1);
    POTTS_REQUIRE(potts_heating_start_U(9) == -19);
}

POTTS_TEST_CASE("Periodic square-lattice neighbors are exact") {
    for (int L : {2, 3, 4}) {
        const int N = L * L;
        for (int site = 0; site < N; ++site) {
            const int x = site % L;
            const int y = site / L;
            const neighborsIndexes got = SLF(site, L, N);
            POTTS_REQUIRE(got.left == ((x - 1 + L) % L) + y * L);
            POTTS_REQUIRE(got.right == ((x + 1) % L) + y * L);
            POTTS_REQUIRE(got.up == x + ((y - 1 + L) % L) * L);
            POTTS_REQUIRE(got.down == x + ((y + 1) % L) * L);
        }
    }
}

POTTS_TEST_CASE("Full energies match an independent bond oracle") {
    for (int q : {2, 3, 4}) {
        const int count = integer_power(q, 9);
        for (int code = 0; code < count; ++code) {
            const std::vector<char> spins = decode_configuration(code, 9, q);
            POTTS_REQUIRE(production_host_energy(spins, 3)
                          == independent_energy(spins, 3));
        }
    }
}

POTTS_TEST_CASE("Local energy changes match an independent oracle") {
    for (int q : {2, 3, 4}) {
        const int neighbor_configurations = integer_power(q, 4);
        for (int code = 0; code < neighbor_configurations; ++code) {
            int remainder = code;
            neighborsValues neighbors{};
            char* values[] = {&neighbors.up, &neighbors.down,
                              &neighbors.left, &neighbors.right};
            for (char* value : values) {
                *value = static_cast<char>(remainder % q);
                remainder /= q;
            }
            for (int old_state = 0; old_state < q; ++old_state) {
                for (int new_state = 0; new_state < q; ++new_state) {
                    int matches_old = 0;
                    int matches_new = 0;
                    for (const char* value : values) {
                        matches_old += *value == old_state;
                        matches_new += *value == new_state;
                    }
                    const int expected = matches_old - matches_new;
                    POTTS_REQUIRE(local_energy(static_cast<char>(new_state), neighbors)
                                      - local_energy(static_cast<char>(old_state), neighbors)
                                  == expected);
                }
            }
        }
    }
}

POTTS_TEST_CASE("q=2 Potts energy obeys the affine Ising map") {
    for (int L : {2, 3}) {
        const int N = L * L;
        const int count = integer_power(2, N);
        for (int code = 0; code < count; ++code) {
            const std::vector<char> spins = decode_configuration(code, N, 2);
            const int potts = independent_energy(spins, L);
            const int ising = independent_ising_energy(spins, L);
            POTTS_REQUIRE(2 * potts == -2 * N + ising);
        }
    }
}

POTTS_TEST_CASE("Cooling heating and terminal shell semantics are strict") {
    ScopedStdoutSilence silence;
    check_shell(false);
    check_shell(true);
    check_terminal_shell(false);
    check_terminal_shell(true);
}

POTTS_TEST_CASE("Resampling RNG state replays parent choices exactly") {
    ScopedStdoutSilence silence;
    std::array<int, 5> energies{{0, -1, -2, -3, -4}};
    std::array<int, 5> order{{0, 1, 2, 3, 4}};
    std::array<int, 5> update{};
    std::array<int, 5> families{{0, 1, 2, 3, 4}};
    initialize_resampling_rng(31415);
    const PottsResamplingRngState initial = get_resampling_rng_state();
    int ceiling = 1;
    POTTS_REQUIRE(resample(energies.data(), order.data(), update.data(),
                           families.data(), 5, &ceiling, false).status
                  == POTTS_RESAMPLE_OK);
    POTTS_REQUIRE(ceiling == 0);
    const std::array<int, 5> expected_update = update;
    const std::array<int, 5> expected_families = families;
    const PottsResamplingRngState expected_rng = get_resampling_rng_state();

    order = {{0, 1, 2, 3, 4}};
    update = {{0, 0, 0, 0, 0}};
    families = {{0, 1, 2, 3, 4}};
    set_resampling_rng_state(initial);
    ceiling = 1;
    POTTS_REQUIRE(resample(energies.data(), order.data(), update.data(),
                           families.data(), 5, &ceiling, false).status
                  == POTTS_RESAMPLE_OK);
    POTTS_REQUIRE(update == expected_update);
    POTTS_REQUIRE(families == expected_families);
    const PottsResamplingRngState replay_rng = get_resampling_rng_state();
    POTTS_REQUIRE(replay_rng.state == expected_rng.state);
    POTTS_REQUIRE(replay_rng.stream == expected_rng.stream);

    const int destination = order[0];
    POTTS_REQUIRE(energies[update[destination]] < ceiling);
    POTTS_REQUIRE(families[destination] == update[destination]);
}
