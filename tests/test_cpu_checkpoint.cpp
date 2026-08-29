#include "test_harness.h"

#include "checkpoint_1d.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

struct TemporaryDirectory {
    std::string path;
    TemporaryDirectory() {
        std::array<char, 64> pattern{};
        std::snprintf(pattern.data(), pattern.size(), "/tmp/mcpa-1d-checkpoint.XXXXXX");
        char* created = ::mkdtemp(pattern.data());
        ISING_REQUIRE(created != nullptr);
        path = created;
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

constexpr int kN = 5;
constexpr int kR = 7;
constexpr int kSteps = 3;
constexpr int kSeed = 123;

Ising1DCheckpointIdentity identity() {
    return {kN, kR, kSteps, kSeed, 100, 7U * 64U};
}

Ising1DCheckpointState state_for(int generation) {
    Ising1DCheckpointState state;
    state.U = kN + 2 - 4 * generation;
    state.completed_shells = static_cast<std::uint64_t>(generation);
    for (std::size_t i = 0; i < state.output_offsets.size(); ++i)
        state.output_offsets[i] = 1000 * generation + static_cast<int>(i);
    state.resampling_rng = {0x123456789abcdef0ULL + generation,
                            0xfedcba9876543211ULL + 2U * generation};
    state.spins.resize(kN * kR);
    for (std::size_t i = 0; i < state.spins.size(); ++i)
        state.spins[i] = ((i + generation) % 3U == 0) ? -1 : 1;
    state.energies.resize(kR);
    state.families.resize(kR);
    state.order.resize(kR);
    for (int r = 0; r < kR; ++r) {
        int exact_energy = 0;
        const std::size_t shift = static_cast<std::size_t>(r) * kN;
        for (int site = 0; site < kN; ++site)
            exact_energy -= state.spins[shift + static_cast<std::size_t>(site)]
                            * state.spins[shift
                                + static_cast<std::size_t>((site + 1) % kN)];
        state.energies[r] = exact_energy;
        state.families[r] = (r + generation) % kR;
        state.order[r] = kR - 1 - r;
    }
    state.philox.resize(identity().philox_bytes);
    for (std::size_t i = 0; i < state.philox.size(); ++i)
        state.philox[i] = static_cast<std::uint8_t>((17U * i + generation) & 0xffU);
    return state;
}

void require_equal(const Ising1DCheckpointState& actual,
                   const Ising1DCheckpointState& expected) {
    ISING_REQUIRE(actual.U == expected.U);
    ISING_REQUIRE(actual.completed_shells == expected.completed_shells);
    ISING_REQUIRE(actual.output_offsets == expected.output_offsets);
    ISING_REQUIRE(actual.resampling_rng.state == expected.resampling_rng.state);
    ISING_REQUIRE(actual.resampling_rng.stream == expected.resampling_rng.stream);
    ISING_REQUIRE(actual.spins == expected.spins);
    ISING_REQUIRE(actual.energies == expected.energies);
    ISING_REQUIRE(actual.families == expected.families);
    ISING_REQUIRE(actual.order == expected.order);
    ISING_REQUIRE(actual.philox == expected.philox);
}

Ising1DCheckpointState load(const std::string& base) {
    Ising1DCheckpointState loaded;
    std::string error;
    ISING_REQUIRE(load_ising1d_checkpoint(base, identity(), &loaded, nullptr, &error)
                  == Ising1DCheckpointLoadStatus::loaded);
    return loaded;
}

void corrupt(const std::string& path) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    ISING_REQUIRE(file.good());
    file.seekg(20);
    char value = 0;
    file.read(&value, 1);
    ISING_REQUIRE(file.good());
    value ^= 0x40;
    file.seekp(20);
    file.write(&value, 1);
    ISING_REQUIRE(file.good());
}

} // namespace

ISING_TEST_CASE("Checkpoint round trip preserves packed Ising and RNG state") {
    TemporaryDirectory directory;
    const std::string base = directory.path + "/run";
    const Ising1DCheckpointState expected = state_for(3);
    std::string error;
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity(), expected, &error));
    require_equal(load(base), expected);
}

ISING_TEST_CASE("Checkpoint rotation retains the previous complete generation") {
    TemporaryDirectory directory;
    const std::string base = directory.path + "/run";
    const Ising1DCheckpointState previous = state_for(4);
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity(), previous));
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity(), state_for(5)));
    ISING_REQUIRE(std::filesystem::exists(ising1d_checkpoint_path(base, ".prev.bin")));
    corrupt(ising1d_checkpoint_path(base, ".bin"));
    require_equal(load(base), previous);
}

ISING_TEST_CASE("Checkpoint CRC rejects corruption without a valid fallback") {
    TemporaryDirectory directory;
    const std::string base = directory.path + "/run";
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity(), state_for(6)));
    corrupt(ising1d_checkpoint_path(base, ".bin"));
    Ising1DCheckpointState loaded;
    ISING_REQUIRE(load_ising1d_checkpoint(base, identity(), &loaded)
                  == Ising1DCheckpointLoadStatus::invalid);
}

ISING_TEST_CASE("Complete checkpoint temporary file is recoverable") {
    TemporaryDirectory directory;
    const std::string base = directory.path + "/run";
    const Ising1DCheckpointState expected = state_for(7);
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity(), expected));
    ISING_REQUIRE(std::rename(ising1d_checkpoint_path(base, ".bin").c_str(),
                              ising1d_checkpoint_path(base, ".tmp").c_str()) == 0);
    require_equal(load(base), expected);
}

ISING_TEST_CASE("Checkpoint done marker suppresses completed trajectory resume") {
    TemporaryDirectory directory;
    const std::string base = directory.path + "/run";
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity(), state_for(8)));
    ISING_REQUIRE(mark_ising1d_checkpoint_done(base));
    Ising1DCheckpointState loaded;
    ISING_REQUIRE(load_ising1d_checkpoint(base, identity(), &loaded)
                  == Ising1DCheckpointLoadStatus::done);
}

ISING_TEST_CASE("Checkpoint rejects inconsistent replica semantics") {
    TemporaryDirectory directory;
    const std::string base = directory.path + "/run";
    Ising1DCheckpointState invalid_energy = state_for(2);
    ++invalid_energy.energies[0];
    std::string error;
    ISING_REQUIRE(!save_ising1d_checkpoint(base, identity(), invalid_energy, &error));

    Ising1DCheckpointState invalid_family = state_for(2);
    invalid_family.families[0] = kR;
    ISING_REQUIRE(!save_ising1d_checkpoint(base, identity(), invalid_family, &error));

    Ising1DCheckpointState invalid_order = state_for(2);
    invalid_order.order[0] = invalid_order.order[1];
    ISING_REQUIRE(!save_ising1d_checkpoint(base, identity(), invalid_order, &error));

    Ising1DCheckpointState invalid_offset = state_for(2);
    invalid_offset.output_offsets[0] = -1;
    ISING_REQUIRE(!save_ising1d_checkpoint(base, identity(), invalid_offset, &error));
}

ISING_TEST_CASE("Checkpoint identity includes the detailed output cap") {
    TemporaryDirectory directory;
    const std::string base = directory.path + "/run";
    ISING_REQUIRE(save_ising1d_checkpoint(base, identity(), state_for(3)));
    Ising1DCheckpointIdentity mismatched = identity();
    mismatched.detailed_cap = 0;
    Ising1DCheckpointState loaded;
    ISING_REQUIRE(load_ising1d_checkpoint(base, mismatched, &loaded)
                  == Ising1DCheckpointLoadStatus::invalid);
}
