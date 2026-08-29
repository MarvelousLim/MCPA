#include <doctest/doctest.h>

#include "checkpoint.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct TemporaryCheckpointDirectory {
    std::string path;

    TemporaryCheckpointDirectory() {
        std::array<char, 64> pattern{};
        std::snprintf(pattern.data(), pattern.size(),
                      "/tmp/mcpa-checkpoint-test.XXXXXX");
        char* created = mkdtemp(pattern.data());
        REQUIRE(created != nullptr);
        path = created;
    }

    ~TemporaryCheckpointDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct SavedState {
    static constexpr int L = 3;
    static constexpr int N = L * L;
    static constexpr int R = 8;
    static constexpr int n_steps = 1;
    static constexpr int seed = 91;

    std::vector<int> spins = std::vector<int>(R * N);
    std::vector<int> energies = std::vector<int>(R);
    std::vector<int> families = std::vector<int>(R);
    std::vector<int> order = std::vector<int>(R);
    std::vector<unsigned char> philox = std::vector<unsigned char>(R * 64);
    int U = 0;
    int64_t step = 0;
    uint64_t resampling_state = 0;
    uint64_t resampling_stream = 1;
    std::array<int64_t, 3> output_positions{};
};

SavedState make_state(int generation) {
    SavedState state;
    for (size_t i = 0; i < state.spins.size(); ++i)
        state.spins[i] = ((static_cast<int>(i) + generation) % 3 == 0) ? -1 : 1;
    for (int r = 0; r < SavedState::R; ++r) {
        state.energies[r] = -18 + 4 * ((r + generation) % 7);
        state.families[r] = (r + generation) % SavedState::R;
        state.order[r] = SavedState::R - 1 - r;
    }
    for (size_t i = 0; i < state.philox.size(); ++i)
        state.philox[i] = static_cast<unsigned char>((i * 17 + generation) & 0xff);
    state.U = 22 - 4 * generation;
    state.step = generation;
    state.resampling_state = 0x123456789abcdef0ULL + generation;
    state.resampling_stream = 0xfedcba9876543211ULL + 2 * generation;
    state.output_positions = {{100 + generation, 200 + generation, 300 + generation}};
    return state;
}

CheckpointManager make_manager(const TemporaryCheckpointDirectory& directory) {
    CheckpointManager manager{};
    checkpoint_init(manager, SavedState::L, SavedState::N, SavedState::R,
                    SavedState::n_steps, SavedState::seed, 0.0f, 0,
                    directory.path.c_str(), true, 3600, "bw_checkpoint_test");
    return manager;
}

void save_state(CheckpointManager& manager, const SavedState& state) {
    manager.step_count = state.step;
    REQUIRE(checkpoint_save(
        manager, SavedState::L, SavedState::N, SavedState::R,
        SavedState::n_steps, SavedState::seed, 0.0f, 0,
        state.spins.data(), state.energies.data(), state.families.data(),
        state.order.data(), state.U, state.philox.data(), state.philox.size(),
        state.resampling_state, state.resampling_stream,
        state.output_positions.data()));
}

SavedState load_state(const CheckpointManager& manager) {
    SavedState loaded;
    loaded.U = 999;
    loaded.step = -1;
    loaded.output_positions.fill(-1);
    REQUIRE(checkpoint_load(
        manager, SavedState::L, SavedState::N, SavedState::R,
        SavedState::n_steps, SavedState::seed, 0.0f, 0,
        loaded.spins.data(), loaded.energies.data(), loaded.families.data(),
        loaded.order.data(), loaded.U, loaded.step,
        loaded.philox.data(), loaded.philox.size(),
        loaded.resampling_state, loaded.resampling_stream,
        loaded.output_positions.data()));
    return loaded;
}

void check_equal(const SavedState& actual, const SavedState& expected) {
    CHECK(actual.spins == expected.spins);
    CHECK(actual.energies == expected.energies);
    CHECK(actual.families == expected.families);
    CHECK(actual.order == expected.order);
    CHECK(actual.philox == expected.philox);
    CHECK(actual.U == expected.U);
    CHECK(actual.step == expected.step);
    CHECK(actual.resampling_state == expected.resampling_state);
    CHECK(actual.resampling_stream == expected.resampling_stream);
    CHECK(actual.output_positions == expected.output_positions);
}

std::string checkpoint_path(const CheckpointManager& manager, const char* suffix) {
    std::array<char, 1024> path{};
    chk_path(manager, suffix, path.data(), path.size());
    return path.data();
}

void overwrite_with_partial_data(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "wb");
    REQUIRE(file != nullptr);
    const std::array<unsigned char, 7> partial{{'M', 'C', 'P', 'A', 0, 1, 2}};
    REQUIRE(std::fwrite(partial.data(), 1, partial.size(), file) == partial.size());
    REQUIRE(std::fclose(file) == 0);
}

void flip_payload_bit(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "r+b");
    REQUIRE(file != nullptr);
    REQUIRE(std::fseek(file, static_cast<long>(sizeof(CheckpointHeader)) + 3,
                       SEEK_SET) == 0);
    unsigned char value = 0;
    REQUIRE(std::fread(&value, 1, 1, file) == 1);
    value ^= 0x20U;
    REQUIRE(std::fseek(file, -1, SEEK_CUR) == 0);
    REQUIRE(std::fwrite(&value, 1, 1, file) == 1);
    REQUIRE(std::fclose(file) == 0);
}

} // namespace

TEST_CASE("Checkpoint round-trip preserves MCPA state and both RNG streams") {
    TemporaryCheckpointDirectory directory;
    CheckpointManager manager = make_manager(directory);
    const SavedState expected = make_state(3);
    save_state(manager, expected);
    check_equal(load_state(manager), expected);
}

TEST_CASE("Corrupted current checkpoint falls back to the previous generation") {
    TemporaryCheckpointDirectory directory;
    CheckpointManager manager = make_manager(directory);
    const SavedState previous = make_state(4);
    save_state(manager, previous);
    save_state(manager, make_state(5));
    overwrite_with_partial_data(checkpoint_path(manager, ".bin"));
    check_equal(load_state(manager), previous);
}

TEST_CASE("Checksum-detected bit corruption falls back to the previous generation") {
    TemporaryCheckpointDirectory directory;
    CheckpointManager manager = make_manager(directory);
    const SavedState previous = make_state(5);
    save_state(manager, previous);
    save_state(manager, make_state(6));
    flip_payload_bit(checkpoint_path(manager, ".bin"));
    check_equal(load_state(manager), previous);
}

TEST_CASE("Interrupted temporary checkpoint leaves the current generation usable") {
    TemporaryCheckpointDirectory directory;
    CheckpointManager manager = make_manager(directory);
    const SavedState current = make_state(6);
    save_state(manager, current);
    overwrite_with_partial_data(checkpoint_path(manager, ".tmp"));
    check_equal(load_state(manager), current);
}

TEST_CASE("Missing current checkpoint falls back across the rename boundary") {
    TemporaryCheckpointDirectory directory;
    CheckpointManager manager = make_manager(directory);
    const SavedState previous = make_state(7);
    save_state(manager, previous);
    save_state(manager, make_state(8));
    REQUIRE(std::remove(checkpoint_path(manager, ".bin").c_str()) == 0);
    check_equal(load_state(manager), previous);
}

TEST_CASE("Complete first temporary checkpoint is recoverable before its rename") {
    TemporaryCheckpointDirectory directory;
    CheckpointManager manager = make_manager(directory);
    const SavedState expected = make_state(9);
    save_state(manager, expected);
    REQUIRE(std::rename(checkpoint_path(manager, ".bin").c_str(),
                        checkpoint_path(manager, ".tmp").c_str()) == 0);
    CHECK(checkpoint_exists(manager));
    check_equal(load_state(manager), expected);
}
