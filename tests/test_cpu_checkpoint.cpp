#include "test_harness.h"

#include "checkpoint.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

class ScopedSilence {
public:
    ScopedSilence()
        : saved_out_(dup(fileno(stdout))), saved_err_(dup(fileno(stderr))),
          sink_(std::tmpfile()) {
        BC_REQUIRE(saved_out_ >= 0);
        BC_REQUIRE(saved_err_ >= 0);
        BC_REQUIRE(sink_ != nullptr);
        std::fflush(nullptr);
        BC_REQUIRE(dup2(fileno(sink_), fileno(stdout)) >= 0);
        BC_REQUIRE(dup2(fileno(sink_), fileno(stderr)) >= 0);
    }
    ~ScopedSilence() {
        std::fflush(nullptr);
        dup2(saved_out_, fileno(stdout));
        dup2(saved_err_, fileno(stderr));
        close(saved_out_);
        close(saved_err_);
        std::fclose(sink_);
    }
private:
    int saved_out_;
    int saved_err_;
    FILE* sink_;
};

class TempDirectory {
public:
    TempDirectory() {
        std::array<char, 64> pattern{};
        std::snprintf(pattern.data(), pattern.size(), "/tmp/mcpa-bc-chk-XXXXXX");
        char* result = mkdtemp(pattern.data());
        BC_REQUIRE(result != nullptr);
        path_ = result;
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

struct Fixture {
    static constexpr int L = 3;
    static constexpr int N = 9;
    static constexpr int R = 4;
    static constexpr int n_steps = 2;
    static constexpr int seed = 73;
    static constexpr int d_num = 49;
    static constexpr int d_den = 25;
    static constexpr int heat = 0;
    static constexpr int detail_cap = 100;

    std::vector<int> spins = std::vector<int>(R * N);
    std::array<int, R> e_j{};
    std::array<int, R> e_delta{};
    std::array<int, R> families{};
    std::array<int, R> order{};
    std::array<unsigned char, 37> philox{};
    std::array<int64_t, 3> positions{{101, 202, 303}};

    explicit Fixture(int generation) {
        for (int i = 0; i < R * N; ++i) spins[i] = (i + generation) % 3 - 1;
        for (int r = 0; r < R; ++r) {
            int exact_j = 0;
            int exact_delta = 0;
            const int* replica = spins.data() + r * N;
            for (int y = 0; y < L; ++y) {
                for (int x = 0; x < L; ++x) {
                    const int site = x + y * L;
                    const int right = (x + 1) % L + y * L;
                    const int down = x + ((y + 1) % L) * L;
                    exact_j -= replica[site] * (replica[right] + replica[down]);
                    exact_delta += replica[site] * replica[site];
                }
            }
            e_j[r] = exact_j;
            e_delta[r] = exact_delta;
            families[r] = (r + generation) % R;
            order[r] = R - 1 - r;
        }
        for (size_t i = 0; i < philox.size(); ++i)
            philox[i] = static_cast<unsigned char>(3 * i + generation);
    }
};

CheckpointManager manager_for(const TempDirectory& directory) {
    CheckpointManager manager{};
    const std::string nested = directory.path() + "/nested/checkpoints";
    BC_REQUIRE(checkpoint_init_bc(
        manager, Fixture::L, Fixture::N, Fixture::R, Fixture::n_steps,
        Fixture::seed, Fixture::d_num, Fixture::d_den, Fixture::heat,
        Fixture::detail_cap, nested.c_str(), true, 3600));
    return manager;
}

void save_fixture(CheckpointManager& manager, const Fixture& fixture,
                  int U, int64_t step) {
    manager.step_count = step;
    BC_REQUIRE(checkpoint_save_bc(
        manager, Fixture::L, Fixture::N, Fixture::R, Fixture::n_steps,
        Fixture::seed, Fixture::d_num, Fixture::d_den, Fixture::heat,
        Fixture::detail_cap, fixture.spins.data(), fixture.e_j.data(),
        fixture.e_delta.data(),
        fixture.families.data(), fixture.order.data(), U,
        fixture.philox.data(), fixture.philox.size(),
        0x12340000ULL + static_cast<uint64_t>(step),
        0x56780001ULL + 2ULL * static_cast<uint64_t>(step),
        fixture.positions.data()));
}

bool load_fixture(const CheckpointManager& manager, int expected_d_num,
                  Fixture& loaded, int& U, int64_t& step,
                  uint64_t& pcg_state, uint64_t& pcg_stream,
                  std::array<int64_t, 3>& positions,
                  int expected_detail_cap = Fixture::detail_cap) {
    return checkpoint_load_bc(
        manager, Fixture::L, Fixture::N, Fixture::R, Fixture::n_steps,
        Fixture::seed, expected_d_num, Fixture::d_den, Fixture::heat,
        expected_detail_cap, loaded.spins.data(), loaded.e_j.data(),
        loaded.e_delta.data(),
        loaded.families.data(), loaded.order.data(), U, step,
        loaded.philox.data(), loaded.philox.size(), pcg_state, pcg_stream,
        positions.data());
}

void require_fixture_equal(const Fixture& actual, const Fixture& expected) {
    BC_REQUIRE(actual.spins == expected.spins);
    BC_REQUIRE(actual.e_j == expected.e_j);
    BC_REQUIRE(actual.e_delta == expected.e_delta);
    BC_REQUIRE(actual.families == expected.families);
    BC_REQUIRE(actual.order == expected.order);
    BC_REQUIRE(actual.philox == expected.philox);
}

std::string path_for(const CheckpointManager& manager, const char* suffix) {
    std::array<char, BC_CHECKPOINT_PATH_CAPACITY> path{};
    BC_REQUIRE(checkpoint_path(manager, suffix, path.data(), path.size()));
    return path.data();
}

uint32_t test_crc32(uint32_t crc, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return crc;
}

void make_current_family_semantically_invalid(const CheckpointManager& manager) {
    const std::string path = path_for(manager, ".bin");
    FILE* file = std::fopen(path.c_str(), "r+b");
    BC_REQUIRE(file != nullptr);
    BcCheckpointHeader header{};
    BC_REQUIRE(std::fread(&header, sizeof(header), 1, file) == 1);
    std::vector<unsigned char> payload(static_cast<size_t>(header.payload_bytes));
    BC_REQUIRE(std::fread(payload.data(), 1, payload.size(), file) == payload.size());
    const size_t spin_words = (static_cast<size_t>(Fixture::R) * Fixture::N + 15U) / 16U;
    const size_t family_offset = spin_words * sizeof(uint32_t)
                               + 2U * Fixture::R * sizeof(int);
    const int invalid_family = Fixture::R;
    std::memcpy(payload.data() + family_offset, &invalid_family, sizeof(invalid_family));
    header.checksum = 0;
    uint32_t crc = test_crc32(0xffffffffU, &header, sizeof(header));
    crc = test_crc32(crc, payload.data(), payload.size()) ^ 0xffffffffU;
    header.checksum = crc;
    BC_REQUIRE(std::fseek(file, 0, SEEK_SET) == 0);
    BC_REQUIRE(std::fwrite(&header, sizeof(header), 1, file) == 1);
    BC_REQUIRE(std::fwrite(payload.data(), 1, payload.size(), file) == payload.size());
    BC_REQUIRE(std::fclose(file) == 0);
}

} // namespace

BC_TEST_CASE("BC checkpoint roundtrip preserves complete trajectory state") {
    ScopedSilence silence;
    TempDirectory directory;
    CheckpointManager manager = manager_for(directory);
    const Fixture expected(1);
    save_fixture(manager, expected, -321, 7);
    FILE* checkpoint = std::fopen(path_for(manager, ".bin").c_str(), "rb");
    BC_REQUIRE(checkpoint != nullptr);
    BcCheckpointHeader header{};
    BC_REQUIRE(std::fread(&header, sizeof(header), 1, checkpoint) == 1);
    BC_REQUIRE(std::fclose(checkpoint) == 0);
    BC_REQUIRE(std::memcmp(header.magic, BC_CHECKPOINT_MAGIC, 8) == 0);
    BC_REQUIRE(header.version == 8);
    BC_REQUIRE(header.detail_cap == Fixture::detail_cap);
    Fixture loaded(0);
    int U = 0;
    int64_t step = 0;
    uint64_t pcg_state = 0, pcg_stream = 0;
    std::array<int64_t, 3> positions{};
    BC_REQUIRE(load_fixture(manager, Fixture::d_num, loaded, U, step,
                            pcg_state, pcg_stream, positions));
    require_fixture_equal(loaded, expected);
    BC_REQUIRE(U == -321);
    BC_REQUIRE(step == 7);
    BC_REQUIRE(pcg_state == 0x12340007ULL);
    BC_REQUIRE(pcg_stream == 0x5678000fULL);
    BC_REQUIRE(positions == expected.positions);
}

BC_TEST_CASE("BC checkpoint rejects parameter mismatch") {
    ScopedSilence silence;
    TempDirectory directory;
    CheckpointManager manager = manager_for(directory);
    save_fixture(manager, Fixture(1), -10, 1);
    Fixture loaded(0);
    int U = 0; int64_t step = 0; uint64_t state = 0, stream = 0;
    std::array<int64_t, 3> positions{};
    BC_REQUIRE(!load_fixture(manager, Fixture::d_num + 1, loaded, U, step,
                             state, stream, positions));
    BC_REQUIRE(!load_fixture(manager, Fixture::d_num, loaded, U, step,
                             state, stream, positions,
                             Fixture::detail_cap + 1));
    CheckpointManager other_cap{};
    const std::string nested = directory.path() + "/nested/checkpoints";
    BC_REQUIRE(checkpoint_init_bc(
        other_cap, Fixture::L, Fixture::N, Fixture::R, Fixture::n_steps,
        Fixture::seed, Fixture::d_num, Fixture::d_den, Fixture::heat,
        Fixture::detail_cap + 1, nested.c_str(), true, 3600));
    BC_REQUIRE(path_for(manager, ".bin") == path_for(other_cap, ".bin"));
}

BC_TEST_CASE("BC checkpoint rejects CRC-valid impossible state without partial load") {
    ScopedSilence silence;
    TempDirectory directory;
    CheckpointManager manager = manager_for(directory);
    save_fixture(manager, Fixture(1), -10, 1);
    make_current_family_semantically_invalid(manager);
    Fixture loaded(3);
    const Fixture unchanged = loaded;
    int U = 808; int64_t step = 909; uint64_t state = 707, stream = 505;
    std::array<int64_t, 3> positions{{4, 5, 6}};
    BC_REQUIRE(!load_fixture(manager, Fixture::d_num, loaded, U, step,
                             state, stream, positions));
    require_fixture_equal(loaded, unchanged);
    BC_REQUIRE(U == 808);
    BC_REQUIRE(step == 909);
    BC_REQUIRE(state == 707);
    BC_REQUIRE(stream == 505);
    BC_REQUIRE((positions == std::array<int64_t, 3>{{4, 5, 6}}));
}

BC_TEST_CASE("BC checkpoint corruption falls back to previous generation") {
    ScopedSilence silence;
    TempDirectory directory;
    CheckpointManager manager = manager_for(directory);
    const Fixture previous(1), current(2);
    save_fixture(manager, previous, -100, 1);
    save_fixture(manager, current, -200, 2);
    FILE* file = std::fopen(path_for(manager, ".bin").c_str(), "r+b");
    BC_REQUIRE(file != nullptr);
    BC_REQUIRE(std::fseek(file, static_cast<long>(sizeof(BcCheckpointHeader)) + 3, SEEK_SET) == 0);
    int byte = std::fgetc(file);
    BC_REQUIRE(byte != EOF);
    BC_REQUIRE(std::fseek(file, -1, SEEK_CUR) == 0);
    BC_REQUIRE(std::fputc(byte ^ 0x5a, file) != EOF);
    std::fclose(file);
    Fixture loaded(0);
    int U = 0; int64_t step = 0; uint64_t state = 0, stream = 0;
    std::array<int64_t, 3> positions{};
    BC_REQUIRE(load_fixture(manager, Fixture::d_num, loaded, U, step,
                            state, stream, positions));
    require_fixture_equal(loaded, previous);
    BC_REQUIRE(U == -100);
    BC_REQUIRE(step == 1);
}

BC_TEST_CASE("BC checkpoint retains previous generation") {
    ScopedSilence silence;
    TempDirectory directory;
    CheckpointManager manager = manager_for(directory);
    save_fixture(manager, Fixture(1), -1, 1);
    save_fixture(manager, Fixture(2), -2, 2);
    BC_REQUIRE(std::filesystem::is_regular_file(path_for(manager, ".bin")));
    BC_REQUIRE(std::filesystem::is_regular_file(path_for(manager, ".prev.bin")));
}

BC_TEST_CASE("BC checkpoint recovers a valid temporary generation") {
    ScopedSilence silence;
    TempDirectory directory;
    CheckpointManager manager = manager_for(directory);
    const Fixture expected(3);
    save_fixture(manager, expected, -30, 3);
    std::filesystem::rename(path_for(manager, ".bin"), path_for(manager, ".tmp"));
    Fixture loaded(0);
    int U = 0; int64_t step = 0; uint64_t state = 0, stream = 0;
    std::array<int64_t, 3> positions{};
    BC_REQUIRE(load_fixture(manager, Fixture::d_num, loaded, U, step,
                            state, stream, positions));
    require_fixture_equal(loaded, expected);
}

BC_TEST_CASE("BC checkpoint done marker suppresses resume") {
    ScopedSilence silence;
    TempDirectory directory;
    CheckpointManager manager = manager_for(directory);
    save_fixture(manager, Fixture(1), -10, 1);
    BC_REQUIRE(checkpoint_mark_done(manager));
    BC_REQUIRE(checkpoint_is_done(manager));
    BC_REQUIRE(!checkpoint_exists(manager));
    Fixture loaded(0);
    int U = 0; int64_t step = 0; uint64_t state = 0, stream = 0;
    std::array<int64_t, 3> positions{};
    BC_REQUIRE(!load_fixture(manager, Fixture::d_num, loaded, U, step,
                             state, stream, positions));
    CheckpointManager other_cap{};
    const std::string nested = directory.path() + "/nested/checkpoints";
    BC_REQUIRE(checkpoint_init_bc(
        other_cap, Fixture::L, Fixture::N, Fixture::R, Fixture::n_steps,
        Fixture::seed, Fixture::d_num, Fixture::d_den, Fixture::heat,
        Fixture::detail_cap + 1, nested.c_str(), true, 3600));
    BC_REQUIRE(!checkpoint_is_done(other_cap));
    BC_REQUIRE(checkpoint_exists(other_cap));
}
