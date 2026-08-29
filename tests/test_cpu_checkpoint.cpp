#include "test_harness.h"

#include "potts_checkpoint.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

struct TempDirectory {
    std::string path;
    TempDirectory() {
        std::array<char, 64> pattern{};
        std::snprintf(pattern.data(), pattern.size(), "/tmp/mcpa-potts-checkpoint.XXXXXX");
        char* created = ::mkdtemp(pattern.data());
        POTTS_REQUIRE(created != nullptr);
        path = created;
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

constexpr int kL = 3;
constexpr int kN = 9;
constexpr int kR = 7;
constexpr int kQ = 3;

PottsCheckpointIdentity identity() {
    return {kL, kN, kR, 2, 911, kQ, false, 100, 7U * 64U};
}

int energy(const char* spins) {
    int result = 0;
    for (int y = 0; y < kL; ++y) {
        for (int x = 0; x < kL; ++x) {
            const int site = x + y * kL;
            result -= spins[site] == spins[(x + 1) % kL + y * kL];
            result -= spins[site] == spins[x + ((y + 1) % kL) * kL];
        }
    }
    return result;
}

PottsCheckpointState make_state(int generation) {
    PottsCheckpointState state;
    state.U = -1 - generation % (2 * kN);
    state.completed_shells = generation;
    for (std::size_t i = 0; i < state.output_offsets.size(); ++i)
        state.output_offsets[i] = 100 * generation + static_cast<int>(i);
    state.resampling_rng = {0x123456789abcdef0ULL + generation,
                            0xfedcba9876543211ULL + 2U * generation};
    state.spins.resize(kN * kR);
    state.energies.resize(kR);
    state.families.resize(kR);
    state.order.resize(kR);
    for (int r = 0; r < kR; ++r) {
        char* replica = state.spins.data() + r * kN;
        for (int site = 0; site < kN; ++site)
            replica[site] = static_cast<char>((site + 2 * r + generation) % kQ);
        state.energies[r] = energy(replica);
        state.families[r] = (r + generation) % kR;
        state.order[r] = kR - 1 - r;
    }
    state.philox.resize(identity().philox_bytes);
    for (std::size_t i = 0; i < state.philox.size(); ++i)
        state.philox[i] = static_cast<std::uint8_t>((13U * i + generation) & 0xffU);
    return state;
}

void require_equal(const PottsCheckpointState& actual,
                   const PottsCheckpointState& expected) {
    POTTS_REQUIRE(actual.U == expected.U);
    POTTS_REQUIRE(actual.completed_shells == expected.completed_shells);
    POTTS_REQUIRE(actual.output_offsets == expected.output_offsets);
    POTTS_REQUIRE(actual.resampling_rng.state == expected.resampling_rng.state);
    POTTS_REQUIRE(actual.resampling_rng.stream == expected.resampling_rng.stream);
    POTTS_REQUIRE(actual.spins == expected.spins);
    POTTS_REQUIRE(actual.energies == expected.energies);
    POTTS_REQUIRE(actual.families == expected.families);
    POTTS_REQUIRE(actual.order == expected.order);
    POTTS_REQUIRE(actual.philox == expected.philox);
}

PottsCheckpointState load(const std::string& base) {
    PottsCheckpointState state;
    POTTS_REQUIRE(load_potts_checkpoint(base, identity(), &state)
                  == PottsCheckpointLoadStatus::loaded);
    return state;
}

std::uint32_t crc32(const std::vector<std::uint8_t>& bytes, std::size_t length) {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return crc ^ 0xffffffffU;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    POTTS_REQUIRE(file.good());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(file), {});
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    POTTS_REQUIRE(file.good());
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    POTTS_REQUIRE(file.good());
}

void corrupt_payload(const std::string& path) {
    std::vector<std::uint8_t> bytes = read_bytes(path);
    POTTS_REQUIRE(bytes.size() > 121);
    bytes[120] ^= 0x40U;
    write_bytes(path, bytes);
}

} // namespace

POTTS_TEST_CASE("Potts checkpoint round trip preserves byte spins and both RNG streams") {
    TempDirectory directory;
    const std::string base = directory.path + "/run";
    const PottsCheckpointState expected = make_state(2);
    POTTS_REQUIRE(save_potts_checkpoint(base, identity(), expected));
    require_equal(load(base), expected);
}

POTTS_TEST_CASE("Potts checkpoint CRC corruption falls back to retained previous") {
    TempDirectory directory;
    const std::string base = directory.path + "/run";
    const PottsCheckpointState previous = make_state(3);
    POTTS_REQUIRE(save_potts_checkpoint(base, identity(), previous));
    POTTS_REQUIRE(save_potts_checkpoint(base, identity(), make_state(4)));
    POTTS_REQUIRE(std::filesystem::exists(potts_checkpoint_path(base, ".prev.bin")));
    corrupt_payload(potts_checkpoint_path(base, ".bin"));
    require_equal(load(base), previous);
}

POTTS_TEST_CASE("Potts checkpoint complete temporary generation is recoverable") {
    TempDirectory directory;
    const std::string base = directory.path + "/run";
    const PottsCheckpointState expected = make_state(5);
    POTTS_REQUIRE(save_potts_checkpoint(base, identity(), expected));
    POTTS_REQUIRE(std::rename(potts_checkpoint_path(base, ".bin").c_str(),
                              potts_checkpoint_path(base, ".tmp").c_str()) == 0);
    require_equal(load(base), expected);
}

POTTS_TEST_CASE("Potts checkpoint done marker suppresses completed restart") {
    TempDirectory directory;
    const std::string base = directory.path + "/run";
    POTTS_REQUIRE(save_potts_checkpoint(base, identity(), make_state(6)));
    POTTS_REQUIRE(mark_potts_checkpoint_done(base));
    PottsCheckpointState state;
    POTTS_REQUIRE(load_potts_checkpoint(base, identity(), &state)
                  == PottsCheckpointLoadStatus::done);
}

POTTS_TEST_CASE("Potts checkpoint semantic validation rejects CRC-valid bad spins") {
    TempDirectory directory;
    const std::string base = directory.path + "/run";
    POTTS_REQUIRE(save_potts_checkpoint(base, identity(), make_state(7)));
    const std::string path = potts_checkpoint_path(base, ".bin");
    std::vector<std::uint8_t> bytes = read_bytes(path);
    constexpr std::size_t payload_start = 8U + 10U * 4U + 8U * 8U;
    POTTS_REQUIRE(bytes.size() > payload_start + 4U);
    bytes[payload_start] = kQ;
    const std::uint32_t checksum = crc32(bytes, bytes.size() - 4U);
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes[bytes.size() - 4U + shift / 8U] =
            static_cast<std::uint8_t>(checksum >> shift);
    write_bytes(path, bytes);
    PottsCheckpointState state;
    POTTS_REQUIRE(load_potts_checkpoint(base, identity(), &state)
                  == PottsCheckpointLoadStatus::invalid);
}

POTTS_TEST_CASE("Potts checkpoint semantic validation covers q energy and order") {
    PottsCheckpointState state = make_state(8);
    std::string error;
    state.energies[0] += 1;
    POTTS_REQUIRE(!validate_potts_checkpoint_state(identity(), state, &error));
    state = make_state(8);
    state.order[0] = state.order[1];
    POTTS_REQUIRE(!validate_potts_checkpoint_state(identity(), state, &error));
    state = make_state(8);
    state.spins[0] = kQ;
    POTTS_REQUIRE(!validate_potts_checkpoint_state(identity(), state, &error));
}

POTTS_TEST_CASE("Potts checkpoint identity includes the detailed cap") {
    TempDirectory directory;
    const std::string base = directory.path + "/run";
    POTTS_REQUIRE(save_potts_checkpoint(base, identity(), make_state(9)));
    PottsCheckpointIdentity changed = identity();
    changed.detailed_cap = -1;
    PottsCheckpointState state;
    POTTS_REQUIRE(load_potts_checkpoint(base, changed, &state)
                  == PottsCheckpointLoadStatus::invalid);
}
