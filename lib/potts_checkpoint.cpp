#include "potts_checkpoint.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <system_error>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

constexpr char kMagic[8] = {'M', 'C', 'P', 'A', '2', 'D', 'P', '3'};
constexpr std::uint32_t kVersion = 3;

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

bool take_u32(const std::vector<std::uint8_t>& in, std::size_t* offset,
              std::uint32_t* value) {
    if (*offset > in.size() || in.size() - *offset < 4) return false;
    *value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        *value |= static_cast<std::uint32_t>(in[(*offset)++]) << shift;
    return true;
}

bool take_u64(const std::vector<std::uint8_t>& in, std::size_t* offset,
              std::uint64_t* value) {
    if (*offset > in.size() || in.size() - *offset < 8) return false;
    *value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        *value |= static_cast<std::uint64_t>(in[(*offset)++]) << shift;
    return true;
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return crc ^ 0xffffffffU;
}

bool checked_product(std::size_t a, std::size_t b, std::size_t* product) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    *product = a * b;
    return true;
}

int independent_energy(const char* spins, int L) {
    int energy = 0;
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = x + y * L;
            const int right = (x + 1) % L + y * L;
            const int down = x + ((y + 1) % L) * L;
            energy -= spins[site] == spins[right];
            energy -= spins[site] == spins[down];
        }
    }
    return energy;
}

std::vector<std::uint8_t> serialize(const PottsCheckpointIdentity& identity,
                                    const PottsCheckpointState& state) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(128 + state.spins.size() + 12U * state.energies.size()
                  + state.philox.size());
    bytes.insert(bytes.end(), kMagic, kMagic + sizeof(kMagic));
    append_u32(bytes, kVersion);
    append_u32(bytes, static_cast<std::uint32_t>(identity.L));
    append_u32(bytes, static_cast<std::uint32_t>(identity.N));
    append_u32(bytes, static_cast<std::uint32_t>(identity.R));
    append_u32(bytes, static_cast<std::uint32_t>(identity.n_steps));
    append_u32(bytes, static_cast<std::uint32_t>(identity.seed));
    append_u32(bytes, static_cast<std::uint32_t>(identity.q));
    append_u32(bytes, identity.heat ? 1U : 0U);
    append_u32(bytes, static_cast<std::uint32_t>(identity.detailed_cap));
    append_u32(bytes, static_cast<std::uint32_t>(state.U));
    append_u64(bytes, state.completed_shells);
    append_u64(bytes, static_cast<std::uint64_t>(state.spins.size()));
    append_u64(bytes, static_cast<std::uint64_t>(state.philox.size()));
    for (std::int64_t offset : state.output_offsets)
        append_u64(bytes, static_cast<std::uint64_t>(offset));
    append_u64(bytes, state.resampling_rng.state);
    append_u64(bytes, state.resampling_rng.stream);
    bytes.insert(bytes.end(), state.spins.begin(), state.spins.end());
    for (int value : state.energies) append_u32(bytes, static_cast<std::uint32_t>(value));
    for (int value : state.families) append_u32(bytes, static_cast<std::uint32_t>(value));
    for (int value : state.order) append_u32(bytes, static_cast<std::uint32_t>(value));
    bytes.insert(bytes.end(), state.philox.begin(), state.philox.end());
    append_u32(bytes, crc32(bytes.data(), bytes.size()));
    return bytes;
}

bool read_file(const std::string& path, std::vector<std::uint8_t>* bytes) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    if (std::fseek(file, 0, SEEK_END) != 0) { std::fclose(file); return false; }
    const long length = std::ftell(file);
    if (length < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file); return false;
    }
    bytes->resize(static_cast<std::size_t>(length));
    const bool read = bytes->empty()
        || std::fread(bytes->data(), 1, bytes->size(), file) == bytes->size();
    const bool closed = std::fclose(file) == 0;
    return read && closed;
}

bool parse(const std::vector<std::uint8_t>& bytes,
           const PottsCheckpointIdentity& expected,
           PottsCheckpointState* state, std::string* error) {
    if (bytes.size() < 8 + 10 * 4 + 8 * 8 + 4
        || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
        if (error) *error = "bad magic or truncated checkpoint";
        return false;
    }
    std::size_t offset = 8;
    std::uint32_t version = 0, L = 0, N = 0, R = 0, n_steps = 0;
    std::uint32_t seed = 0, q = 0, heat = 0, detailed_cap = 0, U = 0;
    std::uint64_t shells = 0, spin_bytes = 0, philox_bytes = 0;
    if (!take_u32(bytes, &offset, &version) || !take_u32(bytes, &offset, &L)
        || !take_u32(bytes, &offset, &N) || !take_u32(bytes, &offset, &R)
        || !take_u32(bytes, &offset, &n_steps) || !take_u32(bytes, &offset, &seed)
        || !take_u32(bytes, &offset, &q) || !take_u32(bytes, &offset, &heat)
        || !take_u32(bytes, &offset, &detailed_cap)
        || !take_u32(bytes, &offset, &U) || !take_u64(bytes, &offset, &shells)
        || !take_u64(bytes, &offset, &spin_bytes)
        || !take_u64(bytes, &offset, &philox_bytes)) {
        if (error) *error = "truncated checkpoint header";
        return false;
    }
    PottsCheckpointState decoded;
    for (std::int64_t& output_offset : decoded.output_offsets) {
        std::uint64_t raw = 0;
        if (!take_u64(bytes, &offset, &raw)) { if (error) *error = "truncated output offsets"; return false; }
        output_offset = static_cast<std::int64_t>(raw);
    }
    if (!take_u64(bytes, &offset, &decoded.resampling_rng.state)
        || !take_u64(bytes, &offset, &decoded.resampling_rng.stream)) {
        if (error) *error = "truncated PCG state";
        return false;
    }
    if (version != kVersion || L != static_cast<std::uint32_t>(expected.L)
        || N != static_cast<std::uint32_t>(expected.N)
        || R != static_cast<std::uint32_t>(expected.R)
        || n_steps != static_cast<std::uint32_t>(expected.n_steps)
        || seed != static_cast<std::uint32_t>(expected.seed)
        || q != static_cast<std::uint32_t>(expected.q)
        || heat != static_cast<std::uint32_t>(expected.heat)
        || detailed_cap != static_cast<std::uint32_t>(expected.detailed_cap)
        || philox_bytes != expected.philox_bytes) {
        if (error) *error = "checkpoint identity/version mismatch";
        return false;
    }
    std::size_t total_spins = 0;
    if (!checked_product(N, R, &total_spins) || spin_bytes != total_spins
        || philox_bytes > std::numeric_limits<std::size_t>::max()) {
        if (error) *error = "invalid checkpoint dimensions";
        return false;
    }
    const std::size_t arrays = static_cast<std::size_t>(R) * 12U;
    if (total_spins > std::numeric_limits<std::size_t>::max() - arrays
        || total_spins + arrays > std::numeric_limits<std::size_t>::max()
                                  - static_cast<std::size_t>(philox_bytes)) {
        if (error) *error = "checkpoint payload size overflow";
        return false;
    }
    const std::size_t payload = total_spins + arrays
                              + static_cast<std::size_t>(philox_bytes);
    if (offset > bytes.size() || bytes.size() - offset != payload + 4U) {
        if (error) *error = "truncated or trailing checkpoint payload";
        return false;
    }
    std::size_t crc_offset = bytes.size() - 4U;
    std::uint32_t stored_crc = 0;
    if (!take_u32(bytes, &crc_offset, &stored_crc)
        || stored_crc != crc32(bytes.data(), bytes.size() - 4U)) {
        if (error) *error = "CRC32 mismatch";
        return false;
    }
    decoded.U = static_cast<std::int32_t>(U);
    decoded.completed_shells = shells;
    decoded.spins.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                         bytes.begin() + static_cast<std::ptrdiff_t>(offset + total_spins));
    offset += total_spins;
    auto take_ints = [&](std::vector<int>* values) {
        values->resize(R);
        for (std::size_t i = 0; i < R; ++i) {
            std::uint32_t raw = 0;
            if (!take_u32(bytes, &offset, &raw)) return false;
            (*values)[i] = static_cast<std::int32_t>(raw);
        }
        return true;
    };
    if (!take_ints(&decoded.energies) || !take_ints(&decoded.families)
        || !take_ints(&decoded.order)) {
        if (error) *error = "truncated integer arrays";
        return false;
    }
    decoded.philox.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                          bytes.begin() + static_cast<std::ptrdiff_t>(offset + philox_bytes));
    if (!validate_potts_checkpoint_state(expected, decoded, error)) return false;
    *state = std::move(decoded);
    return true;
}

bool path_exists(const std::string& path) {
    std::error_code error;
    return std::filesystem::exists(path, error);
}

bool write_synced(const std::string& path, const std::vector<std::uint8_t>& bytes,
                  std::string* error) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) { if (error) *error = std::strerror(errno); return false; }
    bool ok = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    ok = std::fflush(file) == 0 && ok;
#ifndef _WIN32
    ok = ::fsync(::fileno(file)) == 0 && ok;
#endif
    ok = std::fclose(file) == 0 && ok;
    if (!ok && error) *error = "checkpoint write/sync failed";
    return ok;
}

} // namespace

std::string potts_checkpoint_path(const std::string& base, const char* suffix) {
    return base + "_chk" + suffix;
}

bool validate_potts_checkpoint_state(const PottsCheckpointIdentity& identity,
                                     const PottsCheckpointState& state,
                                     std::string* error) {
    std::size_t total_spins = 0;
    if (identity.L < 2
        || static_cast<long long>(identity.N)
               != static_cast<long long>(identity.L) * identity.L
        || identity.R <= 0 || identity.n_steps <= 0
        || identity.detailed_cap < -1
        || !potts_supported_q(identity.q)
        || !checked_product(static_cast<std::size_t>(identity.N),
                            static_cast<std::size_t>(identity.R), &total_spins)) {
        if (error) *error = "invalid Potts checkpoint identity";
        return false;
    }
    if (state.spins.size() != total_spins
        || state.energies.size() != static_cast<std::size_t>(identity.R)
        || state.families.size() != static_cast<std::size_t>(identity.R)
        || state.order.size() != static_cast<std::size_t>(identity.R)
        || state.philox.size() != identity.philox_bytes) {
        if (error) *error = "checkpoint state dimensions do not match identity";
        return false;
    }
    if (state.U < -2 * identity.N || state.U > 0) {
        if (error) *error = "checkpoint U is outside the Potts spectrum";
        return false;
    }
    if ((state.resampling_rng.stream & 1U) == 0) {
        if (error) *error = "checkpoint PCG stream is even";
        return false;
    }
    for (std::int64_t offset : state.output_offsets) {
        if (offset < 0) { if (error) *error = "negative output offset"; return false; }
    }
    std::vector<bool> seen(static_cast<std::size_t>(identity.R), false);
    for (int r = 0; r < identity.R; ++r) {
        if (state.families[r] < 0 || state.families[r] >= identity.R
            || state.order[r] < 0 || state.order[r] >= identity.R
            || seen[state.order[r]]) {
            if (error) *error = "invalid family or non-permutation energy order";
            return false;
        }
        seen[state.order[r]] = true;
        const char* replica = state.spins.data()
                            + static_cast<std::size_t>(r) * identity.N;
        for (int site = 0; site < identity.N; ++site) {
            if (replica[site] < 0 || replica[site] >= identity.q) {
                if (error) *error = "spin outside runtime q domain";
                return false;
            }
        }
        if (state.energies[r] != independent_energy(replica, identity.L)) {
            if (error) *error = "tracked energy disagrees with Potts spins";
            return false;
        }
    }
    return true;
}

bool save_potts_checkpoint(const std::string& base,
                           const PottsCheckpointIdentity& identity,
                           const PottsCheckpointState& state,
                           std::string* error) {
    if (!validate_potts_checkpoint_state(identity, state, error)) return false;
    const std::filesystem::path parent = std::filesystem::path(base).parent_path();
    std::error_code fs_error;
    if (!parent.empty()) std::filesystem::create_directories(parent, fs_error);
    if (fs_error) { if (error) *error = fs_error.message(); return false; }
    const std::string current = potts_checkpoint_path(base, ".bin");
    const std::string previous = potts_checkpoint_path(base, ".prev.bin");
    const std::string temporary = potts_checkpoint_path(base, ".tmp");
    const std::string done = potts_checkpoint_path(base, ".done");
    if (!write_synced(temporary, serialize(identity, state), error)) return false;
    for (const std::string* obsolete : {&done, &previous}) {
        if (!path_exists(*obsolete)) continue;
        std::filesystem::remove(*obsolete, fs_error);
        if (fs_error) { if (error) *error = fs_error.message(); return false; }
    }
    if (path_exists(current)) {
        std::filesystem::rename(current, previous, fs_error);
        if (fs_error) { if (error) *error = fs_error.message(); return false; }
    }
    std::filesystem::rename(temporary, current, fs_error);
    if (fs_error) { if (error) *error = fs_error.message(); return false; }
#ifndef _WIN32
    if (!parent.empty()) {
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
        if (directory >= 0) { (void)::fsync(directory); ::close(directory); }
    }
#endif
    return true;
}

PottsCheckpointLoadStatus load_potts_checkpoint(
    const std::string& base, const PottsCheckpointIdentity& identity,
    PottsCheckpointState* state, std::string* loaded_path, std::string* error) {
    if (path_exists(potts_checkpoint_path(base, ".done")))
        return PottsCheckpointLoadStatus::done;
    const std::array<std::string, 3> candidates{{
        potts_checkpoint_path(base, ".bin"),
        potts_checkpoint_path(base, ".prev.bin"),
        potts_checkpoint_path(base, ".tmp")}};
    bool found = false;
    std::string last_error;
    for (const std::string& path : candidates) {
        if (!path_exists(path)) continue;
        found = true;
        std::vector<std::uint8_t> bytes;
        PottsCheckpointState decoded;
        if (read_file(path, &bytes) && parse(bytes, identity, &decoded, &last_error)) {
            *state = std::move(decoded);
            if (loaded_path) *loaded_path = path;
            return PottsCheckpointLoadStatus::loaded;
        }
    }
    if (error) *error = found ? last_error : std::string{};
    return found ? PottsCheckpointLoadStatus::invalid
                 : PottsCheckpointLoadStatus::not_found;
}

bool mark_potts_checkpoint_done(const std::string& base, std::string* error) {
    const std::string path = potts_checkpoint_path(base, ".done");
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) { if (error) *error = std::strerror(errno); return false; }
    constexpr char marker[] = "complete\n";
    bool ok = std::fwrite(marker, 1, sizeof(marker) - 1, file) == sizeof(marker) - 1;
    ok = std::fflush(file) == 0 && ok;
#ifndef _WIN32
    ok = ::fsync(::fileno(file)) == 0 && ok;
#endif
    ok = std::fclose(file) == 0 && ok;
    if (!ok && error) *error = "done marker write/sync failed";
    return ok;
}
