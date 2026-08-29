#include "checkpoint_1d.h"

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

constexpr char kMagic[8] = {'M', 'C', 'P', 'A', '1', 'D', 'I', '1'};
constexpr std::uint32_t kVersion = 3;

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

bool take_u32(const std::vector<std::uint8_t>& bytes, std::size_t* offset,
              std::uint32_t* value) {
    if (*offset > bytes.size() || bytes.size() - *offset < 4) return false;
    *value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        *value |= static_cast<std::uint32_t>(bytes[(*offset)++]) << shift;
    return true;
}

bool take_u64(const std::vector<std::uint8_t>& bytes, std::size_t* offset,
              std::uint64_t* value) {
    if (*offset > bytes.size() || bytes.size() - *offset < 8) return false;
    *value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        *value |= static_cast<std::uint64_t>(bytes[(*offset)++]) << shift;
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

bool valid_state_shape(const Ising1DCheckpointIdentity& identity,
                       const Ising1DCheckpointState& state,
                       std::string* error) {
    std::size_t total_spins = 0;
    if (identity.N <= 0 || identity.R <= 0 || identity.n_steps <= 0
        || identity.detailed_cap < -1
        || !checked_product(static_cast<std::size_t>(identity.N),
                            static_cast<std::size_t>(identity.R), &total_spins)) {
        if (error) *error = "invalid checkpoint identity";
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
    if (!std::all_of(state.spins.begin(), state.spins.end(),
                     [](char spin) { return spin == -1 || spin == 1; })) {
        if (error) *error = "checkpoint contains a spin outside {-1,+1}";
        return false;
    }
    if (!std::all_of(state.output_offsets.begin(), state.output_offsets.end(),
                     [](std::int64_t offset) { return offset >= 0; })) {
        if (error) *error = "checkpoint contains a negative output offset";
        return false;
    }
    std::vector<bool> order_seen(static_cast<std::size_t>(identity.R), false);
    for (int replica = 0; replica < identity.R; ++replica) {
        const int family = state.families[static_cast<std::size_t>(replica)];
        const int ordered = state.order[static_cast<std::size_t>(replica)];
        if (family < 0 || family >= identity.R) {
            if (error) *error = "checkpoint family ID is outside [0,R)";
            return false;
        }
        if (ordered < 0 || ordered >= identity.R
            || order_seen[static_cast<std::size_t>(ordered)]) {
            if (error) *error = "checkpoint order is not a permutation of [0,R)";
            return false;
        }
        order_seen[static_cast<std::size_t>(ordered)] = true;

        int exact_energy = 0;
        const std::size_t shift
            = static_cast<std::size_t>(replica) * identity.N;
        for (int site = 0; site < identity.N; ++site) {
            const int right = (site + 1) % identity.N;
            exact_energy -= state.spins[shift + static_cast<std::size_t>(site)]
                            * state.spins[shift + static_cast<std::size_t>(right)];
        }
        if (state.energies[static_cast<std::size_t>(replica)] != exact_energy) {
            if (error) *error = "checkpoint energy does not match packed spins";
            return false;
        }
    }
    return true;
}

std::vector<std::uint8_t> serialize(const Ising1DCheckpointIdentity& identity,
                                    const Ising1DCheckpointState& state) {
    std::vector<std::uint8_t> bytes;
    const std::size_t spin_bytes = (state.spins.size() + 7U) / 8U;
    bytes.reserve(128 + spin_bytes + 3U * state.energies.size() * 4U
                  + state.philox.size());
    bytes.insert(bytes.end(), kMagic, kMagic + sizeof(kMagic));
    append_u32(bytes, kVersion);
    append_u32(bytes, static_cast<std::uint32_t>(identity.N));
    append_u32(bytes, static_cast<std::uint32_t>(identity.R));
    append_u32(bytes, static_cast<std::uint32_t>(identity.n_steps));
    append_u32(bytes, static_cast<std::uint32_t>(identity.seed));
    append_u32(bytes, static_cast<std::uint32_t>(identity.detailed_cap));
    append_u32(bytes, static_cast<std::uint32_t>(state.U));
    append_u64(bytes, state.completed_shells);
    append_u64(bytes, static_cast<std::uint64_t>(spin_bytes));
    append_u64(bytes, static_cast<std::uint64_t>(state.philox.size()));
    for (std::int64_t value : state.output_offsets)
        append_u64(bytes, static_cast<std::uint64_t>(value));
    append_u64(bytes, state.resampling_rng.state);
    append_u64(bytes, state.resampling_rng.stream);

    const std::size_t packed_start = bytes.size();
    bytes.resize(packed_start + spin_bytes, 0);
    for (std::size_t i = 0; i < state.spins.size(); ++i)
        if (state.spins[i] == 1) bytes[packed_start + i / 8U] |= 1U << (i % 8U);
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
    const long size = std::ftell(file);
    if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file); return false;
    }
    bytes->resize(static_cast<std::size_t>(size));
    const bool ok = bytes->empty()
        || std::fread(bytes->data(), 1, bytes->size(), file) == bytes->size();
    const bool closed = std::fclose(file) == 0;
    return ok && closed;
}

bool parse(const std::vector<std::uint8_t>& bytes,
           const Ising1DCheckpointIdentity& expected,
           Ising1DCheckpointState* state, std::string* error) {
    if (bytes.size() < 8 + 7 * 4 + 8 * 8 + 4
        || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
        if (error) *error = "bad magic or truncated header";
        return false;
    }
    std::size_t offset = 8;
    std::uint32_t version = 0, N = 0, R = 0, n_steps = 0;
    std::uint32_t seed = 0, detailed_cap = 0, U = 0;
    std::uint64_t completed = 0, spin_bytes = 0, philox_bytes = 0;
    if (!take_u32(bytes, &offset, &version) || !take_u32(bytes, &offset, &N)
        || !take_u32(bytes, &offset, &R) || !take_u32(bytes, &offset, &n_steps)
        || !take_u32(bytes, &offset, &seed)
        || !take_u32(bytes, &offset, &detailed_cap)
        || !take_u32(bytes, &offset, &U)
        || !take_u64(bytes, &offset, &completed)
        || !take_u64(bytes, &offset, &spin_bytes)
        || !take_u64(bytes, &offset, &philox_bytes)) {
        if (error) *error = "truncated header";
        return false;
    }
    Ising1DCheckpointState decoded;
    for (std::int64_t& value : decoded.output_offsets) {
        std::uint64_t raw = 0;
        if (!take_u64(bytes, &offset, &raw)) { if (error) *error = "truncated offsets"; return false; }
        value = static_cast<std::int64_t>(raw);
    }
    if (!take_u64(bytes, &offset, &decoded.resampling_rng.state)
        || !take_u64(bytes, &offset, &decoded.resampling_rng.stream)) {
        if (error) *error = "truncated RNG state";
        return false;
    }
    if (version != kVersion || N != static_cast<std::uint32_t>(expected.N)
        || R != static_cast<std::uint32_t>(expected.R)
        || n_steps != static_cast<std::uint32_t>(expected.n_steps)
        || seed != static_cast<std::uint32_t>(expected.seed)
        || detailed_cap != static_cast<std::uint32_t>(expected.detailed_cap)
        || philox_bytes != expected.philox_bytes) {
        if (error) *error = "checkpoint identity/version mismatch";
        return false;
    }
    std::size_t total_spins = 0;
    if (!checked_product(N, R, &total_spins)
        || spin_bytes != (total_spins + 7U) / 8U
        || philox_bytes > std::numeric_limits<std::size_t>::max()) {
        if (error) *error = "invalid payload dimensions";
        return false;
    }
    std::size_t payload_size = static_cast<std::size_t>(spin_bytes);
    const std::size_t vector_bytes = static_cast<std::size_t>(R) * 3U * 4U;
    if (payload_size > std::numeric_limits<std::size_t>::max() - vector_bytes
        || payload_size + vector_bytes > std::numeric_limits<std::size_t>::max()
                                        - static_cast<std::size_t>(philox_bytes)) {
        if (error) *error = "payload size overflow";
        return false;
    }
    payload_size += vector_bytes + static_cast<std::size_t>(philox_bytes);
    if (offset > bytes.size() || bytes.size() - offset != payload_size + 4U) {
        if (error) *error = "truncated or trailing payload";
        return false;
    }
    std::uint32_t stored_crc = 0;
    std::size_t crc_offset = bytes.size() - 4U;
    if (!take_u32(bytes, &crc_offset, &stored_crc)
        || stored_crc != crc32(bytes.data(), bytes.size() - 4U)) {
        if (error) *error = "CRC32 mismatch";
        return false;
    }
    decoded.U = static_cast<std::int32_t>(U);
    decoded.completed_shells = completed;
    if (total_spins % 8U != 0U) {
        const unsigned used_bits = static_cast<unsigned>(total_spins % 8U);
        const std::uint8_t unused_mask
            = static_cast<std::uint8_t>(0xffU << used_bits);
        if ((bytes[offset + static_cast<std::size_t>(spin_bytes) - 1U]
             & unused_mask) != 0U) {
            if (error) *error = "checkpoint has nonzero packed-spin padding";
            return false;
        }
    }
    decoded.spins.resize(total_spins);
    for (std::size_t i = 0; i < total_spins; ++i)
        decoded.spins[i] = (bytes[offset + i / 8U] & (1U << (i % 8U))) ? 1 : -1;
    offset += static_cast<std::size_t>(spin_bytes);
    auto read_ints = [&](std::vector<int>* values) {
        values->resize(R);
        for (std::size_t i = 0; i < R; ++i) {
            std::uint32_t raw = 0;
            if (!take_u32(bytes, &offset, &raw)) return false;
            (*values)[i] = static_cast<std::int32_t>(raw);
        }
        return true;
    };
    if (!read_ints(&decoded.energies) || !read_ints(&decoded.families)
        || !read_ints(&decoded.order)) {
        if (error) *error = "truncated integer arrays";
        return false;
    }
    decoded.philox.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                          bytes.begin() + static_cast<std::ptrdiff_t>(offset + philox_bytes));
    if ((decoded.resampling_rng.stream & 1U) == 0) {
        if (error) *error = "invalid even PCG stream";
        return false;
    }
    if (!valid_state_shape(expected, decoded, error)) return false;
    *state = std::move(decoded);
    return true;
}

bool path_exists(const std::string& path) {
    std::error_code error;
    return std::filesystem::exists(path, error);
}

bool write_tmp(const std::string& path, const std::vector<std::uint8_t>& bytes,
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

std::string ising1d_checkpoint_path(const std::string& base, const char* suffix) {
    return base + "_chk" + suffix;
}

bool save_ising1d_checkpoint(const std::string& base,
                             const Ising1DCheckpointIdentity& identity,
                             const Ising1DCheckpointState& state,
                             std::string* error) {
    if (!valid_state_shape(identity, state, error)) return false;
    std::error_code fs_error;
    const std::filesystem::path parent = std::filesystem::path(base).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, fs_error);
    if (fs_error) { if (error) *error = fs_error.message(); return false; }
    const std::string current = ising1d_checkpoint_path(base, ".bin");
    const std::string previous = ising1d_checkpoint_path(base, ".prev.bin");
    const std::string temporary = ising1d_checkpoint_path(base, ".tmp");
    const std::string done = ising1d_checkpoint_path(base, ".done");
    const std::vector<std::uint8_t> bytes = serialize(identity, state);
    if (!write_tmp(temporary, bytes, error)) return false;
    if (path_exists(done)) {
        std::filesystem::remove(done, fs_error);
        if (fs_error) { if (error) *error = fs_error.message(); return false; }
    }
    if (path_exists(previous)) {
        std::filesystem::remove(previous, fs_error);
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

Ising1DCheckpointLoadStatus load_ising1d_checkpoint(
    const std::string& base, const Ising1DCheckpointIdentity& identity,
    Ising1DCheckpointState* state, std::string* loaded_path, std::string* error) {
    if (path_exists(ising1d_checkpoint_path(base, ".done")))
        return Ising1DCheckpointLoadStatus::done;
    const std::array<std::string, 3> candidates{{
        ising1d_checkpoint_path(base, ".bin"),
        ising1d_checkpoint_path(base, ".prev.bin"),
        ising1d_checkpoint_path(base, ".tmp")}};
    bool found = false;
    std::string last_error;
    for (const std::string& path : candidates) {
        if (!path_exists(path)) continue;
        found = true;
        std::vector<std::uint8_t> bytes;
        Ising1DCheckpointState decoded;
        if (read_file(path, &bytes) && parse(bytes, identity, &decoded, &last_error)) {
            *state = std::move(decoded);
            if (loaded_path) *loaded_path = path;
            return Ising1DCheckpointLoadStatus::loaded;
        }
    }
    if (error) *error = found ? last_error : std::string{};
    return found ? Ising1DCheckpointLoadStatus::invalid
                 : Ising1DCheckpointLoadStatus::not_found;
}

bool mark_ising1d_checkpoint_done(const std::string& base, std::string* error) {
    const std::string path = ising1d_checkpoint_path(base, ".done");
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
