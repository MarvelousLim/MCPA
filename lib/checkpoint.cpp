#include "checkpoint.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

bool file_exists(const char* path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

bool done_matches_identity(const CheckpointManager& manager, const char* path) {
    FILE* file = std::fopen(path, "r");
    if (!file) return false;
    char magic[16]{};
    int version = 0;
    int detail_cap = 0;
    const bool matches = std::fscanf(file, "%15s %d %d", magic, &version,
                                     &detail_cap) == 3
        && std::strcmp(magic, BC_CHECKPOINT_MAGIC) == 0
        && version == BC_CHECKPOINT_VERSION
        && detail_cap == manager.detail_cap;
    std::fclose(file);
    return matches;
}

bool ensure_directory(const char* path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (!error && std::filesystem::is_directory(path, error) && !error) return true;
    std::fprintf(stderr, "[chk] cannot create directory %s: %s\n", path,
                 error ? error.message().c_str() : "path is not a directory");
    return false;
}

void sync_directory(const char* path) {
#ifndef _WIN32
    const int descriptor = open(path, O_RDONLY | O_DIRECTORY);
    if (descriptor >= 0) { (void)fsync(descriptor); close(descriptor); }
#else
    (void)path;
#endif
}

size_t spin_word_count(size_t count) { return (count + 15U) / 16U; }

bool pack_spins(const int* spins, size_t count, std::vector<uint32_t>& packed) {
    packed.assign(spin_word_count(count), 0xffffffffU);
    for (size_t i = 0; i < count; ++i) {
        if (spins[i] < -1 || spins[i] > 1) return false;
        const uint32_t value = static_cast<uint32_t>(spins[i] + 1);
        const size_t word = i / 16U;
        const unsigned bit = static_cast<unsigned>((i % 16U) * 2U);
        packed[word] = (packed[word] & ~(0x3U << bit)) | (value << bit);
    }
    return true;
}

bool unpack_spins(const uint32_t* packed, size_t count, int* spins) {
    for (size_t i = 0; i < count; ++i) {
        const uint32_t value = (packed[i / 16U] >> ((i % 16U) * 2U)) & 0x3U;
        if (value == 3U) return false;
        spins[i] = static_cast<int>(value) - 1;
    }
    return true;
}

bool valid_population(const int* spins, const int* e_j, const int* e_delta,
                      const int* families, const int* order,
                      int L, int N, int R) {
    std::vector<unsigned char> order_seen(static_cast<size_t>(R), 0);
    for (int r = 0; r < R; ++r) {
        if (families[r] < 0 || families[r] >= R
            || order[r] < 0 || order[r] >= R || order_seen[order[r]]) return false;
        order_seen[order[r]] = 1;
        int exact_j = 0;
        int exact_delta = 0;
        const int* replica = spins + static_cast<size_t>(r) * N;
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                const int site = x + y * L;
                const int right = (x + 1) % L + y * L;
                const int down = x + ((y + 1) % L) * L;
                const int spin = replica[site];
                if (spin < -1 || spin > 1) return false;
                exact_j -= spin * (replica[right] + replica[down]);
                exact_delta += spin * spin;
            }
        }
        if (e_delta[r] < 0 || e_delta[r] > N
            || e_j[r] != exact_j || e_delta[r] != exact_delta) return false;
    }
    return true;
}

uint32_t crc32_update(uint32_t crc, const void* data, size_t length) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return crc;
}

uint32_t checkpoint_crc(BcCheckpointHeader header,
                        const void* payload, size_t payload_bytes) {
    header.checksum = 0;
    uint32_t crc = crc32_update(0xffffffffU, &header, sizeof(header));
    crc = crc32_update(crc, payload, payload_bytes);
    return crc ^ 0xffffffffU;
}

bool exact_file_size(FILE* file, size_t expected) {
    if (std::fseek(file, 0, SEEK_END) != 0) return false;
    const long size = std::ftell(file);
    return size >= 0 && static_cast<size_t>(size) == expected
        && std::fseek(file, static_cast<long>(sizeof(BcCheckpointHeader)), SEEK_SET) == 0;
}

bool load_file(const char* path,
               int L, int N, int R, int nSteps, int seed,
               int D_num, int D_denum, int heat, int detail_cap,
               int* spins, int* e_j, int* e_delta, int* families, int* order,
               int& U, int64_t& step_count,
               void* philox_states, size_t philox_state_bytes,
               uint64_t& pcg_state, uint64_t& pcg_stream,
               int64_t* output_positions) {
    if (philox_state_bytes && !philox_states) return false;
    FILE* file = std::fopen(path, "rb");
    if (!file) return false;
    BcCheckpointHeader header{};
    const bool header_ok = std::fread(&header, sizeof(header), 1, file) == 1
        && std::memcmp(header.magic, BC_CHECKPOINT_MAGIC, 8) == 0
        && header.version == BC_CHECKPOINT_VERSION
        && header.L == L && header.N == N && header.R == R
        && header.nSteps == nSteps && header.seed == seed
        && header.D_num == D_num && header.D_denum == D_denum
        && header.heat == heat && header.detail_cap == detail_cap
        && header.rng_state_bytes >= 0
        && header.step_count >= 0
        && header.out_pos[0] >= -1 && header.out_pos[1] >= -1
        && header.out_pos[2] >= -1
        && static_cast<uint64_t>(header.rng_state_bytes) == philox_state_bytes;
    if (!header_ok) {
        std::fprintf(stderr, "[chk] %s: incompatible header\n", path);
        std::fclose(file);
        return false;
    }
    const size_t spin_count = static_cast<size_t>(R) * N;
    const size_t words = spin_word_count(spin_count);
    const size_t row = static_cast<size_t>(R) * sizeof(int);
    const size_t payload_bytes = words * sizeof(uint32_t) + 4U * row
                               + philox_state_bytes;
    if (header.spin_words < 0 || static_cast<uint64_t>(header.spin_words) != words
        || header.payload_bytes < 0
        || static_cast<uint64_t>(header.payload_bytes) != payload_bytes
        || !exact_file_size(file, sizeof(header) + payload_bytes)) {
        std::fprintf(stderr, "[chk] %s: invalid payload size\n", path);
        std::fclose(file);
        return false;
    }
    std::vector<uint8_t> payload(payload_bytes);
    const bool read_ok = std::fread(payload.data(), 1, payload.size(), file)
                       == payload.size();
    std::fclose(file);
    if (!read_ok || checkpoint_crc(header, payload.data(), payload.size())
                    != header.checksum) {
        std::fprintf(stderr, "[chk] %s: CRC-32 mismatch or truncation\n", path);
        return false;
    }
    const uint8_t* cursor = payload.data();
    std::vector<uint32_t> packed(words);
    std::memcpy(packed.data(), cursor, words * sizeof(uint32_t));
    std::vector<int> loaded_spins(spin_count);
    std::vector<int> loaded_e_j(static_cast<size_t>(R));
    std::vector<int> loaded_e_delta(static_cast<size_t>(R));
    std::vector<int> loaded_families(static_cast<size_t>(R));
    std::vector<int> loaded_order(static_cast<size_t>(R));
    if (!unpack_spins(packed.data(), spin_count, loaded_spins.data()))
        return false;
    cursor += words * sizeof(uint32_t);
    std::memcpy(loaded_e_j.data(), cursor, row); cursor += row;
    std::memcpy(loaded_e_delta.data(), cursor, row); cursor += row;
    std::memcpy(loaded_families.data(), cursor, row); cursor += row;
    std::memcpy(loaded_order.data(), cursor, row); cursor += row;
    if (!valid_population(loaded_spins.data(), loaded_e_j.data(),
                          loaded_e_delta.data(), loaded_families.data(),
                          loaded_order.data(), L, N, R)
        || (header.resampling_rng_stream & 1ULL) == 0) {
        std::fprintf(stderr, "[chk] %s: semantically invalid BC state\n", path);
        return false;
    }
    std::memcpy(spins, loaded_spins.data(), spin_count * sizeof(int));
    std::memcpy(e_j, loaded_e_j.data(), row);
    std::memcpy(e_delta, loaded_e_delta.data(), row);
    std::memcpy(families, loaded_families.data(), row);
    std::memcpy(order, loaded_order.data(), row);
    if (philox_state_bytes) std::memcpy(philox_states, cursor, philox_state_bytes);
    U = header.U;
    step_count = header.step_count;
    pcg_state = header.resampling_rng_state;
    pcg_stream = header.resampling_rng_stream;
    if (output_positions)
        for (int i = 0; i < 3; ++i) output_positions[i] = header.out_pos[i];
    std::printf("[chk] Loaded %s (U=%d step=%lld)\n", path, U,
                static_cast<long long>(step_count));
    return true;
}

} // namespace

bool checkpoint_path(const CheckpointManager& manager, const char* suffix,
                     char* output, size_t output_size) {
    const int written = std::snprintf(output, output_size, "%s/%s%s%s",
                                      manager.chk_dir, manager.base_name,
                                      BC_CHECKPOINT_SUFFIX, suffix);
    if (written < 0 || static_cast<size_t>(written) >= output_size) {
        if (output_size) output[0] = '\0';
        return false;
    }
    return true;
}

bool checkpoint_init_bc(CheckpointManager& manager,
                        int L, int N, int R, int nSteps, int seed,
                        int D_num, int D_denum, int heat, int detail_cap,
                        const char* directory, bool enabled, int interval_secs) {
    (void)L;
    if (detail_cap < -1) return false;
    std::memset(&manager, 0, sizeof(manager));
    manager.enabled = enabled;
    manager.interval_secs = interval_secs;
    manager.detail_cap = detail_cap;
    manager.last_chk_time = std::time(nullptr);
    const int dir_written = std::snprintf(
        manager.chk_dir, sizeof(manager.chk_dir), "%s",
        directory ? directory : "checkpoints");
    const int base_written = std::snprintf(
        manager.base_name, sizeof(manager.base_name),
        "2DBlume%s_q3_D%dof%d_N%d_R%d_nSteps%d_run%d",
        heat ? "Heating" : "", D_num, D_denum, N, R, nSteps, seed);
    if (dir_written < 0 || static_cast<size_t>(dir_written) >= sizeof(manager.chk_dir)
        || base_written < 0
        || static_cast<size_t>(base_written) >= sizeof(manager.base_name)) return false;
    return !enabled || ensure_directory(manager.chk_dir);
}

bool checkpoint_should_save(const CheckpointManager& manager) {
    return manager.enabled
        && (std::time(nullptr) - manager.last_chk_time) >= manager.interval_secs;
}

bool checkpoint_save_bc(CheckpointManager& manager,
                        int L, int N, int R, int nSteps, int seed,
                        int D_num, int D_denum, int heat, int detail_cap,
                        const int* spins, const int* e_j, const int* e_delta,
                        const int* families, const int* order, int U,
                        const void* philox_states, size_t philox_state_bytes,
                        uint64_t pcg_state, uint64_t pcg_stream,
                        const int64_t* output_positions) {
    if (!manager.enabled) return true;
    if (detail_cap < -1 || !ensure_directory(manager.chk_dir)
        || (philox_state_bytes && !philox_states)
        || !valid_population(spins, e_j, e_delta, families, order, L, N, R))
        return false;
    const size_t spin_count = static_cast<size_t>(R) * N;
    std::vector<uint32_t> packed;
    if (!pack_spins(spins, spin_count, packed)) return false;
    const size_t row = static_cast<size_t>(R) * sizeof(int);
    std::vector<uint8_t> payload(packed.size() * sizeof(uint32_t)
                               + 4U * row + philox_state_bytes);
    uint8_t* cursor = payload.data();
    std::memcpy(cursor, packed.data(), packed.size() * sizeof(uint32_t));
    cursor += packed.size() * sizeof(uint32_t);
    std::memcpy(cursor, e_j, row); cursor += row;
    std::memcpy(cursor, e_delta, row); cursor += row;
    std::memcpy(cursor, families, row); cursor += row;
    std::memcpy(cursor, order, row); cursor += row;
    if (philox_state_bytes) std::memcpy(cursor, philox_states, philox_state_bytes);

    BcCheckpointHeader header{};
    std::memcpy(header.magic, BC_CHECKPOINT_MAGIC, 8);
    header.version = BC_CHECKPOINT_VERSION;
    header.L = L; header.N = N; header.R = R;
    header.nSteps = nSteps; header.seed = seed;
    header.D_num = D_num; header.D_denum = D_denum; header.heat = heat;
    header.detail_cap = detail_cap;
    header.U = U; header.step_count = manager.step_count;
    header.timestamp = static_cast<int64_t>(std::time(nullptr));
    header.spin_words = static_cast<int64_t>(packed.size());
    header.rng_state_bytes = static_cast<int64_t>(philox_state_bytes);
    header.payload_bytes = static_cast<int64_t>(payload.size());
    for (int i = 0; i < 3; ++i)
        header.out_pos[i] = output_positions ? output_positions[i] : -1;
    header.resampling_rng_state = pcg_state;
    header.resampling_rng_stream = pcg_stream | 1ULL;
    header.checksum = checkpoint_crc(header, payload.data(), payload.size());

    char temporary[BC_CHECKPOINT_PATH_CAPACITY];
    char current[BC_CHECKPOINT_PATH_CAPACITY];
    char previous[BC_CHECKPOINT_PATH_CAPACITY];
    char done[BC_CHECKPOINT_PATH_CAPACITY];
    if (!checkpoint_path(manager, ".tmp", temporary, sizeof(temporary))
        || !checkpoint_path(manager, ".bin", current, sizeof(current))
        || !checkpoint_path(manager, ".prev.bin", previous, sizeof(previous))
        || !checkpoint_path(manager, ".done", done, sizeof(done))) return false;
    (void)std::remove(done);
    FILE* file = std::fopen(temporary, "wb");
    if (!file) return false;
    bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1
           && std::fwrite(payload.data(), 1, payload.size(), file) == payload.size()
           && std::fflush(file) == 0;
#ifndef _WIN32
    if (ok) ok = fsync(fileno(file)) == 0;
#endif
    if (std::fclose(file) != 0) ok = false;
    if (!ok) return false;

    std::error_code error;
    if (file_exists(previous)) std::filesystem::remove(previous, error);
    error.clear();
    if (file_exists(current)) {
        std::filesystem::rename(current, previous, error);
        if (error) return false;
    }
    error.clear();
    std::filesystem::rename(temporary, current, error);
    if (error) return false;
    sync_directory(manager.chk_dir);
    manager.last_chk_time = std::time(nullptr);
    return true;
}

bool checkpoint_load_bc(const CheckpointManager& manager,
                        int L, int N, int R, int nSteps, int seed,
                        int D_num, int D_denum, int heat, int detail_cap,
                        int* spins, int* e_j, int* e_delta,
                        int* families, int* order,
                        int& U, int64_t& step_count,
                        void* philox_states, size_t philox_state_bytes,
                        uint64_t& pcg_state, uint64_t& pcg_stream,
                        int64_t* output_positions) {
    if (!manager.enabled) return false;
    char done[BC_CHECKPOINT_PATH_CAPACITY];
    if (!checkpoint_path(manager, ".done", done, sizeof(done))) return false;
    if (file_exists(done)) return false;
    for (const char* suffix : {".bin", ".prev.bin", ".tmp"}) {
        char path[BC_CHECKPOINT_PATH_CAPACITY];
        if (!checkpoint_path(manager, suffix, path, sizeof(path))) return false;
        if (load_file(path, L, N, R, nSteps, seed, D_num, D_denum, heat,
                      detail_cap,
                      spins, e_j, e_delta, families, order, U, step_count,
                      philox_states, philox_state_bytes, pcg_state, pcg_stream,
                      output_positions)) return true;
    }
    return false;
}

bool checkpoint_exists(const CheckpointManager& manager) {
    if (!manager.enabled) return false;
    char done[BC_CHECKPOINT_PATH_CAPACITY];
    if (!checkpoint_path(manager, ".done", done, sizeof(done))) return false;
    if (file_exists(done)) return !done_matches_identity(manager, done);
    for (const char* suffix : {".bin", ".prev.bin", ".tmp"}) {
        char path[BC_CHECKPOINT_PATH_CAPACITY];
        if (!checkpoint_path(manager, suffix, path, sizeof(path))) return false;
        if (file_exists(path)) return true;
    }
    return false;
}

bool checkpoint_is_done(const CheckpointManager& manager) {
    if (!manager.enabled) return false;
    char done[BC_CHECKPOINT_PATH_CAPACITY];
    return checkpoint_path(manager, ".done", done, sizeof(done))
        && file_exists(done) && done_matches_identity(manager, done);
}

bool checkpoint_mark_done(const CheckpointManager& manager) {
    if (!manager.enabled) return true;
    char done[BC_CHECKPOINT_PATH_CAPACITY];
    if (!checkpoint_path(manager, ".done", done, sizeof(done))) return false;
    FILE* file = std::fopen(done, "w");
    if (!file) return false;
    const time_t now = std::time(nullptr);
    const bool ok = std::fprintf(file, "%s %d %d completed %lld\n",
                                 BC_CHECKPOINT_MAGIC, BC_CHECKPOINT_VERSION,
                                 manager.detail_cap,
                                 static_cast<long long>(now)) > 0
                 && std::fflush(file) == 0;
#ifndef _WIN32
    if (ok) (void)fsync(fileno(file));
#endif
    std::fclose(file);
    sync_directory(manager.chk_dir);
    return ok;
}
