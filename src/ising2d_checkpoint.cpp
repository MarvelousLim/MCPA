#include "mcpa/ising2d_checkpoint.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>

namespace mcpa::ising2d {
namespace {

constexpr unsigned char magic[8] = {'M', '2', 'D', 'I', 'C', 'K', 'P', '1'};
// Version 2 binds checkpoints to the rectangular main-table schema that
// includes one-time GPU allocation metadata.  A v1 checkpoint must not append
// 23-column rows beneath its former 16-column header.
constexpr std::uint32_t format_version = 2;
constexpr char done_text[] = "MCPA_2D_ISING_DONE_V2\n";

std::string done_contents(const CheckpointIdentity& identity) {
    std::ostringstream output;
    output << done_text
           << identity.linear_size << ' ' << identity.site_count << ' '
           << identity.replicas << ' ' << identity.sweeps << ' '
           << identity.seed << ' ' << static_cast<int>(identity.direction) << ' '
           << identity.detailed_cap << ' ' << identity.philox_state_bytes << '\n';
    return output.str();
}

[[noreturn]] void throw_system(const std::string& operation) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
}

std::uint32_t crc32(const unsigned char* data, std::size_t size) noexcept {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return ~crc;
}

template <typename UInt>
void append_unsigned(std::vector<unsigned char>& bytes, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t index = 0; index < sizeof(UInt); ++index) {
        bytes.push_back(static_cast<unsigned char>(value & 0xffU));
        value >>= 8U;
    }
}

template <typename Signed>
void append_signed(std::vector<unsigned char>& bytes, Signed value) {
    using UInt = std::make_unsigned_t<Signed>;
    append_unsigned(bytes, static_cast<UInt>(value));
}

class Reader {
public:
    Reader(const std::vector<unsigned char>& bytes, std::size_t limit)
        : bytes_(bytes), limit_(limit) {}

    template <typename UInt>
    UInt read_unsigned() {
        static_assert(std::is_unsigned_v<UInt>);
        require(sizeof(UInt));
        UInt value = 0;
        for (std::size_t index = 0; index < sizeof(UInt); ++index)
            value |= static_cast<UInt>(bytes_[position_++]) << (8U * index);
        return value;
    }

    template <typename Signed>
    Signed read_signed() {
        using UInt = std::make_unsigned_t<Signed>;
        return static_cast<Signed>(read_unsigned<UInt>());
    }

    std::vector<unsigned char> read_bytes(std::size_t count) {
        require(count);
        std::vector<unsigned char> result(
            bytes_.begin() + static_cast<std::ptrdiff_t>(position_),
            bytes_.begin() + static_cast<std::ptrdiff_t>(position_ + count));
        position_ += count;
        return result;
    }

    void expect(const unsigned char* expected, std::size_t count) {
        require(count);
        if (!std::equal(expected, expected + count, bytes_.begin() + position_))
            throw std::runtime_error("checkpoint magic mismatch");
        position_ += count;
    }

    [[nodiscard]] std::size_t position() const noexcept { return position_; }

private:
    void require(std::size_t count) {
        if (count > limit_ - position_)
            throw std::runtime_error("checkpoint payload is truncated");
    }

    const std::vector<unsigned char>& bytes_;
    std::size_t limit_;
    std::size_t position_ = 0;
};

void append_identity(std::vector<unsigned char>& bytes,
                     const CheckpointIdentity& identity) {
    append_signed(bytes, static_cast<std::int32_t>(identity.linear_size));
    append_signed(bytes, static_cast<std::int32_t>(identity.site_count));
    append_signed(bytes, static_cast<std::int32_t>(identity.replicas));
    append_signed(bytes, static_cast<std::int32_t>(identity.sweeps));
    append_unsigned(bytes, identity.seed);
    append_unsigned(bytes, static_cast<std::uint8_t>(identity.direction));
    append_signed(bytes, static_cast<std::int32_t>(identity.detailed_cap));
    append_unsigned(bytes, static_cast<std::uint64_t>(identity.philox_state_bytes));
}

CheckpointIdentity read_identity(Reader& reader) {
    CheckpointIdentity identity{};
    identity.linear_size = reader.read_signed<std::int32_t>();
    identity.site_count = reader.read_signed<std::int32_t>();
    identity.replicas = reader.read_signed<std::int32_t>();
    identity.sweeps = reader.read_signed<std::int32_t>();
    identity.seed = reader.read_unsigned<std::uint64_t>();
    const auto direction = reader.read_unsigned<std::uint8_t>();
    if (direction > static_cast<std::uint8_t>(WalkDirection::heating))
        throw std::runtime_error("checkpoint direction is invalid");
    identity.direction = static_cast<WalkDirection>(direction);
    identity.detailed_cap = reader.read_signed<std::int32_t>();
    const std::uint64_t philox_size = reader.read_unsigned<std::uint64_t>();
    if (philox_size > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("checkpoint Philox state size exceeds size_t");
    identity.philox_state_bytes = static_cast<std::size_t>(philox_size);
    return identity;
}

bool same_identity(const CheckpointIdentity& left,
                   const CheckpointIdentity& right) noexcept {
    return left.linear_size == right.linear_size
        && left.site_count == right.site_count
        && left.replicas == right.replicas
        && left.sweeps == right.sweeps
        && left.seed == right.seed
        && left.direction == right.direction
        && left.detailed_cap == right.detailed_cap
        && left.philox_state_bytes == right.philox_state_bytes;
}

std::vector<unsigned char> encode(const CheckpointIdentity& identity,
                                  const CheckpointState& state) {
    validate_checkpoint_state(identity, state);
    std::vector<unsigned char> bytes;
    const std::size_t expected = 256 + state.spins.size()
        + state.energies.size() * sizeof(Energy)
        + state.families.size() * sizeof(FamilyId)
        + state.order.size() * sizeof(ReplicaIndex) + state.philox.size();
    bytes.reserve(expected);
    bytes.insert(bytes.end(), magic, magic + sizeof(magic));
    append_unsigned(bytes, format_version);
    append_identity(bytes, identity);
    append_signed(bytes, state.boundary);
    append_unsigned(bytes, state.completed_shells);
    for (const std::uint64_t offset : state.output_offsets)
        append_unsigned(bytes, offset);
    append_unsigned(bytes, state.resampling_rng.state);
    append_unsigned(bytes, state.resampling_rng.stream);
    append_unsigned(bytes, static_cast<std::uint64_t>(state.spins.size()));
    append_unsigned(bytes, static_cast<std::uint64_t>(state.energies.size()));
    append_unsigned(bytes, static_cast<std::uint64_t>(state.families.size()));
    append_unsigned(bytes, static_cast<std::uint64_t>(state.order.size()));
    append_unsigned(bytes, static_cast<std::uint64_t>(state.philox.size()));
    for (const Spin spin : state.spins)
        append_signed(bytes, spin);
    for (const Energy energy : state.energies)
        append_signed(bytes, energy);
    for (const FamilyId family : state.families)
        append_signed(bytes, family);
    for (const ReplicaIndex replica : state.order)
        append_signed(bytes, replica);
    bytes.insert(bytes.end(), state.philox.begin(), state.philox.end());
    append_unsigned(bytes, crc32(bytes.data(), bytes.size()));
    return bytes;
}

CheckpointState decode(const std::vector<unsigned char>& bytes,
                       const CheckpointIdentity& expected_identity) {
    if (bytes.size() < sizeof(magic) + sizeof(std::uint32_t) * 2)
        throw std::runtime_error("checkpoint file is too short");
    const std::size_t payload_limit = bytes.size() - sizeof(std::uint32_t);
    std::uint32_t stored_crc = 0;
    for (std::size_t index = 0; index < sizeof(stored_crc); ++index)
        stored_crc |= static_cast<std::uint32_t>(bytes[payload_limit + index])
                    << (8U * index);
    const std::uint32_t calculated_crc = crc32(bytes.data(), payload_limit);
    if (stored_crc != calculated_crc)
        throw std::runtime_error("checkpoint CRC32 mismatch");

    Reader reader(bytes, payload_limit);
    reader.expect(magic, sizeof(magic));
    if (reader.read_unsigned<std::uint32_t>() != format_version)
        throw std::runtime_error("checkpoint version mismatch");
    const CheckpointIdentity stored_identity = read_identity(reader);
    if (!same_identity(stored_identity, expected_identity))
        throw std::runtime_error("checkpoint run identity mismatch");

    CheckpointState state{};
    state.boundary = reader.read_signed<Energy>();
    state.completed_shells = reader.read_unsigned<std::uint64_t>();
    for (std::uint64_t& offset : state.output_offsets)
        offset = reader.read_unsigned<std::uint64_t>();
    state.resampling_rng.state = reader.read_unsigned<std::uint64_t>();
    state.resampling_rng.stream = reader.read_unsigned<std::uint64_t>();
    const std::uint64_t spin_count = reader.read_unsigned<std::uint64_t>();
    const std::uint64_t energy_count = reader.read_unsigned<std::uint64_t>();
    const std::uint64_t family_count = reader.read_unsigned<std::uint64_t>();
    const std::uint64_t order_count = reader.read_unsigned<std::uint64_t>();
    const std::uint64_t philox_count = reader.read_unsigned<std::uint64_t>();
    const std::uint64_t expected_replicas
        = static_cast<std::uint64_t>(expected_identity.replicas);
    const std::uint64_t expected_sites
        = static_cast<std::uint64_t>(expected_identity.site_count);
    if (expected_sites > std::numeric_limits<std::uint64_t>::max()
                             / expected_replicas
        || spin_count != expected_sites * expected_replicas
        || energy_count != expected_replicas
        || family_count != expected_replicas
        || order_count != expected_replicas
        || expected_identity.philox_state_bytes
               > std::numeric_limits<std::uint64_t>::max() / expected_replicas
        || philox_count
               != static_cast<std::uint64_t>(expected_identity.philox_state_bytes)
                    * expected_replicas)
        throw std::runtime_error("checkpoint serialized counts do not match identity");
    const auto checked_size = [](std::uint64_t value, const char* name) {
        if (value > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error(std::string("checkpoint ") + name
                                     + " count exceeds size_t");
        return static_cast<std::size_t>(value);
    };
    state.spins.resize(checked_size(spin_count, "spin"));
    state.energies.resize(checked_size(energy_count, "energy"));
    state.families.resize(checked_size(family_count, "family"));
    state.order.resize(checked_size(order_count, "order"));
    for (Spin& spin : state.spins) spin = reader.read_signed<Spin>();
    for (Energy& energy : state.energies) energy = reader.read_signed<Energy>();
    for (FamilyId& family : state.families)
        family = reader.read_signed<FamilyId>();
    for (ReplicaIndex& replica : state.order)
        replica = reader.read_signed<ReplicaIndex>();
    state.philox = reader.read_bytes(checked_size(philox_count, "Philox"));
    if (reader.position() != payload_limit)
        throw std::runtime_error("checkpoint has trailing payload bytes");
    validate_checkpoint_state(expected_identity, state);
    return state;
}

std::vector<unsigned char> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open checkpoint " + path.string());
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) throw std::runtime_error("cannot size checkpoint " + path.string());
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty())
        input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) throw std::runtime_error("cannot read checkpoint " + path.string());
    return bytes;
}

void write_synced(const std::filesystem::path& path,
                  const unsigned char* data, std::size_t size) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (descriptor < 0) throw_system("open " + path.string());
    std::size_t written = 0;
    while (written < size) {
        const ssize_t count = ::write(descriptor, data + written, size - written);
        if (count < 0) {
            const int saved_errno = errno;
            ::close(descriptor);
            errno = saved_errno;
            throw_system("write " + path.string());
        }
        written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0) {
        const int saved_errno = errno;
        ::close(descriptor);
        errno = saved_errno;
        throw_system("fsync " + path.string());
    }
    if (::close(descriptor) != 0) throw_system("close " + path.string());
}

void sync_directory(const std::filesystem::path& path) {
    const std::filesystem::path directory = path.parent_path();
    const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) throw_system("open checkpoint directory");
    if (::fsync(descriptor) != 0) {
        const int saved_errno = errno;
        ::close(descriptor);
        errno = saved_errno;
        throw_system("fsync checkpoint directory");
    }
    if (::close(descriptor) != 0) throw_system("close checkpoint directory");
}

Energy recompute_energy(const Spin* spins, int L) {
    Energy energy = 0;
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = x + y * L;
            const int right = (x + 1 == L ? 0 : x + 1) + y * L;
            const int down = x + (y + 1 == L ? 0 : y + 1) * L;
            energy -= static_cast<Energy>(spins[site])
                    * static_cast<Energy>(spins[right] + spins[down]);
        }
    }
    return energy;
}

const char* source_name(CheckpointSource source) noexcept {
    switch (source) {
        case CheckpointSource::current: return "current";
        case CheckpointSource::previous: return "previous";
        case CheckpointSource::temporary: return "temporary";
        case CheckpointSource::none: return "none";
    }
    return "unknown";
}

} // namespace

CheckpointFiles make_checkpoint_files(const std::filesystem::path& directory,
                                      const std::string& basename) {
    return CheckpointFiles{
        directory / (basename + ".bin"),
        directory / (basename + ".prev.bin"),
        directory / (basename + ".tmp"),
        directory / (basename + ".done"),
    };
}

void validate_checkpoint_state(const CheckpointIdentity& identity,
                               const CheckpointState& state) {
    if (identity.linear_size < 3 || identity.replicas <= 0
        || identity.sweeps <= 0 || identity.detailed_cap < 0
        || identity.philox_state_bytes == 0)
        throw std::invalid_argument("checkpoint identity has invalid dimensions");
    const std::int64_t sites = static_cast<std::int64_t>(identity.linear_size)
                             * identity.linear_size;
    if (sites != identity.site_count || sites > std::numeric_limits<int>::max())
        throw std::invalid_argument("checkpoint identity has inconsistent L and N");
    const std::size_t replicas = static_cast<std::size_t>(identity.replicas);
    const std::size_t site_count = static_cast<std::size_t>(identity.site_count);
    if (site_count > std::numeric_limits<std::size_t>::max() / replicas
        || state.spins.size() != site_count * replicas
        || state.energies.size() != replicas
        || state.families.size() != replicas
        || state.order.size() != replicas
        || identity.philox_state_bytes
               > std::numeric_limits<std::size_t>::max() / replicas
        || state.philox.size() != identity.philox_state_bytes * replicas)
        throw std::invalid_argument("checkpoint payload array sizes do not match identity");
    if ((state.resampling_rng.stream & 1ULL) == 0)
        throw std::invalid_argument("checkpoint PCG stream must be odd");
    if (state.completed_shells == 0)
        throw std::invalid_argument("checkpoint must follow at least one completed shell");
    const Energy minimum = -2 * static_cast<Energy>(identity.site_count);
    Energy maximum = 2 * static_cast<Energy>(identity.site_count);
    if ((identity.linear_size & 1) != 0)
        maximum -= 4 * static_cast<Energy>(identity.linear_size);
    if (state.boundary < minimum || state.boundary > maximum)
        throw std::invalid_argument("checkpoint boundary is outside the exact spectrum");
    if (state.completed_shells
        > static_cast<std::uint64_t>(maximum - minimum + 2))
        throw std::invalid_argument("checkpoint completed-shell count is impossible");
    for (const std::uint64_t offset : state.output_offsets) {
        if (offset == 0)
            throw std::invalid_argument("checkpoint output offset must include its header");
    }

    std::vector<unsigned char> seen(replicas, 0);
    for (std::size_t replica = 0; replica < replicas; ++replica) {
        if (state.families[replica] < 0
            || state.families[replica] >= identity.replicas)
            throw std::invalid_argument("checkpoint family id is outside [0,R)");
        const ReplicaIndex ordered = state.order[replica];
        if (ordered < 0 || ordered >= identity.replicas || seen[ordered] != 0)
            throw std::invalid_argument("checkpoint order is not a permutation of [0,R)");
        seen[ordered] = 1;
        const Spin* replica_spins = state.spins.data() + replica * site_count;
        for (std::size_t site = 0; site < site_count; ++site) {
            if (!valid_spin(replica_spins[site]))
                throw std::invalid_argument("checkpoint spin is outside {-1,+1}");
        }
        if (recompute_energy(replica_spins, identity.linear_size)
            != state.energies[replica])
            throw std::invalid_argument("checkpoint tracked energy fails full recomputation");
    }
}

void save_checkpoint(const CheckpointFiles& files,
                     const CheckpointIdentity& identity,
                     const CheckpointState& state) {
    std::error_code error;
    std::filesystem::create_directories(files.current.parent_path(), error);
    if (error) throw std::runtime_error("cannot create checkpoint directory: "
                                        + error.message());
    const std::vector<unsigned char> bytes = encode(identity, state);
    write_synced(files.temporary, bytes.data(), bytes.size());
    if (std::filesystem::exists(files.current)) {
        if (::rename(files.current.c_str(), files.previous.c_str()) != 0)
            throw_system("rotate checkpoint current to previous");
    }
    if (::rename(files.temporary.c_str(), files.current.c_str()) != 0)
        throw_system("promote checkpoint temporary to current");
    sync_directory(files.current);
}

CheckpointLoadResult load_checkpoint(const CheckpointFiles& files,
                                     const CheckpointIdentity& identity) {
    struct Candidate {
        CheckpointSource source;
        const std::filesystem::path* path;
        int tie_priority;
    };
    const Candidate candidates[] = {
        {CheckpointSource::current, &files.current, 3},
        {CheckpointSource::temporary, &files.temporary, 2},
        {CheckpointSource::previous, &files.previous, 1},
    };
    CheckpointLoadResult result;
    int best_priority = -1;
    for (const Candidate& candidate : candidates) {
        if (!std::filesystem::exists(*candidate.path)) continue;
        result.had_candidates = true;
        try {
            CheckpointState state = decode(read_file(*candidate.path), identity);
            if (!result.found
                || state.completed_shells > result.state.completed_shells
                || (state.completed_shells == result.state.completed_shells
                    && candidate.tie_priority > best_priority)) {
                result.found = true;
                result.source = candidate.source;
                result.state = std::move(state);
                best_priority = candidate.tie_priority;
            }
        } catch (const std::exception& error) {
            if (!result.diagnostics.empty()) result.diagnostics += "; ";
            result.diagnostics += std::string(source_name(candidate.source))
                                + ": " + error.what();
        }
    }
    return result;
}

bool checkpoint_is_done(const CheckpointFiles& files,
                        const CheckpointIdentity& identity) {
    std::ifstream marker(files.done, std::ios::binary);
    if (!marker) return false;
    std::string contents((std::istreambuf_iterator<char>(marker)),
                         std::istreambuf_iterator<char>());
    if (contents != done_contents(identity))
        throw std::runtime_error("done marker run identity mismatch");
    return true;
}

void mark_checkpoint_done(const CheckpointFiles& files,
                          const CheckpointIdentity& identity) {
    std::error_code error;
    std::filesystem::create_directories(files.done.parent_path(), error);
    if (error) throw std::runtime_error("cannot create checkpoint directory: "
                                        + error.message());
    const std::filesystem::path temporary_done
        = std::filesystem::path(files.done.string() + ".tmp");
    const std::string contents = done_contents(identity);
    write_synced(temporary_done,
                 reinterpret_cast<const unsigned char*>(contents.data()),
                 contents.size());
    if (::rename(temporary_done.c_str(), files.done.c_str()) != 0)
        throw_system("promote done marker");
    sync_directory(files.done);
}

} // namespace mcpa::ising2d
