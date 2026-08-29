#include "test_harness.hpp"

#include "mcpa/ising2d_checkpoint.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace mcpa::ising2d;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        std::array<char, 64> pattern{};
        const std::string value = "/tmp/mcpa-ising-checkpoint-test.XXXXXX";
        std::copy(value.begin(), value.end(), pattern.begin());
        char* created = mkdtemp(pattern.data());
        ISING_REQUIRE(created != nullptr);
        path_ = created;
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

CheckpointIdentity identity() {
    return CheckpointIdentity{3, 9, 4, 2, 0xfedcba9876543210ULL,
                              WalkDirection::cooling, 17, 64};
}

CheckpointState state_for(std::uint64_t completed_shells, Energy boundary) {
    const CheckpointIdentity run = identity();
    CheckpointState state{};
    state.boundary = boundary;
    state.completed_shells = completed_shells;
    state.output_offsets = {{101 + completed_shells,
                             202 + completed_shells,
                             303 + completed_shells}};
    state.resampling_rng = ResamplingRngState{
        0x1234000000000000ULL + completed_shells,
        0x9876000000000001ULL};
    state.families = {0, 1, 1, 3};
    state.order = {2, 0, 3, 1};
    state.spins.resize(static_cast<std::size_t>(run.site_count) * run.replicas);
    const SquareTorus lattice(run.linear_size);
    state.energies.resize(run.replicas);
    for (int replica = 0; replica < run.replicas; ++replica) {
        std::vector<Spin> spins(run.site_count, replica & 1 ? spin_down : spin_up);
        if (replica >= 2) spins[static_cast<std::size_t>(replica)]
            = static_cast<Spin>(-spins[static_cast<std::size_t>(replica)]);
        std::copy(spins.begin(), spins.end(),
                  state.spins.begin() + static_cast<std::ptrdiff_t>(replica * run.site_count));
        state.energies[replica] = lattice.energy(spins);
    }
    state.philox.resize(run.philox_state_bytes * run.replicas);
    for (std::size_t index = 0; index < state.philox.size(); ++index)
        state.philox[index] = static_cast<unsigned char>((index * 73 + completed_shells) & 0xffU);
    return state;
}

bool same_state(const CheckpointState& left, const CheckpointState& right) {
    return left.boundary == right.boundary
        && left.completed_shells == right.completed_shells
        && left.output_offsets == right.output_offsets
        && left.resampling_rng.state == right.resampling_rng.state
        && left.resampling_rng.stream == right.resampling_rng.stream
        && left.spins == right.spins && left.energies == right.energies
        && left.families == right.families && left.order == right.order
        && left.philox == right.philox;
}

void corrupt_middle_byte(const std::filesystem::path& path) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    ISING_REQUIRE(file.good());
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    ISING_REQUIRE(size > 32);
    const std::streamoff position = size / 2;
    file.seekg(position);
    char value = 0;
    file.read(&value, 1);
    ISING_REQUIRE(file.good());
    value ^= static_cast<char>(0x5a);
    file.seekp(position);
    file.write(&value, 1);
    file.flush();
    ISING_REQUIRE(file.good());
}

} // namespace

ISING_TEST_CASE("Checkpoint roundtrip preserves every Ising and RNG field") {
    TemporaryDirectory directory;
    const CheckpointFiles files = make_checkpoint_files(directory.path(), "roundtrip");
    const CheckpointState expected = state_for(3, -2);
    save_checkpoint(files, identity(), expected);
    const CheckpointLoadResult loaded = load_checkpoint(files, identity());
    ISING_REQUIRE(loaded.found);
    ISING_REQUIRE(loaded.had_candidates);
    ISING_REQUIRE(loaded.source == CheckpointSource::current);
    ISING_REQUIRE(same_state(loaded.state, expected));
}

ISING_TEST_CASE("Checkpoint rotation retains current and previous generations") {
    TemporaryDirectory directory;
    const CheckpointFiles files = make_checkpoint_files(directory.path(), "rotation");
    const CheckpointState first = state_for(1, 6);
    const CheckpointState second = state_for(2, 2);
    save_checkpoint(files, identity(), first);
    save_checkpoint(files, identity(), second);
    ISING_REQUIRE(std::filesystem::is_regular_file(files.current));
    ISING_REQUIRE(std::filesystem::is_regular_file(files.previous));
    const CheckpointLoadResult loaded = load_checkpoint(files, identity());
    ISING_REQUIRE(loaded.found);
    ISING_REQUIRE(loaded.source == CheckpointSource::current);
    ISING_REQUIRE(same_state(loaded.state, second));
}

ISING_TEST_CASE("Checkpoint CRC corruption falls back to the previous generation") {
    TemporaryDirectory directory;
    const CheckpointFiles files = make_checkpoint_files(directory.path(), "corrupt");
    const CheckpointState first = state_for(1, 6);
    save_checkpoint(files, identity(), first);
    save_checkpoint(files, identity(), state_for(2, 2));
    corrupt_middle_byte(files.current);
    const CheckpointLoadResult loaded = load_checkpoint(files, identity());
    ISING_REQUIRE(loaded.found);
    ISING_REQUIRE(loaded.source == CheckpointSource::previous);
    ISING_REQUIRE(same_state(loaded.state, first));
    ISING_REQUIRE(loaded.diagnostics.find("CRC32") != std::string::npos);
}

ISING_TEST_CASE("Checkpoint loader recovers a synced completed temporary generation") {
    TemporaryDirectory directory;
    const CheckpointFiles files = make_checkpoint_files(directory.path(), "temporary");
    const CheckpointState expected = state_for(4, -6);
    save_checkpoint(files, identity(), expected);
    std::filesystem::rename(files.current, files.temporary);
    const CheckpointLoadResult loaded = load_checkpoint(files, identity());
    ISING_REQUIRE(loaded.found);
    ISING_REQUIRE(loaded.source == CheckpointSource::temporary);
    ISING_REQUIRE(same_state(loaded.state, expected));
}

ISING_TEST_CASE("Checkpoint done marker is durable and idempotent") {
    TemporaryDirectory directory;
    const CheckpointFiles files = make_checkpoint_files(directory.path(), "done");
    const std::filesystem::path output = directory.path() / "protected_output.txt";
    {
        std::ofstream file(output);
        file << "must-not-change\n";
    }
    ISING_REQUIRE(!checkpoint_is_done(files, identity()));
    mark_checkpoint_done(files, identity());
    ISING_REQUIRE(checkpoint_is_done(files, identity()));
    mark_checkpoint_done(files, identity());
    ISING_REQUIRE(checkpoint_is_done(files, identity()));
    CheckpointIdentity other = identity();
    ++other.detailed_cap;
    ISING_REQUIRE_THROWS(std::runtime_error,
                         (void)checkpoint_is_done(files, other));
    std::ifstream file(output);
    std::string contents;
    std::getline(file, contents);
    ISING_REQUIRE(contents == "must-not-change");
}

ISING_TEST_CASE("Checkpoint semantic validation rejects physical corruption") {
    const CheckpointIdentity run = identity();
    CheckpointState state = state_for(2, 2);
    validate_checkpoint_state(run, state);

    CheckpointState bad_energy = state;
    ++bad_energy.energies[0];
    ISING_REQUIRE_THROWS(std::invalid_argument,
                         validate_checkpoint_state(run, bad_energy));
    CheckpointState bad_spin = state;
    bad_spin.spins[0] = 0;
    ISING_REQUIRE_THROWS(std::invalid_argument,
                         validate_checkpoint_state(run, bad_spin));
    CheckpointState bad_family = state;
    bad_family.families[0] = run.replicas;
    ISING_REQUIRE_THROWS(std::invalid_argument,
                         validate_checkpoint_state(run, bad_family));
    CheckpointState bad_order = state;
    bad_order.order[0] = bad_order.order[1];
    ISING_REQUIRE_THROWS(std::invalid_argument,
                         validate_checkpoint_state(run, bad_order));
    CheckpointState bad_pcg = state;
    bad_pcg.resampling_rng.stream &= ~1ULL;
    ISING_REQUIRE_THROWS(std::invalid_argument,
                         validate_checkpoint_state(run, bad_pcg));
}
