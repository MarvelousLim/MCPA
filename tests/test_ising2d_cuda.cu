#include "test_harness.hpp"

#include "mcpa/ising2d_cuda.hpp"
#include "mcpa/ising2d_checkpoint.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

using mcpa::ising2d::DeviceEngineView;
using mcpa::ising2d::DevicePopulationView;
using mcpa::ising2d::Energy;
using mcpa::ising2d::EngineShape;
using mcpa::ising2d::FlipCount;
using mcpa::ising2d::PhiloxState;
using mcpa::ising2d::ReplicaIndex;
using mcpa::ising2d::Spin;
using mcpa::ising2d::WalkDirection;

constexpr int reference_threads = 64;

void require_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void synchronize(const char* operation) {
    require_cuda(cudaDeviceSynchronize(), operation);
}

int reference_blocks(int replicas) {
    return (replicas + reference_threads - 1) / reference_threads;
}

__device__ Energy reference_full_energy(const Spin* spins, int L) {
    const int N = L * L;
    Energy energy = 0;
    for (int site = 0; site < N; ++site) {
        const int x = site % L;
        const int y = site / L;
        const int right = ((x + 1) % L) + y * L;
        const int down = x + ((y + 1) % L) * L;
        energy -= static_cast<Energy>(spins[site])
                * static_cast<Energy>(spins[right] + spins[down]);
    }
    return energy;
}

__global__ void reference_initialize_philox(PhiloxState* states, int replicas,
                                            std::uint64_t seed) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica < replicas)
        curand_init(seed, static_cast<unsigned long long>(replica), 0,
                    &states[replica]);
}

__global__ void reference_initialize_population(DeviceEngineView device,
                                                EngineShape shape) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica >= shape.replicas) return;
    PhiloxState local = device.philox[replica];
    Spin* spins = device.population.spins
                + static_cast<std::size_t>(replica) * shape.site_count;
    for (int site = 0; site < shape.site_count; ++site) {
        const unsigned int bits = curand(&local);
        spins[site] = (bits % 2u) == 0u ? static_cast<Spin>(-1)
                                       : static_cast<Spin>(1);
    }
    device.population.energies[replica]
        = reference_full_energy(spins, shape.linear_size);
    device.population.flip_counts[replica] = 0;
    device.philox[replica] = local;
}

__global__ void reference_compute_energies(DevicePopulationView device,
                                           EngineShape shape) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica >= shape.replicas) return;
    const Spin* spins = device.spins
                      + static_cast<std::size_t>(replica) * shape.site_count;
    device.energies[replica] = reference_full_energy(spins, shape.linear_size);
}

__global__ void reference_equilibrate(DeviceEngineView device, EngineShape shape,
                                      int sweeps, Energy boundary,
                                      bool heating) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica >= shape.replicas) return;

    PhiloxState local = device.philox[replica];
    Spin* spins = device.population.spins
                + static_cast<std::size_t>(replica) * shape.site_count;
    Energy energy = device.population.energies[replica];
    FlipCount flips = 0;
    for (int attempt = 0; attempt < shape.site_count * sweeps; ++attempt) {
        const unsigned int random_word = curand(&local);
        const int site = static_cast<int>(random_word
                       % static_cast<unsigned int>(shape.site_count));
        const int x = site % shape.linear_size;
        const int y = site / shape.linear_size;
        const int left = (x == 0 ? shape.linear_size - 1 : x - 1)
                       + y * shape.linear_size;
        const int right = (x + 1 == shape.linear_size ? 0 : x + 1)
                        + y * shape.linear_size;
        const int up = x + (y == 0 ? shape.linear_size - 1 : y - 1)
                           * shape.linear_size;
        const int down = x + (y + 1 == shape.linear_size ? 0 : y + 1)
                             * shape.linear_size;
        const Energy sum = static_cast<Energy>(spins[left]) + spins[right]
                         + spins[up] + spins[down];
        const Energy candidate
            = energy + 2 * static_cast<Energy>(spins[site]) * sum;
        const bool accept = heating ? candidate > boundary
                                    : candidate < boundary;
        if (accept) {
            spins[site] = static_cast<Spin>(-spins[site]);
            energy = candidate;
            ++flips;
        }
    }
    device.population.energies[replica] = energy;
    device.population.flip_counts[replica] = flips;
    device.philox[replica] = local;
}

Energy host_reference_energy(const Spin* spins, int L) {
    Energy energy = 0;
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = x + y * L;
            const int right = (x + 1) % L + y * L;
            const int down = x + ((y + 1) % L) * L;
            energy -= static_cast<Energy>(spins[site])
                    * static_cast<Energy>(spins[right] + spins[down]);
        }
    }
    return energy;
}

class DeviceFixture {
public:
    DeviceFixture(int L, int replicas)
        : shape_(mcpa::ising2d::make_engine_shape(L, replicas)) {
        allocate(production_);
        allocate(reference_);
    }

    ~DeviceFixture() {
        release(reference_);
        release(production_);
    }

    DeviceFixture(const DeviceFixture&) = delete;
    DeviceFixture& operator=(const DeviceFixture&) = delete;

    [[nodiscard]] EngineShape shape() const noexcept { return shape_; }
    [[nodiscard]] DeviceEngineView production() const noexcept {
        return production_;
    }
    [[nodiscard]] DeviceEngineView reference() const noexcept {
        return reference_;
    }

    void initialize(std::uint64_t seed) {
        // Known fill makes bytewise comparison include any representation
        // padding not written by the CUDA toolkit's curand_init implementation.
        require_cuda(cudaMemset(production_.philox, 0xa5,
                                mcpa::ising2d::philox_bytes(shape_)),
                     "fill production Philox");
        require_cuda(cudaMemset(reference_.philox, 0xa5,
                                mcpa::ising2d::philox_bytes(shape_)),
                     "fill reference Philox");
        require_cuda(mcpa::ising2d::initialize_philox(
                         production_.philox, shape_, seed),
                     "launch production Philox initialization");
        reference_initialize_philox<<<reference_blocks(shape_.replicas),
                                      reference_threads>>>(
            reference_.philox, shape_.replicas, seed);
        require_cuda(cudaPeekAtLastError(), "launch reference Philox initialization");
        synchronize("Philox initialization");
    }

    void initialize_population() {
        require_cuda(mcpa::ising2d::initialize_population(production_, shape_),
                     "launch production population initialization");
        reference_initialize_population<<<reference_blocks(shape_.replicas),
                                          reference_threads>>>(reference_, shape_);
        require_cuda(cudaPeekAtLastError(), "launch reference population initialization");
        synchronize("population initialization");
    }

    void equilibrate(int sweeps, Energy boundary, WalkDirection direction) {
        require_cuda(mcpa::ising2d::equilibrate(
                         production_, shape_, sweeps, boundary, direction),
                     "launch production equilibrate");
        reference_equilibrate<<<reference_blocks(shape_.replicas),
                                reference_threads>>>(
            reference_, shape_, sweeps, boundary,
            direction == WalkDirection::heating);
        require_cuda(cudaPeekAtLastError(), "launch reference equilibrate");
        synchronize("equilibrate");
    }

    [[nodiscard]] std::vector<Spin> spins(DeviceEngineView device) const {
        std::vector<Spin> result(static_cast<std::size_t>(shape_.site_count)
                                 * shape_.replicas);
        require_cuda(cudaMemcpy(result.data(), device.population.spins,
                                mcpa::ising2d::spin_bytes(shape_),
                                cudaMemcpyDeviceToHost),
                     "copy spins");
        return result;
    }

    [[nodiscard]] std::vector<Energy> energies(DeviceEngineView device) const {
        std::vector<Energy> result(shape_.replicas);
        require_cuda(cudaMemcpy(result.data(), device.population.energies,
                                mcpa::ising2d::energy_bytes(shape_),
                                cudaMemcpyDeviceToHost),
                     "copy energies");
        return result;
    }

    [[nodiscard]] std::vector<FlipCount> flips(DeviceEngineView device) const {
        std::vector<FlipCount> result(shape_.replicas);
        require_cuda(cudaMemcpy(result.data(), device.population.flip_counts,
                                mcpa::ising2d::flip_count_bytes(shape_),
                                cudaMemcpyDeviceToHost),
                     "copy flip counts");
        return result;
    }

    [[nodiscard]] std::vector<unsigned char> rng(DeviceEngineView device) const {
        std::vector<unsigned char> result(mcpa::ising2d::philox_bytes(shape_));
        require_cuda(cudaMemcpy(result.data(), device.philox, result.size(),
                                cudaMemcpyDeviceToHost),
                     "copy Philox states");
        return result;
    }

    void compare_everything() const {
        ISING_REQUIRE(spins(production_) == spins(reference_));
        ISING_REQUIRE(energies(production_) == energies(reference_));
        ISING_REQUIRE(flips(production_) == flips(reference_));
        const std::vector<unsigned char> production_rng = rng(production_);
        const std::vector<unsigned char> reference_rng = rng(reference_);
        ISING_REQUIRE(production_rng.size() == reference_rng.size());
        ISING_REQUIRE(std::memcmp(production_rng.data(), reference_rng.data(),
                                  production_rng.size()) == 0);
    }

    void require_domain_and_energy() const {
        const std::vector<Spin> all_spins = spins(production_);
        const std::vector<Energy> tracked = energies(production_);
        for (int replica = 0; replica < shape_.replicas; ++replica) {
            const Spin* begin = all_spins.data()
                              + static_cast<std::size_t>(replica) * shape_.site_count;
            for (int site = 0; site < shape_.site_count; ++site)
                ISING_REQUIRE(begin[site] == -1 || begin[site] == 1);
            ISING_REQUIRE(tracked[replica]
                          == host_reference_energy(begin, shape_.linear_size));
        }
    }

private:
    void allocate(DeviceEngineView& device) {
        require_cuda(cudaMalloc(&device.population.spins,
                                mcpa::ising2d::spin_bytes(shape_)),
                     "allocate spins");
        require_cuda(cudaMalloc(&device.population.energies,
                                mcpa::ising2d::energy_bytes(shape_)),
                     "allocate energies");
        require_cuda(cudaMalloc(&device.population.flip_counts,
                                mcpa::ising2d::flip_count_bytes(shape_)),
                     "allocate flip counts");
        require_cuda(cudaMalloc(&device.philox,
                                mcpa::ising2d::philox_bytes(shape_)),
                     "allocate Philox states");
    }

    static void release(DeviceEngineView device) noexcept {
        cudaFree(device.philox);
        cudaFree(device.population.flip_counts);
        cudaFree(device.population.energies);
        cudaFree(device.population.spins);
    }

    EngineShape shape_;
    DeviceEngineView production_{};
    DeviceEngineView reference_{};
};

enum class TrajectoryKind { open, frozen, mixed };

void check_trajectory(WalkDirection direction, TrajectoryKind kind,
                      std::uint64_t seed) {
    DeviceFixture fixture(4, 37);
    fixture.initialize(seed);
    fixture.initialize_population();

    const Energy minimum = -2 * fixture.shape().site_count;
    const Energy maximum = 2 * fixture.shape().site_count;
    Energy boundary = 0;
    if (kind == TrajectoryKind::open)
        boundary = direction == WalkDirection::cooling ? maximum + 1 : minimum - 1;
    else if (kind == TrajectoryKind::frozen)
        boundary = direction == WalkDirection::cooling ? minimum : maximum;

    constexpr int sweeps = 5;
    fixture.equilibrate(sweeps, boundary, direction);
    fixture.compare_everything();
    fixture.require_domain_and_energy();

    FlipCount total_flips = 0;
    for (FlipCount count : fixture.flips(fixture.production())) total_flips += count;
    const FlipCount total_attempts
        = static_cast<FlipCount>(fixture.shape().replicas)
        * fixture.shape().site_count * sweeps;
    if (kind == TrajectoryKind::open) ISING_REQUIRE(total_flips == total_attempts);
    if (kind == TrajectoryKind::frozen) ISING_REQUIRE(total_flips == 0);
    if (kind == TrajectoryKind::mixed) {
        ISING_REQUIRE(total_flips > 0);
        ISING_REQUIRE(total_flips < total_attempts);
    }
}

} // namespace

ISING_TEST_CASE("CUDA Philox initialization is byte-exact") {
    DeviceFixture fixture(3, 137);
    fixture.initialize(0x123456789abcdef0ULL);
    ISING_REQUIRE(fixture.rng(fixture.production())
                  == fixture.rng(fixture.reference()));
}

ISING_TEST_CASE("CUDA spin initialization has exact domain energy flips and RNG") {
    for (int L : {3, 4}) {
        DeviceFixture fixture(L, 131);
        fixture.initialize(7001 + L);
        fixture.initialize_population();
        fixture.compare_everything();
        fixture.require_domain_and_energy();
        for (FlipCount count : fixture.flips(fixture.production()))
            ISING_REQUIRE(count == 0);
    }
}

ISING_TEST_CASE("CUDA full energy matches an independent reference") {
    DeviceFixture fixture(4, 137);
    fixture.initialize(8119);
    std::vector<Spin> spins(static_cast<std::size_t>(fixture.shape().site_count)
                            * fixture.shape().replicas);
    for (int replica = 0; replica < fixture.shape().replicas; ++replica) {
        for (int site = 0; site < fixture.shape().site_count; ++site) {
            const int x = site % fixture.shape().linear_size;
            const int y = site / fixture.shape().linear_size;
            spins[static_cast<std::size_t>(replica) * fixture.shape().site_count + site]
                = ((x + 2 * y + 3 * replica) % 5) < 2 ? -1 : 1;
        }
    }
    require_cuda(cudaMemcpy(fixture.production().population.spins, spins.data(),
                            mcpa::ising2d::spin_bytes(fixture.shape()),
                            cudaMemcpyHostToDevice),
                 "upload production spins");
    require_cuda(cudaMemcpy(fixture.reference().population.spins, spins.data(),
                            mcpa::ising2d::spin_bytes(fixture.shape()),
                            cudaMemcpyHostToDevice),
                 "upload reference spins");
    require_cuda(mcpa::ising2d::compute_energies(
                     fixture.production().population, fixture.shape()),
                 "launch production full energy");
    reference_compute_energies<<<reference_blocks(fixture.shape().replicas),
                                 reference_threads>>>(
        fixture.reference().population, fixture.shape());
    require_cuda(cudaPeekAtLastError(), "launch reference full energy");
    synchronize("full energy");
    ISING_REQUIRE(fixture.energies(fixture.production())
                  == fixture.energies(fixture.reference()));
    fixture.require_domain_and_energy();
    ISING_REQUIRE(fixture.rng(fixture.production())
                  == fixture.rng(fixture.reference()));
}

ISING_TEST_CASE("CUDA replica copying uses an immutable source snapshot") {
    constexpr int replicas = 137;
    const EngineShape shape = mcpa::ising2d::make_engine_shape(3, replicas);
    std::vector<Spin> source_spins(
        static_cast<std::size_t>(shape.site_count) * replicas);
    std::vector<Energy> source_energies(replicas);
    std::vector<FlipCount> source_flips(replicas);
    std::vector<ReplicaIndex> parents(replicas);
    for (int replica = 0; replica < replicas; ++replica) {
        source_energies[replica] = 1000003LL * replica - 71;
        source_flips[replica] = static_cast<FlipCount>(replica * 11 + 5);
        parents[replica] = static_cast<ReplicaIndex>(
            replica % 7 == 0 ? 1 : (replica * 53 + 17) % replicas);
        for (int site = 0; site < shape.site_count; ++site) {
            source_spins[static_cast<std::size_t>(replica) * shape.site_count + site]
                = ((replica * 7 + site * 3) & 1) == 0 ? -1 : 1;
        }
    }

    DevicePopulationView population{};
    mcpa::ising2d::DeviceReplicaCopyWorkspace workspace{};
    ReplicaIndex* device_parents = nullptr;
    require_cuda(cudaMalloc(&population.spins, mcpa::ising2d::spin_bytes(shape)),
                 "allocate copy-test spins");
    require_cuda(cudaMalloc(&population.energies, mcpa::ising2d::energy_bytes(shape)),
                 "allocate copy-test energies");
    require_cuda(cudaMalloc(&population.flip_counts,
                            mcpa::ising2d::flip_count_bytes(shape)),
                 "allocate copy-test flips");
    require_cuda(cudaMalloc(&workspace.spins,
                            mcpa::ising2d::replica_copy_spin_bytes(shape)),
                 "allocate copy scratch spins");
    require_cuda(cudaMalloc(&workspace.energies,
                            mcpa::ising2d::replica_copy_energy_bytes(shape)),
                 "allocate copy scratch energies");
    require_cuda(cudaMalloc(&device_parents,
                            static_cast<std::size_t>(replicas)
                                * sizeof(ReplicaIndex)),
                 "allocate parent indices");
    require_cuda(cudaMemcpy(population.spins, source_spins.data(),
                            mcpa::ising2d::spin_bytes(shape),
                            cudaMemcpyHostToDevice),
                 "upload copy-test spins");
    require_cuda(cudaMemcpy(population.energies, source_energies.data(),
                            mcpa::ising2d::energy_bytes(shape),
                            cudaMemcpyHostToDevice),
                 "upload copy-test energies");
    require_cuda(cudaMemcpy(population.flip_counts, source_flips.data(),
                            mcpa::ising2d::flip_count_bytes(shape),
                            cudaMemcpyHostToDevice),
                 "upload copy-test flips");
    require_cuda(cudaMemcpy(device_parents, parents.data(),
                            static_cast<std::size_t>(replicas)
                                * sizeof(ReplicaIndex),
                            cudaMemcpyHostToDevice),
                 "upload parent indices");
    require_cuda(mcpa::ising2d::copy_replicas(
                     population, device_parents, workspace, shape),
                 "launch safe replica copy");
    synchronize("safe replica copy");

    std::vector<Spin> copied_spins(source_spins.size());
    std::vector<Energy> copied_energies(replicas);
    std::vector<FlipCount> copied_flips(replicas);
    require_cuda(cudaMemcpy(copied_spins.data(), population.spins,
                            mcpa::ising2d::spin_bytes(shape),
                            cudaMemcpyDeviceToHost),
                 "download copied spins");
    require_cuda(cudaMemcpy(copied_energies.data(), population.energies,
                            mcpa::ising2d::energy_bytes(shape),
                            cudaMemcpyDeviceToHost),
                 "download copied energies");
    require_cuda(cudaMemcpy(copied_flips.data(), population.flip_counts,
                            mcpa::ising2d::flip_count_bytes(shape),
                            cudaMemcpyDeviceToHost),
                 "download untouched flips");
    for (int destination = 0; destination < replicas; ++destination) {
        const int source = parents[destination];
        ISING_REQUIRE(copied_energies[destination] == source_energies[source]);
        for (int site = 0; site < shape.site_count; ++site) {
            ISING_REQUIRE(
                copied_spins[static_cast<std::size_t>(destination)
                                 * shape.site_count + site]
                == source_spins[static_cast<std::size_t>(source)
                                    * shape.site_count + site]);
        }
    }
    ISING_REQUIRE(copied_flips == source_flips);

    cudaFree(device_parents);
    cudaFree(workspace.energies);
    cudaFree(workspace.spins);
    cudaFree(population.flip_counts);
    cudaFree(population.energies);
    cudaFree(population.spins);
}

ISING_TEST_CASE("Checkpoint restores bitwise cooling and heating continuation") {
    for (const WalkDirection direction : {WalkDirection::cooling,
                                          WalkDirection::heating}) {
        DeviceFixture fixture(3, 37);
        const std::uint64_t seed
            = direction == WalkDirection::cooling ? 12001 : 12002;
        fixture.initialize(seed);
        fixture.initialize_population();

        std::array<char, 64> path_pattern{};
        const std::string pattern = "/tmp/mcpa-ising-gpu-checkpoint.XXXXXX";
        std::copy(pattern.begin(), pattern.end(), path_pattern.begin());
        char* directory_text = mkdtemp(path_pattern.data());
        ISING_REQUIRE(directory_text != nullptr);
        const std::filesystem::path directory(directory_text);
        const auto files = mcpa::ising2d::make_checkpoint_files(
            directory, direction == WalkDirection::cooling ? "cool" : "heat");

        mcpa::ising2d::CheckpointState state{};
        state.boundary = 0;
        state.completed_shells = 1;
        state.output_offsets = {{11, 22, 33}};
        state.resampling_rng = mcpa::ising2d::seed_resampling_rng(seed);
        state.spins = fixture.spins(fixture.production());
        state.energies = fixture.energies(fixture.production());
        state.families.resize(fixture.shape().replicas);
        state.order.resize(fixture.shape().replicas);
        for (int replica = 0; replica < fixture.shape().replicas; ++replica) {
            state.families[replica] = replica;
            state.order[replica] = replica;
        }
        state.philox = fixture.rng(fixture.production());
        const mcpa::ising2d::CheckpointIdentity identity{
            fixture.shape().linear_size, fixture.shape().site_count,
            fixture.shape().replicas, 3, seed, direction, 5,
            sizeof(PhiloxState)};
        mcpa::ising2d::save_checkpoint(files, identity, state);
        const auto loaded = mcpa::ising2d::load_checkpoint(files, identity);
        ISING_REQUIRE(loaded.found);

        require_cuda(cudaMemcpy(fixture.reference().population.spins,
                                loaded.state.spins.data(),
                                mcpa::ising2d::spin_bytes(fixture.shape()),
                                cudaMemcpyHostToDevice),
                     "restore continuation spins");
        require_cuda(cudaMemcpy(fixture.reference().population.energies,
                                loaded.state.energies.data(),
                                mcpa::ising2d::energy_bytes(fixture.shape()),
                                cudaMemcpyHostToDevice),
                     "restore continuation energies");
        require_cuda(cudaMemcpy(fixture.reference().philox,
                                loaded.state.philox.data(),
                                mcpa::ising2d::philox_bytes(fixture.shape()),
                                cudaMemcpyHostToDevice),
                     "restore continuation Philox");
        require_cuda(mcpa::ising2d::equilibrate(
                         fixture.production(), fixture.shape(), 3, 0, direction),
                     "continue baseline after checkpoint");
        require_cuda(mcpa::ising2d::equilibrate(
                         fixture.reference(), fixture.shape(), 3, 0, direction),
                     "continue restored checkpoint");
        synchronize("checkpoint continuation");
        fixture.compare_everything();
        fixture.require_domain_and_energy();

        std::vector<Energy> baseline_energy
            = fixture.energies(fixture.production());
        std::vector<Energy> restored_energy
            = fixture.energies(fixture.reference());
        std::vector<ReplicaIndex> baseline_order(fixture.shape().replicas);
        std::vector<ReplicaIndex> restored_order(fixture.shape().replicas);
        std::vector<ReplicaIndex> baseline_parents(fixture.shape().replicas);
        std::vector<ReplicaIndex> restored_parents(fixture.shape().replicas);
        std::vector<mcpa::ising2d::FamilyId> baseline_families
            = loaded.state.families;
        std::vector<mcpa::ising2d::FamilyId> restored_families
            = loaded.state.families;
        auto baseline_pcg = state.resampling_rng;
        auto restored_pcg = loaded.state.resampling_rng;
        const auto baseline_result = mcpa::ising2d::resample_strict(
            {baseline_energy.data(), baseline_order.data(), baseline_parents.data(),
             baseline_families.data(), fixture.shape().replicas},
            0, direction, baseline_pcg);
        const auto restored_result = mcpa::ising2d::resample_strict(
            {restored_energy.data(), restored_order.data(), restored_parents.data(),
             restored_families.data(), fixture.shape().replicas},
            0, direction, restored_pcg);
        ISING_REQUIRE(baseline_result.status == restored_result.status);
        ISING_REQUIRE(baseline_result.boundary == restored_result.boundary);
        ISING_REQUIRE(baseline_result.culled == restored_result.culled);
        ISING_REQUIRE(baseline_order == restored_order);
        ISING_REQUIRE(baseline_parents == restored_parents);
        ISING_REQUIRE(baseline_families == restored_families);
        ISING_REQUIRE(baseline_pcg.state == restored_pcg.state);
        ISING_REQUIRE(baseline_pcg.stream == restored_pcg.stream);

        std::error_code cleanup_error;
        std::filesystem::remove_all(directory, cleanup_error);
        ISING_REQUIRE(!cleanup_error);
    }
}

ISING_TEST_CASE("Cooling open trajectory matches the independent reference") {
    check_trajectory(WalkDirection::cooling, TrajectoryKind::open, 9101);
}

ISING_TEST_CASE("Cooling frozen trajectory matches the independent reference") {
    check_trajectory(WalkDirection::cooling, TrajectoryKind::frozen, 9102);
}

ISING_TEST_CASE("Cooling mixed trajectory matches the independent reference") {
    check_trajectory(WalkDirection::cooling, TrajectoryKind::mixed, 9103);
}

ISING_TEST_CASE("Heating open trajectory matches the independent reference") {
    check_trajectory(WalkDirection::heating, TrajectoryKind::open, 9201);
}

ISING_TEST_CASE("Heating frozen trajectory matches the independent reference") {
    check_trajectory(WalkDirection::heating, TrajectoryKind::frozen, 9202);
}

ISING_TEST_CASE("Heating mixed trajectory matches the independent reference") {
    check_trajectory(WalkDirection::heating, TrajectoryKind::mixed, 9203);
}
