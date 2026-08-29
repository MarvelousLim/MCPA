#include "mcpa/ising2d_cuda.hpp"

#include <limits>
#include <stdexcept>

namespace mcpa::ising2d {
namespace {

constexpr int threads_per_block = 128;

[[nodiscard]] int block_count(int replicas) noexcept {
    return (replicas + threads_per_block - 1) / threads_per_block;
}

[[nodiscard]] bool valid_shape(EngineShape shape) noexcept {
    if (shape.linear_size < 3 || shape.replicas <= 0) return false;
    const auto sites = static_cast<std::int64_t>(shape.linear_size)
                     * shape.linear_size;
    return sites == shape.site_count && sites <= std::numeric_limits<int>::max();
}

[[nodiscard]] bool valid_population(DevicePopulationView device) noexcept {
    return device.spins != nullptr && device.energies != nullptr
        && device.flip_counts != nullptr;
}

__device__ Energy full_energy(const Spin* spins, int linear_size,
                              int site_count) {
    Energy energy = 0;
    for (int site = 0; site < site_count; ++site) {
        const int x = site % linear_size;
        const int y = site / linear_size;
        const int right = y * linear_size
                        + (x + 1 == linear_size ? 0 : x + 1);
        const int down = (y + 1 == linear_size ? 0 : y + 1) * linear_size + x;
        energy -= static_cast<Energy>(spins[site])
                * static_cast<Energy>(spins[right] + spins[down]);
    }
    return energy;
}

__global__ void initialize_philox_kernel(PhiloxState* states, int replicas,
                                         std::uint64_t seed) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica < replicas)
        curand_init(seed, static_cast<unsigned long long>(replica), 0,
                    &states[replica]);
}

__global__ void initialize_population_kernel(DeviceEngineView device,
                                             EngineShape shape) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica >= shape.replicas) return;

    PhiloxState state = device.philox[replica];
    Spin* spins = device.population.spins
                + static_cast<std::size_t>(replica) * shape.site_count;
    for (int site = 0; site < shape.site_count; ++site)
        spins[site] = (curand(&state) & 1u) == 0u ? spin_down : spin_up;

    device.population.energies[replica]
        = full_energy(spins, shape.linear_size, shape.site_count);
    device.population.flip_counts[replica] = 0;
    device.philox[replica] = state;
}

__global__ void compute_energies_kernel(DevicePopulationView device,
                                        EngineShape shape) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica >= shape.replicas) return;
    const Spin* spins = device.spins
                      + static_cast<std::size_t>(replica) * shape.site_count;
    device.energies[replica]
        = full_energy(spins, shape.linear_size, shape.site_count);
}

__global__ void equilibrate_kernel(DeviceEngineView device, EngineShape shape,
                                   int attempts, Energy boundary,
                                   WalkDirection direction) {
    const int replica = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica >= shape.replicas) return;

    PhiloxState state = device.philox[replica];
    Spin* spins = device.population.spins
                + static_cast<std::size_t>(replica) * shape.site_count;
    Energy energy = device.population.energies[replica];
    FlipCount accepted = 0;

    for (int attempt = 0; attempt < attempts; ++attempt) {
        const int site = static_cast<int>(curand(&state)
                       % static_cast<unsigned int>(shape.site_count));
        const Neighbors adjacent
            = square_torus_neighbors_unchecked(site, shape.linear_size);
        const Energy neighbor_sum
            = static_cast<Energy>(spins[adjacent.left])
            + static_cast<Energy>(spins[adjacent.right])
            + static_cast<Energy>(spins[adjacent.up])
            + static_cast<Energy>(spins[adjacent.down]);
        const Energy delta = 2 * static_cast<Energy>(spins[site]) * neighbor_sum;
        const Energy candidate = energy + delta;
        if (strict_constraint(candidate, boundary, direction)) {
            spins[site] = static_cast<Spin>(-spins[site]);
            energy = candidate;
            ++accepted;
        }
    }

    device.population.energies[replica] = energy;
    device.population.flip_counts[replica] = accepted;
    device.philox[replica] = state;
}

__global__ void gather_replicas_kernel(DevicePopulationView population,
                                       const ReplicaIndex* parents,
                                       DeviceReplicaCopyWorkspace workspace,
                                       EngineShape shape) {
    const int destination = blockIdx.x * blockDim.x + threadIdx.x;
    if (destination >= shape.replicas) return;
    const ReplicaIndex source = parents[destination];
    if (source < 0 || source >= shape.replicas) return;
    const std::size_t destination_shift
        = static_cast<std::size_t>(destination) * shape.site_count;
    const std::size_t source_shift
        = static_cast<std::size_t>(source) * shape.site_count;
    for (int site = 0; site < shape.site_count; ++site)
        workspace.spins[destination_shift + site]
            = population.spins[source_shift + site];
    workspace.energies[destination] = population.energies[source];
}

__global__ void commit_replicas_kernel(DevicePopulationView population,
                                       DeviceReplicaCopyWorkspace workspace,
                                       EngineShape shape) {
    const int destination = blockIdx.x * blockDim.x + threadIdx.x;
    if (destination >= shape.replicas) return;
    const std::size_t shift
        = static_cast<std::size_t>(destination) * shape.site_count;
    for (int site = 0; site < shape.site_count; ++site)
        population.spins[shift + site] = workspace.spins[shift + site];
    population.energies[destination] = workspace.energies[destination];
}

} // namespace

EngineShape make_engine_shape(int linear_size, int replicas) {
    const SquareTorus lattice(linear_size);
    if (replicas <= 0)
        throw std::invalid_argument("2D Ising CUDA engine requires replicas > 0");
    return EngineShape{linear_size, lattice.site_count(), replicas};
}

std::size_t spin_bytes(EngineShape shape) noexcept {
    return static_cast<std::size_t>(shape.site_count) * shape.replicas
         * sizeof(Spin);
}

std::size_t energy_bytes(EngineShape shape) noexcept {
    return static_cast<std::size_t>(shape.replicas) * sizeof(Energy);
}

std::size_t flip_count_bytes(EngineShape shape) noexcept {
    return static_cast<std::size_t>(shape.replicas) * sizeof(FlipCount);
}

std::size_t philox_bytes(EngineShape shape) noexcept {
    return static_cast<std::size_t>(shape.replicas) * sizeof(PhiloxState);
}

std::size_t replica_copy_spin_bytes(EngineShape shape) noexcept {
    return spin_bytes(shape);
}

std::size_t replica_copy_energy_bytes(EngineShape shape) noexcept {
    return energy_bytes(shape);
}

cudaError_t initialize_philox(PhiloxState* states, EngineShape shape,
                              std::uint64_t seed, cudaStream_t stream) noexcept {
    if (!valid_shape(shape) || states == nullptr) return cudaErrorInvalidValue;
    initialize_philox_kernel<<<block_count(shape.replicas), threads_per_block, 0,
                               stream>>>(states, shape.replicas, seed);
    return cudaPeekAtLastError();
}

cudaError_t initialize_population(DeviceEngineView device, EngineShape shape,
                                  cudaStream_t stream) noexcept {
    if (!valid_shape(shape) || !valid_population(device.population)
        || device.philox == nullptr)
        return cudaErrorInvalidValue;
    initialize_population_kernel<<<block_count(shape.replicas), threads_per_block,
                                   0, stream>>>(device, shape);
    return cudaPeekAtLastError();
}

cudaError_t compute_energies(DevicePopulationView device, EngineShape shape,
                             cudaStream_t stream) noexcept {
    if (!valid_shape(shape) || !valid_population(device))
        return cudaErrorInvalidValue;
    compute_energies_kernel<<<block_count(shape.replicas), threads_per_block,
                              0, stream>>>(device, shape);
    return cudaPeekAtLastError();
}

cudaError_t equilibrate(DeviceEngineView device, EngineShape shape, int sweeps,
                        Energy boundary, WalkDirection direction,
                        cudaStream_t stream) noexcept {
    if (!valid_shape(shape) || !valid_population(device.population)
        || device.philox == nullptr || sweeps < 0
        || (sweeps != 0
            && shape.site_count > std::numeric_limits<int>::max() / sweeps))
        return cudaErrorInvalidValue;
    const int attempts = shape.site_count * sweeps;
    equilibrate_kernel<<<block_count(shape.replicas), threads_per_block, 0,
                         stream>>>(device, shape, attempts, boundary, direction);
    return cudaPeekAtLastError();
}

cudaError_t copy_replicas(DevicePopulationView population,
                          const ReplicaIndex* parents,
                          DeviceReplicaCopyWorkspace workspace,
                          EngineShape shape, cudaStream_t stream) noexcept {
    if (!valid_shape(shape) || !valid_population(population)
        || parents == nullptr || workspace.spins == nullptr
        || workspace.energies == nullptr
        || workspace.spins == population.spins
        || workspace.energies == population.energies)
        return cudaErrorInvalidValue;
    const int blocks = block_count(shape.replicas);
    gather_replicas_kernel<<<blocks, threads_per_block, 0, stream>>>(
        population, parents, workspace, shape);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
    commit_replicas_kernel<<<blocks, threads_per_block, 0, stream>>>(
        population, workspace, shape);
    return cudaPeekAtLastError();
}

} // namespace mcpa::ising2d
