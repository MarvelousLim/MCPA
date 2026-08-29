#pragma once

#include "mcpa/ising2d_model.hpp"
#include "mcpa/ising2d_resampling.hpp"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>

namespace mcpa::ising2d {

using PhiloxState = curandStatePhilox4_32_10_t;
using FlipCount = std::uint64_t;

// Non-owning, explicitly typed production device memory. Spins are laid out
// replica-major: spins[replica * shape.site_count + site].
struct DevicePopulationView {
    Spin* spins;
    Energy* energies;
    FlipCount* flip_counts;
};

struct DeviceEngineView {
    DevicePopulationView population;
    PhiloxState* philox;
};

// Required scratch for race-free replica copying. The workspace must not
// alias the live population and contains one complete population snapshot.
struct DeviceReplicaCopyWorkspace {
    Spin* spins;
    Energy* energies;
};

struct EngineShape {
    int linear_size;
    int site_count;
    int replicas;
};

// Checked host-side construction. Throws std::invalid_argument or
// std::overflow_error for unsupported dimensions.
[[nodiscard]] EngineShape make_engine_shape(int linear_size, int replicas);

[[nodiscard]] std::size_t spin_bytes(EngineShape shape) noexcept;
[[nodiscard]] std::size_t energy_bytes(EngineShape shape) noexcept;
[[nodiscard]] std::size_t flip_count_bytes(EngineShape shape) noexcept;
[[nodiscard]] std::size_t philox_bytes(EngineShape shape) noexcept;
[[nodiscard]] std::size_t replica_copy_spin_bytes(EngineShape shape) noexcept;
[[nodiscard]] std::size_t replica_copy_energy_bytes(EngineShape shape) noexcept;

// Asynchronous launches on stream. A successful return means the launch was
// accepted; callers synchronize when they need completion/error observation.
[[nodiscard]] cudaError_t initialize_philox(
    PhiloxState* states, EngineShape shape, std::uint64_t seed,
    cudaStream_t stream = nullptr) noexcept;

// Draws and stores one spin for every site, then writes the exact full energy
// and zero accepted flips. Requires initialized Philox states.
[[nodiscard]] cudaError_t initialize_population(
    DeviceEngineView device, EngineShape shape,
    cudaStream_t stream = nullptr) noexcept;

// Recomputes tracked H=-sum(right+down bonds); does not touch RNG or flips.
[[nodiscard]] cudaError_t compute_energies(
    DevicePopulationView device, EngineShape shape,
    cudaStream_t stream = nullptr) noexcept;

// Performs site-flip attempts under the strict directional constraint and
// writes the accepted count for this call. Requires sweeps >= 0.
[[nodiscard]] cudaError_t equilibrate(
    DeviceEngineView device, EngineShape shape, int sweeps, Energy boundary,
    WalkDirection direction, cudaStream_t stream = nullptr) noexcept;

// parents[destination] names a source in [0,R). A gather into workspace is
// completed before any live replica is overwritten. RNG state and per-call
// flip counts remain attached to destination slots and are not copied.
[[nodiscard]] cudaError_t copy_replicas(
    DevicePopulationView population, const ReplicaIndex* parents,
    DeviceReplicaCopyWorkspace workspace, EngineShape shape,
    cudaStream_t stream = nullptr) noexcept;

} // namespace mcpa::ising2d
