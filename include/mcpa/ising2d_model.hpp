#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpa::ising2d {

using Spin = std::int8_t;
using Energy = std::int64_t;

inline constexpr Spin spin_down = -1;
inline constexpr Spin spin_up = 1;

enum class WalkDirection {
    cooling,
    heating,
};

#if defined(__CUDACC__)
#define MCPA_ISING2D_HOST_DEVICE __host__ __device__
#else
#define MCPA_ISING2D_HOST_DEVICE
#endif

[[nodiscard]] MCPA_ISING2D_HOST_DEVICE constexpr bool
valid_spin(Spin spin) noexcept {
    return spin == spin_down || spin == spin_up;
}

[[nodiscard]] MCPA_ISING2D_HOST_DEVICE constexpr bool
strict_constraint(Energy candidate, Energy boundary,
                  WalkDirection direction) noexcept {
    return direction == WalkDirection::cooling
               ? candidate < boundary
               : candidate > boundary;
}

struct Neighbors {
    int left;
    int right;
    int up;
    int down;
};

// Device-safe geometry primitive. The caller supplies a valid site in
// [0, L*L) and L >= 3; the checked SquareTorus API owns host-side validation.
[[nodiscard]] MCPA_ISING2D_HOST_DEVICE constexpr Neighbors
square_torus_neighbors_unchecked(int site_index, int linear_size) noexcept {
    const int x = site_index % linear_size;
    const int y = site_index / linear_size;
    const int row = y * linear_size;
    const int up_row = (y == 0 ? linear_size - 1 : y - 1) * linear_size;
    const int down_row = (y + 1 == linear_size ? 0 : y + 1) * linear_size;
    return Neighbors{
        row + (x == 0 ? linear_size - 1 : x - 1),
        row + (x + 1 == linear_size ? 0 : x + 1),
        up_row + x,
        down_row + x,
    };
}

// Conventional nearest-neighbor Ising model on an L x L periodic square
// lattice. Bonds are counted once and H = -sum_<ij> s_i s_j.
class SquareTorus {
public:
    explicit SquareTorus(int linear_size);

    [[nodiscard]] int linear_size() const noexcept { return linear_size_; }
    [[nodiscard]] int site_count() const noexcept { return site_count_; }

    [[nodiscard]] int site(int x, int y) const noexcept;
    [[nodiscard]] Neighbors neighbors(int site_index) const;

    [[nodiscard]] bool is_valid_spin(Spin spin) const noexcept;
    void validate_spins(const std::vector<Spin>& spins) const;

    [[nodiscard]] Energy energy(const std::vector<Spin>& spins) const;
    [[nodiscard]] Energy magnetization(const std::vector<Spin>& spins) const;
    [[nodiscard]] Energy flip_delta(const std::vector<Spin>& spins,
                                    int site_index) const;

    // Exact spectrum endpoints for the periodic square lattice. Odd L is
    // frustrated at the antiferromagnetic endpoint.
    [[nodiscard]] Energy minimum_energy() const noexcept;
    [[nodiscard]] Energy maximum_energy() const noexcept;

    // Initial strict MCPA boundary outside the complete spectrum:
    // cooling accepts E < U, heating accepts E > U.
    [[nodiscard]] Energy outside_spectrum_sentinel(WalkDirection direction) const noexcept;
    [[nodiscard]] bool satisfies_strict_constraint(Energy candidate,
                                                   Energy boundary,
                                                   WalkDirection direction) const noexcept;

private:
    [[nodiscard]] int wrap(int coordinate) const noexcept;
    void validate_site(int site_index) const;

    int linear_size_;
    int site_count_;
};

} // namespace mcpa::ising2d

#undef MCPA_ISING2D_HOST_DEVICE
