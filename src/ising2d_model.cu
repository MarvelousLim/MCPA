#include "mcpa/ising2d_model.hpp"

#include <limits>
#include <stdexcept>

namespace mcpa::ising2d {

SquareTorus::SquareTorus(int linear_size)
    : linear_size_(linear_size), site_count_(0) {
    if (linear_size_ < 3) {
        throw std::invalid_argument("2D Ising square torus requires L >= 3");
    }

    const auto sites = static_cast<std::int64_t>(linear_size_) * linear_size_;
    if (sites > std::numeric_limits<int>::max()) {
        throw std::overflow_error("L*L does not fit the model site index type");
    }
    site_count_ = static_cast<int>(sites);
}

int SquareTorus::wrap(int coordinate) const noexcept {
    const int remainder = coordinate % linear_size_;
    return remainder < 0 ? remainder + linear_size_ : remainder;
}

int SquareTorus::site(int x, int y) const noexcept {
    return wrap(x) + linear_size_ * wrap(y);
}

void SquareTorus::validate_site(int site_index) const {
    if (site_index < 0 || site_index >= site_count_) {
        throw std::out_of_range("2D Ising site index is outside the lattice");
    }
}

Neighbors SquareTorus::neighbors(int site_index) const {
    validate_site(site_index);
    return square_torus_neighbors_unchecked(site_index, linear_size_);
}

bool SquareTorus::is_valid_spin(Spin spin) const noexcept {
    return valid_spin(spin);
}

void SquareTorus::validate_spins(const std::vector<Spin>& spins) const {
    if (spins.size() != static_cast<std::size_t>(site_count_)) {
        throw std::invalid_argument("2D Ising spin array must contain exactly L*L sites");
    }
    for (const Spin spin : spins) {
        if (!is_valid_spin(spin)) {
            throw std::invalid_argument("2D Ising spins must be exactly -1 or +1");
        }
    }
}

Energy SquareTorus::energy(const std::vector<Spin>& spins) const {
    validate_spins(spins);
    Energy result = 0;
    for (int y = 0; y < linear_size_; ++y) {
        for (int x = 0; x < linear_size_; ++x) {
            const int current = site(x, y);
            const int right = site(x + 1, y);
            const int down = site(x, y + 1);
            result -= static_cast<Energy>(spins[current])
                      * static_cast<Energy>(spins[right] + spins[down]);
        }
    }
    return result;
}

Energy SquareTorus::magnetization(const std::vector<Spin>& spins) const {
    validate_spins(spins);
    Energy result = 0;
    for (const Spin spin : spins) result += spin;
    return result;
}

Energy SquareTorus::flip_delta(const std::vector<Spin>& spins, int site_index) const {
    validate_spins(spins);
    const Neighbors adjacent = neighbors(site_index);
    const Energy neighbor_sum = static_cast<Energy>(spins[adjacent.left])
                                + spins[adjacent.right]
                                + spins[adjacent.up]
                                + spins[adjacent.down];
    return 2 * static_cast<Energy>(spins[site_index]) * neighbor_sum;
}

Energy SquareTorus::minimum_energy() const noexcept {
    return -2 * static_cast<Energy>(site_count_);
}

Energy SquareTorus::maximum_energy() const noexcept {
    const Energy unfrustrated = 2 * static_cast<Energy>(site_count_);
    if (linear_size_ % 2 == 0) return unfrustrated;
    return unfrustrated - 4 * static_cast<Energy>(linear_size_);
}

Energy SquareTorus::outside_spectrum_sentinel(WalkDirection direction) const noexcept {
    return direction == WalkDirection::cooling
               ? maximum_energy() + 1
               : minimum_energy() - 1;
}

bool SquareTorus::satisfies_strict_constraint(Energy candidate,
                                              Energy boundary,
                                              WalkDirection direction) const noexcept {
    return strict_constraint(candidate, boundary, direction);
}

} // namespace mcpa::ising2d
