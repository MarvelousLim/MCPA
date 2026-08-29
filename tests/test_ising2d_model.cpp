#include "test_harness.hpp"

#include "mcpa/ising2d_model.hpp"

#include <map>
#include <stdexcept>
#include <vector>

namespace {

using mcpa::ising2d::Energy;
using mcpa::ising2d::Neighbors;
using mcpa::ising2d::Spin;
using mcpa::ising2d::SquareTorus;
using mcpa::ising2d::WalkDirection;

std::vector<Spin> decode_binary_configuration(int bits, int site_count) {
    std::vector<Spin> spins(static_cast<std::size_t>(site_count));
    for (int site = 0; site < site_count; ++site) {
        spins[static_cast<std::size_t>(site)] =
            (bits & (1 << site)) != 0 ? Spin{1} : Spin{-1};
    }
    return spins;
}

Energy independent_energy(const std::vector<Spin>& spins, int linear_size) {
    Energy result = 0;
    const auto wrapped_site = [linear_size](int x, int y) {
        x = (x % linear_size + linear_size) % linear_size;
        y = (y % linear_size + linear_size) % linear_size;
        return x + linear_size * y;
    };
    for (int y = 0; y < linear_size; ++y) {
        for (int x = 0; x < linear_size; ++x) {
            const int current = wrapped_site(x, y);
            result -= static_cast<Energy>(spins[current])
                      * static_cast<Energy>(spins[wrapped_site(x + 1, y)]
                                            + spins[wrapped_site(x, y + 1)]);
        }
    }
    return result;
}

void require_observables(const SquareTorus& model,
                         const std::vector<Spin>& spins,
                         Energy expected_energy,
                         Energy expected_magnetization) {
    ISING_REQUIRE(model.energy(spins) == expected_energy);
    ISING_REQUIRE(model.energy(spins)
                  == independent_energy(spins, model.linear_size()));
    ISING_REQUIRE(model.magnetization(spins) == expected_magnetization);
}

} // namespace

ISING_TEST_CASE("Exact L=3 DOS matches all 512 states") {
    const SquareTorus model(3);
    std::map<Energy, int> dos;
    for (int bits = 0; bits < 512; ++bits) {
        const std::vector<Spin> spins = decode_binary_configuration(bits, 9);
        const Energy energy = model.energy(spins);
        ISING_REQUIRE(energy == independent_energy(spins, 3));
        ++dos[energy];
    }

    const std::map<Energy, int> expected{
        {-18, 2}, {-10, 18}, {-6, 48}, {-2, 198}, {2, 144}, {6, 102},
    };
    ISING_REQUIRE(dos == expected);

    int total = 0;
    for (const auto& [energy, degeneracy] : dos) {
        (void)energy;
        total += degeneracy;
    }
    ISING_REQUIRE(total == 512);
}

ISING_TEST_CASE("Constructed L=3 and L=4 states have exact observables") {
    const SquareTorus odd_model(3);
    require_observables(odd_model, std::vector<Spin>(9, Spin{1}), -18, 9);
    require_observables(odd_model, std::vector<Spin>(9, Spin{-1}), -18, -9);

    std::vector<Spin> one_flipped(9, Spin{1});
    one_flipped[0] = Spin{-1};
    require_observables(odd_model, one_flipped, -10, 7);

    const SquareTorus even_model(4);
    require_observables(even_model, std::vector<Spin>(16, Spin{1}), -32, 16);

    std::vector<Spin> checkerboard(16);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            checkerboard[static_cast<std::size_t>(x + 4 * y)] =
                (x + y) % 2 == 0 ? Spin{1} : Spin{-1};
        }
    }
    require_observables(even_model, checkerboard, 32, 0);

    std::vector<Spin> even_one_flipped(16, Spin{1});
    even_one_flipped[0] = Spin{-1};
    require_observables(even_model, even_one_flipped, -24, 14);
}

ISING_TEST_CASE("Local flip delta matches full energy recomputation") {
    for (const int linear_size : {3, 4}) {
        const SquareTorus model(linear_size);
        const int configuration_count = 1 << model.site_count();
        for (int bits = 0; bits < configuration_count; ++bits) {
            std::vector<Spin> spins =
                decode_binary_configuration(bits, model.site_count());
            const Energy before = model.energy(spins);
            for (int site = 0; site < model.site_count(); ++site) {
                const Energy delta = model.flip_delta(spins, site);
                spins[static_cast<std::size_t>(site)] *= Spin{-1};
                ISING_REQUIRE(model.energy(spins) - before == delta);
                spins[static_cast<std::size_t>(site)] *= Spin{-1};
            }
        }
    }
}

ISING_TEST_CASE("Periodic square-torus geometry is exact") {
    const SquareTorus l3(3);
    ISING_REQUIRE(l3.site(-1, 0) == 2);
    ISING_REQUIRE(l3.site(3, 0) == 0);
    ISING_REQUIRE(l3.site(0, -1) == 6);
    ISING_REQUIRE(l3.site(0, 3) == 0);

    const Neighbors corner3 = l3.neighbors(0);
    ISING_REQUIRE(corner3.left == 2);
    ISING_REQUIRE(corner3.right == 1);
    ISING_REQUIRE(corner3.up == 6);
    ISING_REQUIRE(corner3.down == 3);

    const SquareTorus l4(4);
    const Neighbors corner4 = l4.neighbors(15);
    ISING_REQUIRE(corner4.left == 14);
    ISING_REQUIRE(corner4.right == 12);
    ISING_REQUIRE(corner4.up == 11);
    ISING_REQUIRE(corner4.down == 3);
    ISING_REQUIRE_THROWS(std::out_of_range, (void)l4.neighbors(16));
}

ISING_TEST_CASE("Spectrum endpoints and strict sentinels are exact") {
    const SquareTorus odd_model(3);
    ISING_REQUIRE(odd_model.minimum_energy() == -18);
    ISING_REQUIRE(odd_model.maximum_energy() == 6);
    ISING_REQUIRE(odd_model.outside_spectrum_sentinel(WalkDirection::cooling) == 7);
    ISING_REQUIRE(odd_model.outside_spectrum_sentinel(WalkDirection::heating) == -19);

    const Energy ceiling = odd_model.outside_spectrum_sentinel(WalkDirection::cooling);
    const Energy floor = odd_model.outside_spectrum_sentinel(WalkDirection::heating);
    ISING_REQUIRE(odd_model.satisfies_strict_constraint(6, ceiling,
                                                       WalkDirection::cooling));
    ISING_REQUIRE(!odd_model.satisfies_strict_constraint(ceiling, ceiling,
                                                        WalkDirection::cooling));
    ISING_REQUIRE(odd_model.satisfies_strict_constraint(-18, floor,
                                                       WalkDirection::heating));
    ISING_REQUIRE(!odd_model.satisfies_strict_constraint(floor, floor,
                                                        WalkDirection::heating));
    ISING_REQUIRE(!odd_model.satisfies_strict_constraint(-10, -10,
                                                        WalkDirection::cooling));
    ISING_REQUIRE(!odd_model.satisfies_strict_constraint(-10, -10,
                                                        WalkDirection::heating));

    const SquareTorus even_model(4);
    ISING_REQUIRE(even_model.minimum_energy() == -32);
    ISING_REQUIRE(even_model.maximum_energy() == 32);
    ISING_REQUIRE(even_model.outside_spectrum_sentinel(WalkDirection::cooling) == 33);
    ISING_REQUIRE(even_model.outside_spectrum_sentinel(WalkDirection::heating) == -33);
}

ISING_TEST_CASE("Spin domain and lattice size are enforced") {
    ISING_REQUIRE_THROWS(std::invalid_argument, SquareTorus(0));
    ISING_REQUIRE_THROWS(std::invalid_argument, SquareTorus(2));

    const SquareTorus model(3);
    ISING_REQUIRE(model.is_valid_spin(Spin{-1}));
    ISING_REQUIRE(model.is_valid_spin(Spin{1}));
    ISING_REQUIRE(!model.is_valid_spin(Spin{0}));
    ISING_REQUIRE(!model.is_valid_spin(Spin{2}));

    ISING_REQUIRE_THROWS(std::invalid_argument,
                         (void)model.energy(std::vector<Spin>(8, Spin{1})));

    std::vector<Spin> invalid(9, Spin{1});
    invalid[4] = Spin{0};
    ISING_REQUIRE_THROWS(std::invalid_argument, (void)model.energy(invalid));
    ISING_REQUIRE_THROWS(std::invalid_argument, (void)model.flip_delta(invalid, 4));
}
