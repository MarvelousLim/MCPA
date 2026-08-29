#include "ising1d_output.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

std::vector<IsingReplicaMeasurement> measure_ising_replicas(
    const char* spins, const int* flip_counts, int N, int R) {
    std::vector<IsingReplicaMeasurement> measurements(static_cast<std::size_t>(R));
    for (int replica = 0; replica < R; ++replica) {
        int magnetization = 0;
        const std::size_t shift = static_cast<std::size_t>(replica) * N;
        for (int site = 0; site < N; ++site)
            magnetization += spins[shift + static_cast<std::size_t>(site)];
        IsingReplicaMeasurement& measured = measurements[static_cast<std::size_t>(replica)];
        measured.flips = flip_counts[replica];
        measured.magnetization = magnetization;
        measured.absolute_magnetization = std::abs(magnetization);
        measured.magnetization_squared
            = static_cast<std::int64_t>(magnetization) * magnetization;
    }
    return measurements;
}

IsingFamilyMetrics calculate_ising_family_metrics(
    const int* families, int R, const int* energies, int selected_energy) {
    IsingFamilyMetrics result;
    std::vector<int> histogram(static_cast<std::size_t>(R), 0);
    int selected = 0;
    for (int replica = 0; replica < R; ++replica) {
        if (energies != nullptr && energies[replica] != selected_energy) continue;
        const int family = families[replica];
        if (family < 0 || family >= R) continue;
        ++histogram[static_cast<std::size_t>(family)];
        ++selected;
    }
    if (selected == 0) return result;
    double shannon = 0.0;
    double simpson = 0.0;
    std::int64_t size_squared_sum = 0;
    for (const int size : histogram) {
        if (size == 0) continue;
        ++result.count;
        result.max_size = std::max(result.max_size, size);
        const double fraction = static_cast<double>(size) / selected;
        shannon -= fraction * std::log(fraction);
        simpson += fraction * fraction;
        size_squared_sum += static_cast<std::int64_t>(size) * size;
    }
    result.max_fraction = static_cast<double>(result.max_size) / selected;
    result.shannon_entropy = shannon;
    result.effective_shannon = std::exp(shannon);
    result.simpson_concentration = simpson;
    result.replica_family_avg_sq
        = static_cast<double>(size_squared_sum) / selected;
    return result;
}

IsingShellAggregate aggregate_ising_shell(
    const std::vector<IsingReplicaMeasurement>& measurements,
    const int* energies, const int* pre_resampling_families,
    int R, int shell_energy, int N, int n_steps) {
    IsingShellAggregate result;
    for (int replica = 0; replica < R; ++replica) {
        if (energies[replica] != shell_energy) continue;
        const IsingReplicaMeasurement& measured
            = measurements[static_cast<std::size_t>(replica)];
        ++result.shell_replica_count;
        result.accepted_flip_sum += measured.flips;
        result.magnetization_sum += measured.magnetization;
        result.absolute_magnetization_sum += measured.absolute_magnetization;
        result.magnetization_squared_sum += measured.magnetization_squared;
    }
    if (result.shell_replica_count != 0) {
        const double count = result.shell_replica_count;
        result.accepted_flip_mean = result.accepted_flip_sum / count;
        result.accepted_flip_rate = result.accepted_flip_sum
            / (count * static_cast<double>(N) * n_steps);
        result.magnetization_mean = result.magnetization_sum / count;
        result.absolute_magnetization_mean
            = result.absolute_magnetization_sum / count;
        result.magnetization_squared_mean
            = result.magnetization_squared_sum / count;
    }
    result.pre_resampling_family = calculate_ising_family_metrics(
        pre_resampling_families, R, energies, shell_energy);
    return result;
}
