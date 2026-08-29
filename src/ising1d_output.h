#pragma once

#include <cstdint>
#include <vector>

struct IsingFamilyMetrics {
    int count = 0;
    int max_size = 0;
    double max_fraction = 0.0;
    double shannon_entropy = 0.0;
    double effective_shannon = 0.0;
    double simpson_concentration = 0.0;
    double replica_family_avg_sq = 0.0;
};

struct IsingReplicaMeasurement {
    int flips = 0;
    int magnetization = 0;
    int absolute_magnetization = 0;
    std::int64_t magnetization_squared = 0;
};

struct IsingShellAggregate {
    int shell_replica_count = 0;
    std::int64_t accepted_flip_sum = 0;
    double accepted_flip_mean = 0.0;
    double accepted_flip_rate = 0.0;
    std::int64_t magnetization_sum = 0;
    double magnetization_mean = 0.0;
    std::int64_t absolute_magnetization_sum = 0;
    double absolute_magnetization_mean = 0.0;
    std::int64_t magnetization_squared_sum = 0;
    double magnetization_squared_mean = 0.0;
    IsingFamilyMetrics pre_resampling_family;
};

std::vector<IsingReplicaMeasurement> measure_ising_replicas(
    const char* spins, const int* flip_counts, int N, int R);

IsingFamilyMetrics calculate_ising_family_metrics(
    const int* families, int R, const int* energies = nullptr,
    int selected_energy = 0);

IsingShellAggregate aggregate_ising_shell(
    const std::vector<IsingReplicaMeasurement>& measurements,
    const int* energies, const int* pre_resampling_families,
    int R, int shell_energy, int N, int n_steps);
