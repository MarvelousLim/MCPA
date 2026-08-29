#include "potts_output.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

double potts_scalar_order(const char* spins, int N, int q) {
    if (!spins || N <= 0 || !potts_supported_q(q))
        return std::numeric_limits<double>::quiet_NaN();
    int counts[4] = {0, 0, 0, 0};
    for (int site = 0; site < N; ++site) {
        const int color = static_cast<unsigned char>(spins[site]);
        if (color >= q) return std::numeric_limits<double>::quiet_NaN();
        ++counts[color];
    }
    const int maximum = *std::max_element(counts, counts + q);
    return (static_cast<double>(q) * maximum / N - 1.0) / (q - 1.0);
}

PottsFamilyStatistics potts_family_statistics(
        const int* families, int count, int family_domain) {
    PottsFamilyStatistics stats;
    if (!families || count <= 0 || family_domain <= 0) return stats;
    std::vector<int> histogram(static_cast<std::size_t>(family_domain), 0);
    for (int r = 0; r < count; ++r) {
        if (families[r] >= 0 && families[r] < family_domain)
            ++histogram[families[r]];
    }
    std::uint64_t square_sum = 0;
    for (int count : histogram) {
        if (count == 0) continue;
        ++stats.family_count;
        stats.max_family_size = std::max(stats.max_family_size, count);
        square_sum += static_cast<std::uint64_t>(count)
                    * static_cast<std::uint64_t>(count);
    }
    stats.max_family_fraction = static_cast<double>(stats.max_family_size) / count;
    stats.replica_family_avg_sq = static_cast<double>(square_sum)
                                / (static_cast<double>(count) * count);
    stats.simpson_effective_families = square_sum == 0
        ? 0.0 : static_cast<double>(count) * count / square_sum;
    return stats;
}

PottsShellStatistics potts_shell_statistics(
        const char* spins, const int* energies,
        const std::uint64_t* accepted_flips,
        const int* pre_resampling_families,
        int R, int N, int q, int shell) {
    PottsShellStatistics result;
    if (!spins || !energies || !pre_resampling_families
            || R <= 0 || N <= 0 || !potts_supported_q(q)) return result;
    for (int r = 0; r < R; ++r) {
        if (energies[r] != shell) continue;
        result.replica_indices.push_back(r);
        result.pre_resampling_families.push_back(pre_resampling_families[r]);
        result.replicas.push_back(PottsReplicaStatistics{
            accepted_flips ? accepted_flips[r] : 0,
            potts_scalar_order(spins + static_cast<std::size_t>(r) * N, N, q)});
    }
    result.pre_resampling_family_statistics = potts_family_statistics(
        result.pre_resampling_families.data(),
        static_cast<int>(result.pre_resampling_families.size()), R);
    return result;
}

const char* potts_resample_status_name(PottsResampleStatus status) {
    switch (status) {
        case POTTS_RESAMPLE_OK: return "ok";
        case POTTS_RESAMPLE_NO_NEXT_SHELL: return "no_next_shell";
        case POTTS_RESAMPLE_TERMINAL_FULL_CULL: return "terminal_full_cull";
    }
    return "invalid";
}

bool write_potts_output_headers(FILE* main_file, FILE* aggregate_file,
                                FILE* detailed_file) {
    if (!main_file || !aggregate_file || !detailed_file) return false;
    const int main_result = std::fprintf(main_file,
        "E culling_factor replica_family_avg_sq nCull "
        "culling_factor_full_precision status nSteps equilibrate_seconds "
        "population_family_count population_family_max_size "
        "population_family_max_fraction population_family_simpson_effective "
        "gpu_name gpu_compute_capability "
        "gpu_total_memory_bytes gpu_free_memory_before_setup_bytes "
        "gpu_free_memory_after_setup_bytes cuda_driver_version "
        "cuda_runtime_version output_version\n");
    const int aggregate_result = std::fprintf(aggregate_file,
        "E population accepted_flip_sum accepted_flip_mean accepted_flip_rate "
        "m_mean m2_mean m3_mean m4_mean shell_pre_family_count "
        "shell_pre_family_max_size shell_pre_family_max_fraction "
        "shell_pre_replica_family_avg_sq shell_pre_family_simpson_effective "
        "output_version\n");
    const int detailed_result = std::fprintf(detailed_file,
        "E replica pre_resampling_family accepted_flip_count m "
        "detailed_cap selection_policy output_version\n");
    return main_result > 0 && aggregate_result > 0 && detailed_result > 0;
}

bool write_potts_main_row(FILE* file, const PottsResampleResult& resample,
                          int n_steps, double equilibrate_seconds,
                          const PottsFamilyStatistics& genealogy,
                          const PottsGpuMetadata* gpu) {
    if (!file) return false;
    char capability[32] = "NA";
    if (gpu) std::snprintf(capability, sizeof(capability), "%d.%d",
                           gpu->compute_major, gpu->compute_minor);
    const int result = std::fprintf(file,
        "%d %.6f %.17g %d %.17g %s %d %.17g %d %d %.17g %.17g "
        "%s %s %s %s %s %s %s %d\n",
        resample.new_U, resample.culling_fraction,
        genealogy.replica_family_avg_sq, resample.n_cull,
        resample.culling_fraction, potts_resample_status_name(resample.status),
        n_steps, equilibrate_seconds, genealogy.family_count,
        genealogy.max_family_size, genealogy.max_family_fraction,
        genealogy.simpson_effective_families,
        gpu ? gpu->name.c_str() : "NA", gpu ? capability : "NA",
        gpu ? std::to_string(gpu->total_memory_bytes).c_str() : "NA",
        gpu ? std::to_string(gpu->free_memory_before_setup_bytes).c_str() : "NA",
        gpu ? std::to_string(gpu->free_memory_after_setup_bytes).c_str() : "NA",
        gpu ? std::to_string(gpu->driver_version).c_str() : "NA",
        gpu ? std::to_string(gpu->runtime_version).c_str() : "NA",
        kPottsOutputVersion);
    return result > 0;
}

bool write_potts_aggregate_row(
        FILE* file, int shell, int N, int n_steps,
        const PottsShellStatistics& statistics) {
    const std::size_t population = statistics.replicas.size();
    if (!file || population == 0
            || statistics.replica_indices.size() != population
            || statistics.pre_resampling_families.size() != population)
        return false;
    std::uint64_t flips = 0;
    long double moments[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    for (const PottsReplicaStatistics& replica : statistics.replicas) {
        flips += replica.accepted_flips;
        long double power = replica.scalar_order;
        for (long double& moment : moments) {
            moment += power;
            power *= replica.scalar_order;
        }
    }
    const double mean_flips = static_cast<double>(flips) / population;
    const long double attempts = static_cast<long double>(population) * N * n_steps;
    const double rate = attempts == 0 ? 0.0 : static_cast<double>(flips / attempts);
    const int result = std::fprintf(file,
        "%d %d %llu %.17g %.17g %.17g %.17g %.17g %.17g "
        "%d %d %.17g %.17g %.17g %d\n",
        shell, static_cast<int>(population),
        static_cast<unsigned long long>(flips), mean_flips, rate,
        static_cast<double>(moments[0] / population),
        static_cast<double>(moments[1] / population),
        static_cast<double>(moments[2] / population),
        static_cast<double>(moments[3] / population),
        statistics.pre_resampling_family_statistics.family_count,
        statistics.pre_resampling_family_statistics.max_family_size,
        statistics.pre_resampling_family_statistics.max_family_fraction,
        statistics.pre_resampling_family_statistics.replica_family_avg_sq,
        statistics.pre_resampling_family_statistics.simpson_effective_families,
        kPottsOutputVersion);
    return result > 0;
}

bool write_potts_detailed_rows(
        FILE* file, int shell, const PottsShellStatistics& statistics,
        int detailed_cap) {
    const std::size_t population = statistics.replicas.size();
    if (!file || detailed_cap < -1
            || statistics.replica_indices.size() != population
            || statistics.pre_resampling_families.size() != population)
        return false;
    const std::size_t count = detailed_cap == -1 ? population
        : std::min(population, static_cast<std::size_t>(detailed_cap));
    for (std::size_t selected = 0; selected < count; ++selected) {
        const std::size_t replica = static_cast<std::size_t>(
            statistics.replica_indices[selected]);
        if (std::fprintf(file, "%d %zu %d %llu %.17g %d replica_index_prefix %d\n",
                         shell, replica,
                         statistics.pre_resampling_families[selected],
                         static_cast<unsigned long long>(
                             statistics.replicas[selected].accepted_flips),
                         statistics.replicas[selected].scalar_order, detailed_cap,
                         kPottsOutputVersion) <= 0) return false;
    }
    return true;
}
