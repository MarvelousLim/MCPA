#pragma once

#include "potts_lib.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

constexpr int kPottsOutputVersion = 3;

struct PottsFamilyStatistics {
    int family_count = 0;
    int max_family_size = 0;
    double max_family_fraction = 0.0;
    double replica_family_avg_sq = 0.0;
    double simpson_effective_families = 0.0;
};

struct PottsReplicaStatistics {
    std::uint64_t accepted_flips = 0;
    double scalar_order = 0.0;
};

struct PottsShellStatistics {
    std::vector<int> replica_indices;
    std::vector<int> pre_resampling_families;
    std::vector<PottsReplicaStatistics> replicas;
    PottsFamilyStatistics pre_resampling_family_statistics;
};

struct PottsGpuMetadata {
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t free_memory_before_setup_bytes = 0;
    std::uint64_t free_memory_after_setup_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

double potts_scalar_order(const char* spins, int N, int q);
PottsFamilyStatistics potts_family_statistics(
    const int* families, int count, int family_domain);
PottsShellStatistics potts_shell_statistics(
    const char* spins, const int* energies, const std::uint64_t* accepted_flips,
    const int* pre_resampling_families, int R, int N, int q, int shell);

const char* potts_resample_status_name(PottsResampleStatus status);
bool write_potts_output_headers(FILE* main_file, FILE* aggregate_file,
                                FILE* detailed_file);
bool write_potts_main_row(FILE* file, const PottsResampleResult& resample,
                          int n_steps, double equilibrate_seconds,
                          const PottsFamilyStatistics& genealogy,
                          const PottsGpuMetadata* gpu_metadata);
bool write_potts_aggregate_row(
    FILE* file, int shell, int N, int n_steps,
    const PottsShellStatistics& statistics);
bool write_potts_detailed_rows(
    FILE* file, int shell, const PottsShellStatistics& statistics,
    int detailed_cap);
