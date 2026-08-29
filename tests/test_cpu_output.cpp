#include "test_harness.h"

#include "potts_output.h"

#include <cmath>
#include <cstdio>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> lines(FILE* file) {
    POTTS_REQUIRE(std::fflush(file) == 0);
    POTTS_REQUIRE(std::fseek(file, 0, SEEK_SET) == 0);
    std::vector<std::string> result;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), file)) {
        std::string line(buffer);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        result.push_back(line);
    }
    return result;
}

std::vector<std::string> fields(const std::string& line) {
    std::istringstream input(line);
    return {std::istream_iterator<std::string>(input), {}};
}

PottsShellStatistics shell_stats() {
    PottsShellStatistics shell;
    shell.replica_indices = {1, 3};
    shell.pre_resampling_families = {1, 0};
    shell.replicas = {{2, 0.25}, {4, 1.0}};
    shell.pre_resampling_family_statistics = potts_family_statistics(
        shell.pre_resampling_families.data(), 2, 4);
    return shell;
}

} // namespace

POTTS_TEST_CASE("Potts three-file schemas are rectangular and GPU metadata appears once") {
    FILE* main_file = std::tmpfile();
    FILE* aggregate_file = std::tmpfile();
    FILE* detailed_file = std::tmpfile();
    POTTS_REQUIRE(main_file && aggregate_file && detailed_file);
    POTTS_REQUIRE(write_potts_output_headers(
        main_file, aggregate_file, detailed_file));
    const int families[] = {0, 0, 2, 3};
    const PottsFamilyStatistics family = potts_family_statistics(families, 4, 4);
    PottsGpuMetadata gpu{"RTX_Test_GPU", 8, 9, 1000, 800, 700, 12040, 12040};
    const PottsResampleResult first{POTTS_RESAMPLE_OK, 1, -2, 2, 0.5};
    const PottsResampleResult terminal{
        POTTS_RESAMPLE_TERMINAL_FULL_CULL, -2, -8, 4, 1.0};
    POTTS_REQUIRE(write_potts_main_row(main_file, first, 2, 0.125, family, &gpu));
    POTTS_REQUIRE(write_potts_main_row(main_file, terminal, 2, 0.25, family, nullptr));
    POTTS_REQUIRE(write_potts_aggregate_row(
        aggregate_file, -2, 9, 2, shell_stats()));
    POTTS_REQUIRE(write_potts_detailed_rows(
        detailed_file, -2, shell_stats(), 2));

    const auto main_lines = lines(main_file);
    const auto aggregate_lines = lines(aggregate_file);
    const auto detailed_lines = lines(detailed_file);
    POTTS_REQUIRE(main_lines.size() == 3);
    POTTS_REQUIRE(fields(main_lines[0])[0] == "E");
    POTTS_REQUIRE(fields(main_lines[0])[1] == "culling_factor");
    POTTS_REQUIRE(fields(main_lines[0])[2] == "replica_family_avg_sq");
    POTTS_REQUIRE(fields(main_lines[0]).size() == fields(main_lines[1]).size());
    POTTS_REQUIRE(fields(main_lines[0]).size() == fields(main_lines[2]).size());
    POTTS_REQUIRE(main_lines[1].find("RTX_Test_GPU") != std::string::npos);
    POTTS_REQUIRE(main_lines[2].find("RTX_Test_GPU") == std::string::npos);
    POTTS_REQUIRE(main_lines[2].find("terminal_full_cull") != std::string::npos);
    POTTS_REQUIRE(aggregate_lines.size() == 2);
    POTTS_REQUIRE(fields(aggregate_lines[0]).size()
                  == fields(aggregate_lines[1]).size());
    POTTS_REQUIRE(detailed_lines.size() == 3);
    POTTS_REQUIRE(fields(detailed_lines[0]).size()
                  == fields(detailed_lines[1]).size());
    std::fclose(main_file);
    std::fclose(aggregate_file);
    std::fclose(detailed_file);
}

POTTS_TEST_CASE("Potts detailed cap is deterministic and preserves pre-resampling families") {
    for (const auto& expected : std::vector<std::pair<int, int>>{
             {0, 0}, {1, 1}, {2, 2}, {-1, 2}}) {
        FILE* file = std::tmpfile();
        POTTS_REQUIRE(file != nullptr);
        POTTS_REQUIRE(write_potts_detailed_rows(
            file, -7, shell_stats(), expected.first));
        const auto output = lines(file);
        POTTS_REQUIRE(static_cast<int>(output.size()) == expected.second);
        for (int r = 0; r < expected.second; ++r) {
            const auto row = fields(output[r]);
            POTTS_REQUIRE(std::stoi(row[1]) == (r == 0 ? 1 : 3));
            POTTS_REQUIRE(std::stoi(row[2]) == (r == 0 ? 1 : 0));
            POTTS_REQUIRE(row[6] == "replica_index_prefix");
        }
        std::fclose(file);
    }
}

POTTS_TEST_CASE("Potts shell statistics select exactly the new energy subset") {
    constexpr int R = 4;
    constexpr int N = 4;
    const char spins[R * N] = {
        0, 0, 0, 0,
        0, 0, 1, 1,
        1, 1, 1, 1,
        0, 1, 0, 1};
    const int energies[R] = {-8, -4, -8, -4};
    const std::uint64_t flips[R] = {10, 20, 30, 40};
    const int families[R] = {3, 1, 3, 1};
    const PottsShellStatistics shell = potts_shell_statistics(
        spins, energies, flips, families, R, N, 2, -4);
    POTTS_REQUIRE(shell.replica_indices == std::vector<int>({1, 3}));
    POTTS_REQUIRE(shell.pre_resampling_families == std::vector<int>({1, 1}));
    POTTS_REQUIRE(shell.replicas.size() == 2);
    POTTS_REQUIRE(shell.replicas[0].accepted_flips == 20);
    POTTS_REQUIRE(shell.replicas[1].accepted_flips == 40);
    POTTS_REQUIRE(shell.pre_resampling_family_statistics.family_count == 1);
    POTTS_REQUIRE(shell.pre_resampling_family_statistics.max_family_size == 2);
    POTTS_REQUIRE(shell.pre_resampling_family_statistics.replica_family_avg_sq == 1.0);
}

POTTS_TEST_CASE("q=2 Potts scalar order is exactly absolute Ising magnetization") {
    for (int code = 0; code < 512; ++code) {
        char spins[9];
        int magnetization = 0;
        for (int site = 0; site < 9; ++site) {
            spins[site] = static_cast<char>((code >> site) & 1);
            magnetization += spins[site] ? 1 : -1;
        }
        const double expected = std::abs(magnetization) / 9.0;
        POTTS_REQUIRE(std::abs(potts_scalar_order(spins, 9, 2) - expected)
                      < 1e-15);
    }
}
