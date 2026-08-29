#include "test_harness.hpp"

#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

const std::string& ising_process_binary();

namespace {

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream input(line);
    for (std::string field; std::getline(input, field, '\t');)
        fields.push_back(field);
    return fields;
}

std::vector<std::vector<std::string>> read_table(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    ISING_REQUIRE(input.good());
    std::vector<std::vector<std::string>> rows;
    for (std::string line; std::getline(input, line);)
        rows.push_back(split_tabs(line));
    return rows;
}

std::filesystem::path find_suffix(const std::filesystem::path& directory,
                                  const std::string& suffix) {
    std::filesystem::path result;
    int matches = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() >= suffix.size()
            && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            result = entry.path();
            ++matches;
        }
    }
    ISING_REQUIRE(matches == 1);
    return result;
}

void check_bounded_process(bool heating) {
    std::array<char, 64> template_path{};
    const std::string pattern = "/tmp/mcpa-ising-process-test.XXXXXX";
    std::copy(pattern.begin(), pattern.end(), template_path.begin());
    char* root_text = mkdtemp(template_path.data());
    ISING_REQUIRE(root_text != nullptr);
    const std::filesystem::path root(root_text);
    const std::filesystem::path log = root / "process.log";
    const int seed = heating ? 8802 : 8801;
    const std::string command
        = "MCPA_OUTPUT_ROOT='" + root.string()
        + "' MCPA_DETAILED_CAP=3 MCPA_DETERMINISTIC_RUN_METADATA=1 '"
        + ising_process_binary()
        + "' " + std::to_string(seed) + " 3 1 32 1 "
        + (heating ? "1" : "0") + " >'" + log.string() + "' 2>&1";
    const int raw_status = std::system(command.c_str());
    ISING_REQUIRE(raw_status != -1);
    ISING_REQUIRE(WIFEXITED(raw_status));
    ISING_REQUIRE(WEXITSTATUS(raw_status) == 0);

    const std::filesystem::path output = root / "2DIsing";
    ISING_REQUIRE(std::filesystem::is_directory(output));
    int file_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(output))
        if (entry.is_regular_file()) ++file_count;
    ISING_REQUIRE(file_count == 3);
    const auto main = read_table(find_suffix(output, "_main.txt"));
    const auto aggregate = read_table(find_suffix(output, "_agg_stats.txt"));
    const auto detailed = read_table(find_suffix(output, "_detailed_stats.txt"));

    ISING_REQUIRE(main.size() >= 2);
    ISING_REQUIRE(main.size() <= 7); // header plus the six exact L=3 shells
    ISING_REQUIRE(main[0].size() == 23);
    ISING_REQUIRE(main[0][0] == "E");
    ISING_REQUIRE(main[0][3] == "nCull");
    ISING_REQUIRE(main[0][4] == "culling_factor_full_precision");
    ISING_REQUIRE(main[0][16] == "gpu_name");
    ISING_REQUIRE(main[0][18] == "gpu_total_memory_bytes");
    ISING_REQUIRE(main[0][19] == "gpu_free_memory_before_setup_bytes");
    ISING_REQUIRE(main[0][20] == "gpu_free_memory_after_setup_bytes");
    std::vector<long long> shells;
    std::vector<unsigned long long> culled;
    int terminal_rows = 0;
    for (std::size_t row = 1; row < main.size(); ++row) {
        ISING_REQUIRE(main[row].size() == 23);
        if (row == 1) {
            ISING_REQUIRE(main[row][16] == "TEST_GPU");
            ISING_REQUIRE(main[row][17] == "0.0");
            for (std::size_t column = 18; column < 23; ++column)
                ISING_REQUIRE(main[row][column] == "0");
        } else {
            for (std::size_t column = 16; column < 23; ++column)
                ISING_REQUIRE(main[row][column] == "NA");
        }
        shells.push_back(std::stoll(main[row][0]));
        culled.push_back(std::stoull(main[row][3]));
        const double fraction = std::stod(main[row][4]);
        ISING_REQUIRE(culled.back() >= 1);
        ISING_REQUIRE(culled.back() <= 32);
        ISING_REQUIRE(std::abs(fraction
                              - static_cast<double>(culled.back()) / 32.0)
                      < 1e-15);
        if (main[row][5] == "terminal_full_cull") ++terminal_rows;
        else ISING_REQUIRE(main[row][5] == "ok");
    }
    ISING_REQUIRE(terminal_rows == 1);
    ISING_REQUIRE(main.back()[5] == "terminal_full_cull");
    ISING_REQUIRE(culled.back() == 32);
    for (std::size_t index = 1; index < shells.size(); ++index) {
        if (heating) ISING_REQUIRE(shells[index] > shells[index - 1]);
        else ISING_REQUIRE(shells[index] < shells[index - 1]);
    }

    ISING_REQUIRE(aggregate.size() == main.size());
    ISING_REQUIRE(aggregate[0].size() == 13);
    for (std::size_t row = 1; row < aggregate.size(); ++row) {
        ISING_REQUIRE(aggregate[row].size() == 13);
        ISING_REQUIRE(std::stoll(aggregate[row][0]) == shells[row - 1]);
        ISING_REQUIRE(std::stoull(aggregate[row][1]) == culled[row - 1]);
        ISING_REQUIRE(std::stoull(aggregate[row][4])
                      <= std::stoull(aggregate[row][5]));
        ISING_REQUIRE(std::stold(aggregate[row][7]) >= 0.0L);
        ISING_REQUIRE(std::stold(aggregate[row][7]) <= 9.0L);
        ISING_REQUIRE(std::stold(aggregate[row][8]) >= 0.0L);
        ISING_REQUIRE(std::stold(aggregate[row][8]) <= 81.0L);
    }

    ISING_REQUIRE(!detailed.empty());
    ISING_REQUIRE(detailed[0].size() == 9);
    std::map<long long, std::vector<int>> replica_indices;
    for (std::size_t row = 1; row < detailed.size(); ++row) {
        ISING_REQUIRE(detailed[row].size() == 9);
        const long long shell = std::stoll(detailed[row][0]);
        replica_indices[shell].push_back(std::stoi(detailed[row][1]));
        ISING_REQUIRE(std::stoi(detailed[row][7]) == 3);
        ISING_REQUIRE(detailed[row][8] == "first_replica_indices_at_shell");
    }
    for (std::size_t row = 0; row < shells.size(); ++row) {
        const auto& indices = replica_indices.at(shells[row]);
        ISING_REQUIRE(indices.size()
                      == std::min<std::size_t>(3, culled[row]));
        ISING_REQUIRE(std::is_sorted(indices.begin(), indices.end()));
    }

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    ISING_REQUIRE(!cleanup_error);
}

std::vector<unsigned char> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    ISING_REQUIRE(input.good());
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(input),
                                      std::istreambuf_iterator<char>());
}

pid_t spawn_checkpoint_process(const std::filesystem::path& output_root,
                               const std::filesystem::path& checkpoint_root,
                               const std::filesystem::path& log,
                               int seed, bool heating, bool pause_after_save) {
    const pid_t child = fork();
    ISING_REQUIRE(child >= 0);
    if (child != 0) return child;

    const int log_fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log_fd < 0) _exit(120);
    if (dup2(log_fd, STDOUT_FILENO) < 0 || dup2(log_fd, STDERR_FILENO) < 0)
        _exit(121);
    close(log_fd);
    setenv("MCPA_OUTPUT_ROOT", output_root.c_str(), 1);
    setenv("MCPA_DETAILED_CAP", "3", 1);
    setenv("MCPA_DETERMINISTIC_TIMINGS", "1", 1);
    setenv("MCPA_DETERMINISTIC_RUN_METADATA", "1", 1);
    if (pause_after_save)
        setenv("MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS", "3000", 1);
    else
        unsetenv("MCPA_TEST_PAUSE_AFTER_CHECKPOINT_MS");
    const std::string seed_text = std::to_string(seed);
    execl(ising_process_binary().c_str(), ising_process_binary().c_str(),
          seed_text.c_str(), "3", "1", "32", "1", heating ? "1" : "0",
          checkpoint_root.c_str(), "1", static_cast<char*>(nullptr));
    _exit(122);
}

void require_successful_child(pid_t child) {
    int status = 0;
    ISING_REQUIRE(waitpid(child, &status, 0) == child);
    ISING_REQUIRE(WIFEXITED(status));
    ISING_REQUIRE(WEXITSTATUS(status) == 0);
}

bool has_current_checkpoint(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) return false;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.size() >= 4
            && name.compare(name.size() - 4, 4, ".bin") == 0
            && name.find(".prev.bin") == std::string::npos)
            return true;
    }
    return false;
}

void check_sigkill_restart(bool heating) {
    std::array<char, 64> template_path{};
    const std::string pattern = "/tmp/mcpa-ising-restart-test.XXXXXX";
    std::copy(pattern.begin(), pattern.end(), template_path.begin());
    char* root_text = mkdtemp(template_path.data());
    ISING_REQUIRE(root_text != nullptr);
    const std::filesystem::path root(root_text);
    const std::filesystem::path reference_output = root / "reference-output";
    const std::filesystem::path reference_checkpoint = root / "reference-checkpoint";
    const std::filesystem::path resumed_output = root / "resumed-output";
    const std::filesystem::path resumed_checkpoint = root / "resumed-checkpoint";
    const int seed = heating ? 9902 : 9901;

    require_successful_child(spawn_checkpoint_process(
        reference_output, reference_checkpoint, root / "reference.log",
        seed, heating, false));

    const pid_t interrupted = spawn_checkpoint_process(
        resumed_output, resumed_checkpoint, root / "interrupted.log",
        seed, heating, true);
    bool checkpoint_seen = false;
    for (int poll = 0; poll < 1000; ++poll) {
        if (has_current_checkpoint(resumed_checkpoint)) {
            checkpoint_seen = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ISING_REQUIRE(checkpoint_seen);
    ISING_REQUIRE(kill(interrupted, SIGKILL) == 0);
    int killed_status = 0;
    ISING_REQUIRE(waitpid(interrupted, &killed_status, 0) == interrupted);
    ISING_REQUIRE(WIFSIGNALED(killed_status));
    ISING_REQUIRE(WTERMSIG(killed_status) == SIGKILL);

    require_successful_child(spawn_checkpoint_process(
        resumed_output, resumed_checkpoint, root / "restart.log",
        seed, heating, false));

    const std::filesystem::path reference_directory
        = reference_output / "2DIsing";
    const std::filesystem::path resumed_directory = resumed_output / "2DIsing";
    const std::string suffixes[] = {
        "_main.txt", "_agg_stats.txt", "_detailed_stats.txt"};
    std::array<std::filesystem::path, 3> resumed_files{};
    std::array<std::vector<unsigned char>, 3> before_noop{};
    std::array<std::filesystem::file_time_type, 3> times_before{};
    for (std::size_t index = 0; index < 3; ++index) {
        const std::filesystem::path reference_file
            = find_suffix(reference_directory, suffixes[index]);
        resumed_files[index] = find_suffix(resumed_directory, suffixes[index]);
        const std::vector<unsigned char> reference_bytes = read_binary(reference_file);
        before_noop[index] = read_binary(resumed_files[index]);
        ISING_REQUIRE(before_noop[index] == reference_bytes);
        times_before[index] = std::filesystem::last_write_time(resumed_files[index]);
    }

    require_successful_child(spawn_checkpoint_process(
        resumed_output, resumed_checkpoint, root / "done-noop.log",
        seed, heating, false));
    for (std::size_t index = 0; index < 3; ++index) {
        ISING_REQUIRE(read_binary(resumed_files[index]) == before_noop[index]);
        ISING_REQUIRE(std::filesystem::last_write_time(resumed_files[index])
                      == times_before[index]);
    }

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    ISING_REQUIRE(!cleanup_error);
}

} // namespace

ISING_TEST_CASE("Bounded L=3 cooling process writes one complete terminal dataset") {
    check_bounded_process(false);
}

ISING_TEST_CASE("Bounded L=3 heating process writes one complete terminal dataset") {
    check_bounded_process(true);
}

ISING_TEST_CASE("SIGKILL cooling restart reproduces all outputs and done is a no-op") {
    check_sigkill_restart(false);
}

ISING_TEST_CASE("SIGKILL heating restart reproduces all outputs and done is a no-op") {
    check_sigkill_restart(true);
}
