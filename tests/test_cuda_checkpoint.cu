#include "test_harness.h"

#include "blumeCapel_lib.h"
#include "checkpoint.h"

#include <cuda_runtime.h>

#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void require_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

class ScopedSilence {
public:
    ScopedSilence()
        : saved_out_(dup(fileno(stdout))), saved_err_(dup(fileno(stderr))),
          sink_(std::tmpfile()) {
        BC_REQUIRE(saved_out_ >= 0);
        BC_REQUIRE(saved_err_ >= 0);
        BC_REQUIRE(sink_ != nullptr);
        std::fflush(nullptr);
        BC_REQUIRE(dup2(fileno(sink_), fileno(stdout)) >= 0);
        BC_REQUIRE(dup2(fileno(sink_), fileno(stderr)) >= 0);
    }
    ~ScopedSilence() {
        std::fflush(nullptr);
        dup2(saved_out_, fileno(stdout));
        dup2(saved_err_, fileno(stderr));
        close(saved_out_);
        close(saved_err_);
        std::fclose(sink_);
    }
private:
    int saved_out_;
    int saved_err_;
    FILE* sink_;
};

class TempDirectory {
public:
    TempDirectory() {
        std::array<char, 72> pattern{};
        std::snprintf(pattern.data(), pattern.size(),
                      "/tmp/mcpa-bc-cuda-checkpoint-XXXXXX");
        char* result = mkdtemp(pattern.data());
        BC_REQUIRE(result != nullptr);
        path_ = std::string(result) + "/nested/checkpoints";
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(
            std::filesystem::path(path_).parent_path().parent_path(), error);
    }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

Params make_params(bool heat) {
    Params params{};
    params.L = 3;
    params.N = 9;
    params.R = 32;
    params.seed = 9187;
    params.blocks = 1;
    params.threads = 32;
    params.nSteps = 2;
    params.D_num = 49;
    params.D_denum = 25;
    params.heat = heat;
    params.fullLatticeByteSize = static_cast<size_t>(params.R) * params.N
                               * sizeof(int);
    params.singleIntRowByteSize = static_cast<size_t>(params.R) * sizeof(int);
    params.replicaStatisticsByteSize = static_cast<size_t>(params.R)
                                     * sizeof(replicaStatistics);
    return params;
}

struct Snapshot {
    std::vector<int> spins;
    std::vector<int> e_j;
    std::vector<int> e_delta;
    std::vector<unsigned char> philox;
};

struct Result {
    std::vector<int> spins;
    std::vector<int> e_j;
    std::vector<int> e_delta;
    std::vector<int> families;
    std::vector<int> order;
    std::vector<replicaStatistics> stats;
    std::vector<unsigned char> philox;
    int U = 0;
    ResamplingRngState pcg{};
};

class Trajectory {
public:
    explicit Trajectory(const Params& params)
        : params_(params), spins_(static_cast<size_t>(params.R) * params.N),
          e_j_(params.R), e_delta_(params.R), order_(params.R),
          update_(params.R), families_(params.R), stats_(params.R),
          rng_(curand_states_byte_size(params)) {
        require_cuda(cudaMalloc(&device_.spin, params_.fullLatticeByteSize));
        require_cuda(cudaMalloc(&device_.e_j, params_.singleIntRowByteSize));
        require_cuda(cudaMalloc(&device_.e_delta, params_.singleIntRowByteSize));
        require_cuda(cudaMalloc(&device_.update, params_.singleIntRowByteSize));
        require_cuda(cudaMalloc(&device_.replica_statistics,
                                params_.replicaStatisticsByteSize));
        require_cuda(cudaMalloc(&states_, rng_.size()));
        host_.spin = spins_.data();
        host_.e_j = e_j_.data();
        host_.e_delta = e_delta_.data();
        host_.O = order_.data();
        host_.update = update_.data();
        host_.replica_family = families_.data();
        host_.replica_statistics = stats_.data();
        device_.O = nullptr;
        device_.replica_family = nullptr;
    }

    ~Trajectory() {
        cudaFree(states_);
        cudaFree(device_.replica_statistics);
        cudaFree(device_.update);
        cudaFree(device_.e_delta);
        cudaFree(device_.e_j);
        cudaFree(device_.spin);
    }

    Snapshot initialize_snapshot() {
        void* initialized = setup_curand_states(params_);
        initialize_population(initialized, device_, params_);
        calc_device_energy(device_, params_);
        Snapshot snapshot{
            std::vector<int>(spins_.size()), std::vector<int>(params_.R),
            std::vector<int>(params_.R), std::vector<unsigned char>(rng_.size())};
        require_cuda(cudaMemcpy(snapshot.spins.data(), device_.spin,
                                params_.fullLatticeByteSize,
                                cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(snapshot.e_j.data(), device_.e_j,
                                params_.singleIntRowByteSize,
                                cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(snapshot.e_delta.data(), device_.e_delta,
                                params_.singleIntRowByteSize,
                                cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(snapshot.philox.data(), initialized, rng_.size(),
                                cudaMemcpyDeviceToHost));
        require_cuda(cudaFree(initialized));
        return snapshot;
    }

    void reset(const Snapshot& snapshot) {
        spins_ = snapshot.spins;
        e_j_ = snapshot.e_j;
        e_delta_ = snapshot.e_delta;
        rng_ = snapshot.philox;
        initialize_update_arrays(host_, params_);
        U_ = params_.heat ? INT_MIN : INT_MAX;
        require_cuda(cudaMemcpy(device_.spin, spins_.data(),
                                params_.fullLatticeByteSize,
                                cudaMemcpyHostToDevice));
        require_cuda(cudaMemcpy(device_.e_j, e_j_.data(),
                                params_.singleIntRowByteSize,
                                cudaMemcpyHostToDevice));
        require_cuda(cudaMemcpy(device_.e_delta, e_delta_.data(),
                                params_.singleIntRowByteSize,
                                cudaMemcpyHostToDevice));
        require_cuda(cudaMemcpy(states_, rng_.data(), rng_.size(),
                                cudaMemcpyHostToDevice));
        initialize_resampling_rng(params_.seed);
    }

    void advance_one_shell() {
        double X = 1.0;
        for (int retry = 0; retry <= 10 && X >= 1.0; ++retry) {
            equilibrate(states_, device_, params_, U_);
            download_energies();
            int n_culled = 0;
            X = prepare_resample_arrays(host_, params_, &U_, &n_culled);
        }
        BC_REQUIRE(X > 0.0);
        BC_REQUIRE(X < 1.0);
        calc_replica_statistics(device_, params_, U_);
        require_cuda(cudaMemcpy(stats_.data(), device_.replica_statistics,
                                params_.replicaStatisticsByteSize,
                                cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(device_.update, update_.data(),
                                params_.singleIntRowByteSize,
                                cudaMemcpyHostToDevice));
        update_replicas(device_, params_);
        download_population();
    }

    bool save(CheckpointManager& manager, int64_t step) {
        require_cuda(cudaMemcpy(rng_.data(), states_, rng_.size(),
                                cudaMemcpyDeviceToHost));
        manager.step_count = step;
        const ResamplingRngState pcg = get_resampling_rng_state();
        const int64_t positions[3] = {17, 23, 41};
        return checkpoint_save_bc(
            manager, params_.L, params_.N, params_.R, params_.nSteps,
            params_.seed, params_.D_num, params_.D_denum, params_.heat, 100,
            spins_.data(), e_j_.data(), e_delta_.data(), families_.data(),
            order_.data(), U_, rng_.data(), rng_.size(), pcg.state, pcg.stream,
            positions);
    }

    bool load(const CheckpointManager& manager, int64_t& step) {
        uint64_t pcg_state = 0;
        uint64_t pcg_stream = 0;
        int64_t positions[3] = {};
        if (!checkpoint_load_bc(
                manager, params_.L, params_.N, params_.R, params_.nSteps,
                params_.seed, params_.D_num, params_.D_denum, params_.heat, 100,
                spins_.data(), e_j_.data(), e_delta_.data(), families_.data(),
                order_.data(), U_, step, rng_.data(), rng_.size(), pcg_state,
                pcg_stream, positions)) return false;
        BC_REQUIRE(positions[0] == 17);
        BC_REQUIRE(positions[1] == 23);
        BC_REQUIRE(positions[2] == 41);
        require_cuda(cudaMemcpy(device_.spin, spins_.data(),
                                params_.fullLatticeByteSize,
                                cudaMemcpyHostToDevice));
        require_cuda(cudaMemcpy(device_.e_j, e_j_.data(),
                                params_.singleIntRowByteSize,
                                cudaMemcpyHostToDevice));
        require_cuda(cudaMemcpy(device_.e_delta, e_delta_.data(),
                                params_.singleIntRowByteSize,
                                cudaMemcpyHostToDevice));
        require_cuda(cudaMemcpy(states_, rng_.data(), rng_.size(),
                                cudaMemcpyHostToDevice));
        set_resampling_rng_state(ResamplingRngState{pcg_state, pcg_stream});
        return true;
    }

    Result result() {
        download_population();
        require_cuda(cudaMemcpy(rng_.data(), states_, rng_.size(),
                                cudaMemcpyDeviceToHost));
        return Result{spins_, e_j_, e_delta_, families_, order_, stats_, rng_,
                      U_, get_resampling_rng_state()};
    }

private:
    void download_energies() {
        require_cuda(cudaMemcpy(e_j_.data(), device_.e_j,
                                params_.singleIntRowByteSize,
                                cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(e_delta_.data(), device_.e_delta,
                                params_.singleIntRowByteSize,
                                cudaMemcpyDeviceToHost));
    }

    void download_population() {
        require_cuda(cudaMemcpy(spins_.data(), device_.spin,
                                params_.fullLatticeByteSize,
                                cudaMemcpyDeviceToHost));
        download_energies();
    }

    Params params_;
    mainMemoryPointers host_{};
    mainMemoryPointers device_{};
    void* states_ = nullptr;
    std::vector<int> spins_;
    std::vector<int> e_j_;
    std::vector<int> e_delta_;
    std::vector<int> order_;
    std::vector<int> update_;
    std::vector<int> families_;
    std::vector<replicaStatistics> stats_;
    std::vector<unsigned char> rng_;
    int U_ = 0;
};

void require_equal(const Result& actual, const Result& expected) {
    BC_REQUIRE(actual.spins == expected.spins);
    BC_REQUIRE(actual.e_j == expected.e_j);
    BC_REQUIRE(actual.e_delta == expected.e_delta);
    BC_REQUIRE(actual.families == expected.families);
    BC_REQUIRE(actual.order == expected.order);
    BC_REQUIRE(std::memcmp(actual.stats.data(), expected.stats.data(),
                           actual.stats.size() * sizeof(replicaStatistics)) == 0);
    BC_REQUIRE(actual.philox == expected.philox);
    BC_REQUIRE(actual.U == expected.U);
    BC_REQUIRE(actual.pcg.state == expected.pcg.state);
    BC_REQUIRE(actual.pcg.stream == expected.pcg.stream);
}

void check_interrupted_continuation(bool heat) {
    ScopedSilence silence;
    const Params params = make_params(heat);
    Trajectory baseline(params);
    const Snapshot initial = baseline.initialize_snapshot();
    baseline.reset(initial);
    baseline.advance_one_shell();
    baseline.advance_one_shell();
    const Result expected = baseline.result();

    TempDirectory directory;
    CheckpointManager manager{};
    BC_REQUIRE(checkpoint_init_bc(
        manager, params.L, params.N, params.R, params.nSteps, params.seed,
        params.D_num, params.D_denum, params.heat, 100,
        directory.path().c_str(), true, 0));

    Trajectory interrupted(params);
    interrupted.reset(initial);
    interrupted.advance_one_shell();
    BC_REQUIRE(interrupted.save(manager, 1));

    Trajectory resumed(params);
    int64_t step = 0;
    BC_REQUIRE(resumed.load(manager, step));
    BC_REQUIRE(step == 1);
    resumed.advance_one_shell();
    require_equal(resumed.result(), expected);
}

} // namespace

BC_TEST_CASE("Cooling checkpoint continuation is bitwise identical") {
    check_interrupted_continuation(false);
}

BC_TEST_CASE("Heating checkpoint continuation is bitwise identical") {
    check_interrupted_continuation(true);
}
