#pragma once

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define BC_CHECKPOINT_MAGIC "MCPABC8"
#define BC_CHECKPOINT_VERSION 8
#define BC_CHECKPOINT_SUFFIX "_chk"
#define BC_DEFAULT_CHECKPOINT_INTERVAL_SECS 900
#define BC_CHECKPOINT_PATH_CAPACITY 1100

#pragma pack(push, 1)
struct BcCheckpointHeader {
    char magic[8];
    int32_t version;
    int32_t L, N, R;
    int32_t nSteps, seed;
    int32_t D_num, D_denum;
    int32_t heat;
    int32_t detail_cap;
    int32_t U;
    int64_t step_count;
    int64_t timestamp;
    int64_t spin_words;
    int64_t rng_state_bytes;
    int64_t payload_bytes;
    int64_t out_pos[3];
    uint64_t resampling_rng_state;
    uint64_t resampling_rng_stream;
    uint32_t checksum;
};
#pragma pack(pop)

struct CheckpointManager {
    char base_name[512];
    char chk_dir[512];
    bool enabled;
    int interval_secs;
    int detail_cap;
    time_t last_chk_time;
    int64_t step_count;
};

bool checkpoint_path(const CheckpointManager& manager, const char* suffix,
                     char* output, size_t output_size);

bool checkpoint_init_bc(CheckpointManager& manager,
                        int L, int N, int R, int nSteps, int seed,
                        int D_num, int D_denum, int heat, int detail_cap,
                        const char* checkpoint_directory,
                        bool enabled,
                        int interval_secs = BC_DEFAULT_CHECKPOINT_INTERVAL_SECS);
bool checkpoint_should_save(const CheckpointManager& manager);

bool checkpoint_save_bc(CheckpointManager& manager,
                        int L, int N, int R, int nSteps, int seed,
                        int D_num, int D_denum, int heat, int detail_cap,
                        const int* spins, const int* e_j, const int* e_delta,
                        const int* families, const int* order, int U,
                        const void* philox_states, size_t philox_state_bytes,
                        uint64_t resampling_rng_state,
                        uint64_t resampling_rng_stream,
                        const int64_t* output_positions = nullptr);

bool checkpoint_load_bc(const CheckpointManager& manager,
                        int L, int N, int R, int nSteps, int seed,
                        int D_num, int D_denum, int heat, int detail_cap,
                        int* spins, int* e_j, int* e_delta,
                        int* families, int* order,
                        int& U, int64_t& step_count,
                        void* philox_states, size_t philox_state_bytes,
                        uint64_t& resampling_rng_state,
                        uint64_t& resampling_rng_stream,
                        int64_t* output_positions = nullptr);

bool checkpoint_exists(const CheckpointManager& manager);
bool checkpoint_is_done(const CheckpointManager& manager);
bool checkpoint_mark_done(const CheckpointManager& manager);
