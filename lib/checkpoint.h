#pragma once
/*
 * checkpoint.h  — Safe checkpoint/restart for MCPA runs.
 *
 * Rationale
 * ---------
 * Large runs on 30-day-limited clusters need to survive job kills.
 * Checkpoints are written periodically at a clean shell boundary.  Baxter-Wu
 * saves after replica copying, when configurations, energies, family labels,
 * ordering, Philox states and the host resampling RNG all describe the same
 * population that will enter the next shell.
 *
 * Safety guarantees
 * -----------------
 * Two checkpoint files are always kept: current + previous.
 * Write sequence:
 *   (1) write {base}_chk.tmp
 *   (2) rename {base}_chk.bin  →  {base}_chk_prev.bin  (atomic)
 *   (3) rename {base}_chk.tmp  →  {base}_chk.bin        (atomic)
 *   (4) keep both current and previous generations
 *
 * If the process is killed during (1), .tmp is incomplete; fall back to .bin.
 * If killed during (2-3), .tmp is complete; .bin and .tmp exist; pick .bin
 * (it was complete) then .tmp (complete alternative).
 * If .bin later fails validation, resume falls back to .prev.bin.  A complete
 * .tmp is also recoverable when no committed generation survives.
 *
 * A .done marker is written on clean termination.  A matching completed run
 * exits without opening or changing its output files.  Delete or relocate the
 * complete checkpoint set only when a deliberate fresh run is intended.
 *
 * Spin compression
 * ----------------
 * BC spins σ ∈ {-1,0,+1} are packed 16 per uint32_t (2 bits each, value+1).
 * For R=131072, N=1024 this saves 134 MB → 8.4 MB (16× reduction).
 *
 * File naming
 * -----------
 * All files share the same prefix as main output files:
 *   2DBlume_q3_D1.960000_N1024_R131072_nSteps10_run0
 * Checkpoint files are placed in a separate directory and suffixed _chk.*
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

/* ── Constants ─────────────────────────────────────────────────────────────── */
#define CHECKPOINT_MAGIC    "MCPA_BC1"   /* 8 bytes, version stamp */
#define CHECKPOINT_VERSION  6
#define CHK_ENERGY_FLOAT   0   /* BlumeCapel: float E = E_J + D*E_delta */
#define CHK_ENERGY_INT     1   /* BaxterWu:  int   E (always integer)   */
#define CHECKPOINT_SUFFIX   "_chk"
#define DEFAULT_CHK_INTERVAL_SECS 900  /* 15 minutes */
/* chk_dir and base_name may each contain 511 characters. */
#define CHECKPOINT_PATH_CAPACITY 1100

/* ── On-disk header (fixed size, version-stable) ──────────────────────────── */
#pragma pack(push, 1)
struct CheckpointHeader {
    char    magic[8];       /* MCPA_BC1 */
    int32_t version;
    int32_t L, N, R;
    int32_t nSteps, seed;
    float   D;
    int32_t heat;
    float   U;              /* current energy ceiling / floor */
    int64_t step_count;     /* culling steps completed so far */
    int64_t timestamp;      /* unix time of this checkpoint */
    int32_t n_spin_words;   /* number of uint32_t words in packed spin data */
    int32_t energy_type;    /* CHK_ENERGY_FLOAT or CHK_ENERGY_INT */
    int64_t out_pos[3];     /* byte positions in (main, agg, detail) output files at save time;
                             * use to truncate files on resume so data matches checkpoint state. */
    int64_t rng_state_bytes; /* opaque device RNG-state payload; zero when unused */
    uint64_t resampling_rng_state;
    uint64_t resampling_rng_stream;
    uint32_t checksum;      /* CRC-32 integrity check */
};
#pragma pack(pop)

/* ── Runtime manager (kept on the stack in main) ──────────────────────────── */
struct CheckpointManager {
    char     base_name[512]; /* e.g. "2DBlume_q3_D1.960000_N1024_R131072_nSteps10_run0" */
    char     chk_dir[512];   /* directory for checkpoint files */
    bool     enabled;
    int      interval_secs;  /* seconds between checkpoints */
    time_t   last_chk_time;  /* wall time of last successful checkpoint write */
    int64_t  step_count;     /* incremented after each culling step */
    bool     loaded;         /* true if we successfully resumed from a checkpoint */
};

/* ── File path helpers ────────────────────────────────────────────────────── */
static inline void chk_path(const CheckpointManager& m, const char* suffix,
                              char* out, size_t sz) {
    snprintf(out, sz, "%s/%s%s%s", m.chk_dir, m.base_name, CHECKPOINT_SUFFIX, suffix);
}
/* e.g. suffix=".bin", ".prev.bin", ".tmp", ".done" */

/* ── API ──────────────────────────────────────────────────────────────────── */

/* Initialise.  Call once before the main loop.
 * base_name_override: if non-NULL, use this as the checkpoint file prefix
 *   instead of the auto-generated "2DBlume..." name.  Useful for BaxterWu
 *   and other models that don't have a D parameter. */
void checkpoint_init(CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     const char* chk_dir,
                     bool enabled,
                     int interval_secs = DEFAULT_CHK_INTERVAL_SECS,
                     const char* base_name_override = nullptr);

/* True if it is time to write a checkpoint (time-based). */
bool checkpoint_should_save(const CheckpointManager& mgr);

/* Int-energy overloads — for BaxterWu where E is always integer.
 * These use the same file format but store energies as int32_t, avoiding
 * any float conversion and the associated type confusion. */
bool checkpoint_save(CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     const int*   h_spin,
                     const int*   h_E,
                     const int*   h_family,
                     const int*   h_O,
                     int          U,
                     const void*  h_rng_state,
                     size_t       rng_state_bytes,
                     uint64_t     resampling_rng_state,
                     uint64_t     resampling_rng_stream,
                     const int64_t* out_pos = nullptr);

bool checkpoint_load(const CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     int*    h_spin,
                     int*    h_E,
                     int*    h_family,
                     int*    h_O,
                     int&    U,
                     int64_t& step_count,
                     void*   h_rng_state,
                     size_t  rng_state_bytes,
                     uint64_t& resampling_rng_state,
                     uint64_t& resampling_rng_stream,
                     int64_t* out_pos_out = nullptr);

/* Write a checkpoint safely.
 * For Baxter-Wu call only after update_replicas, when spins, energies and
 * family labels all describe the same offspring population.  h_spin, h_E,
 * h_family, h_O and RNG state must all be fresh host-side copies.
 * Returns true on success. */
/* out_pos: byte positions of (main, agg, detail) output files at save time.
 * Pass NULL to leave as -1 (no truncation will happen on resume). */
bool checkpoint_save(CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     const int*   h_spin,
                     const float* h_E,
                     const int*   h_family,
                     const int*   h_O,
                     float U,
                     const int64_t* out_pos = nullptr);

/* Attempt to load the most recent valid checkpoint.
 * Fills h_spin / h_E / h_family / h_O / U / step_count if successful.
 * Returns true if a checkpoint was loaded; false if starting fresh.
 *
 * A false return means that no valid generation was loaded.  Callers must
 * distinguish a genuinely absent set from a present-but-invalid set and from
 * a clean `.done` marker before deciding whether a fresh run is allowed.
 */
/* out_pos_out: if non-NULL, filled with the stored output-file byte positions.
 * Caller should truncate the open output files to these positions to avoid
 * writing duplicate data when resuming. */
bool checkpoint_load(const CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     int*   h_spin,
                     float* h_E,
                     int*   h_family,
                     int*   h_O,
                     float& U,
                     int64_t& step_count,
                     int64_t* out_pos_out = nullptr);

/* Returns true if any candidate generation exists.  This is deliberately a
 * presence check, not a validity check: a truncated file must lead to an
 * explicit invalid-set failure rather than a silently fresh trajectory. */
bool checkpoint_exists(const CheckpointManager& mgr);

/* True when the matching run has a clean-completion marker. */
bool checkpoint_is_done(const CheckpointManager& mgr);

/* Write .done marker.  Call at the very end of a successful run.
 * The executable treats a later matching invocation as a successful no-op. */
void checkpoint_mark_done(const CheckpointManager& mgr);
