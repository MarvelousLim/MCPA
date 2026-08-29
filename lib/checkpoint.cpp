/*
 * checkpoint.cpp — implementation of the safe checkpoint/restart system.
 * CPU-only (no CUDA headers required).
 */

#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>       /* fabs — use this instead of fabsf to avoid C++ ambiguity */
#include <sys/stat.h>   /* mkdir / stat */
#include <errno.h>

#ifdef _WIN32
#include <direct.h>     /* _mkdir */
#define mkdir_p(d) _mkdir(d)
#define RENAME_ATOMIC(src,dst) MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING)
#else
#include <fcntl.h>
#include <unistd.h>
#define mkdir_p(d) mkdir(d, 0755)
#define RENAME_ATOMIC(src,dst) (rename(src, dst) == 0)
#endif

/* ── Internal: spin packing ────────────────────────────────────────────────── */
/* BC spins σ ∈ {-1,0,+1}: encode as (σ+1) ∈ {0,1,2} in 2 bits each.
 * Pack 16 spins per uint32_t.  Unused 2-bit pattern = 0b11 (3). */

static size_t n_spin_words_needed(size_t total_spins) {
    return (total_spins + 15) / 16;
}

static void pack_spins(const int* spins, size_t total, uint32_t* packed) {
    size_t nw = n_spin_words_needed(total);
    memset(packed, 0xFF, nw * sizeof(uint32_t));   /* fill with 0xFF (unused pattern) */
    for (size_t i = 0; i < total; i++) {
        uint32_t val = (uint32_t)(spins[i] + 1) & 0x3;    /* 0,1,2 */
        size_t word = i / 16;
        int bit = (int)((i % 16) * 2);
        packed[word] = (packed[word] & ~(0x3u << bit)) | (val << bit);
    }
}

static void unpack_spins(const uint32_t* packed, size_t total, int* spins) {
    for (size_t i = 0; i < total; i++) {
        size_t word = i / 16;
        int bit = (int)((i % 16) * 2);
        uint32_t val = (packed[word] >> bit) & 0x3;
        spins[i] = (int)val - 1;   /* 0→-1, 1→0, 2→+1 */
    }
}

/* ── Internal: checksum ─────────────────────────────────────────────────────── */
static uint32_t crc32_update(uint32_t crc, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < len; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return crc;
}

static uint32_t crc32_checksum(const void* data, size_t len) {
    return crc32_update(0xffffffffU, data, len) ^ 0xffffffffU;
}

/* ── Internal: file existence check ──────────────────────────────────────────── */
static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* ── Internal: ensure directory exists ────────────────────────────────────── */
static void ensure_dir(const char* dir) {
    if (!file_exists(dir)) {
        if (mkdir_p(dir) != 0 && errno != EEXIST)
            fprintf(stderr, "[chk] WARNING: could not create dir %s\n", dir);
    }
}

static void sync_checkpoint_directory(const char* dir) {
#ifndef _WIN32
    const int descriptor = open(dir, O_RDONLY | O_DIRECTORY);
    if (descriptor >= 0) {
        (void)fsync(descriptor);
        close(descriptor);
    }
#else
    (void)dir;
#endif
}

/* ── Internal: try to read one checkpoint file ─────────────────────────────── */
static bool try_load_file(const char* path,
                           int L, int N, int R,
                           int nSteps, int seed, float D, int heat,
                           int*   h_spin,
                           float* h_E,
                           int*   h_family,
                           int*   h_O,
                           float& U,
                           int64_t& step_count) {
    if (!file_exists(path)) return false;

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[chk] cannot open %s\n", path);
        return false;
    }

    /* Read header */
    CheckpointHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) { fclose(fp); return false; }

    /* Validate magic / version */
    if (memcmp(hdr.magic, CHECKPOINT_MAGIC, 8) != 0 || hdr.version != CHECKPOINT_VERSION) {
        fprintf(stderr, "[chk] %s: bad magic or version\n", path);
        fclose(fp); return false;
    }

    /* Validate params match — refuse to load a checkpoint from a different run */
    if (hdr.L != L || hdr.N != N || hdr.R != R ||
        hdr.nSteps != nSteps || hdr.seed != seed ||
        fabs((double)(hdr.D - D)) > 1e-5 || hdr.heat != heat) {
        fprintf(stderr, "[chk] %s: params mismatch (different run?)\n", path);
        fclose(fp); return false;
    }

    int nw = hdr.n_spin_words;
    size_t spin_bytes   = (size_t)nw     * sizeof(uint32_t);
    size_t E_bytes      = (size_t)hdr.R  * sizeof(float);
    size_t family_bytes = (size_t)hdr.R  * sizeof(int);
    size_t O_bytes      = (size_t)hdr.R  * sizeof(int);

    /* Read all payload into a temporary buffer for checksum */
    size_t total = spin_bytes + E_bytes + family_bytes + O_bytes;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) { fclose(fp); return false; }
    if (fread(buf, 1, total, fp) != total) {
        fprintf(stderr, "[chk] %s: truncated data\n", path);
        free(buf); fclose(fp); return false;
    }
    fclose(fp);

    /* Validate checksum */
    uint32_t csum = crc32_checksum(buf, total);
    if (csum != hdr.checksum) {
        fprintf(stderr, "[chk] %s: checksum mismatch (file corrupted)\n", path);
        free(buf); return false;
    }

    /* Unpack spins */
    const uint32_t* packed_spins = (const uint32_t*)buf;
    unpack_spins(packed_spins, (size_t)hdr.N * (size_t)hdr.R, h_spin);

    const uint8_t* p = buf + spin_bytes;
    memcpy(h_E,      p,               E_bytes);      p += E_bytes;
    memcpy(h_family, p,               family_bytes); p += family_bytes;
    memcpy(h_O,      p,               O_bytes);

    U          = hdr.U;
    step_count = hdr.step_count;

    free(buf);
    printf("[chk] Loaded checkpoint from %s  (U=%.4f  steps=%lld  saved %s)\n",
           path, U, (long long)step_count, ctime((const time_t*)&hdr.timestamp));
    return true;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Public API
 * ══════════════════════════════════════════════════════════════════════════════*/

void checkpoint_init(CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     const char* chk_dir,
                     bool enabled,
                     int interval_secs,
                     const char* base_name_override) {
    (void)L;  /* L is encoded through N in the legacy Blume-Capel filename. */
    memset(&mgr, 0, sizeof(mgr));
    mgr.enabled        = enabled;
    mgr.interval_secs  = interval_secs;
    mgr.last_chk_time  = time(nullptr);   /* don't save immediately on start */
    mgr.step_count     = 0;
    mgr.loaded         = false;

    strncpy(mgr.chk_dir, chk_dir, sizeof(mgr.chk_dir) - 1);
    /* Build human-readable base name (or use caller-supplied override) */
    if (base_name_override && base_name_override[0]) {
        strncpy(mgr.base_name, base_name_override, sizeof(mgr.base_name) - 1);
        mgr.base_name[sizeof(mgr.base_name)-1] = 0;
    } else {
        snprintf(mgr.base_name, sizeof(mgr.base_name),
                 "2DBlume%s_q3_D%.6f_N%d_R%d_nSteps%d_run%d",
                 heat ? "Heating" : "", D, N, R, nSteps, seed);
    }

    if (enabled) {
        ensure_dir(chk_dir);
        printf("[chk] Checkpoint dir: %s/\n", chk_dir);
        printf("[chk] Base name:      %s\n", mgr.base_name);
        printf("[chk] Interval:       %d s (%.1f h)\n",
               interval_secs, interval_secs / 3600.0);
    }
}

bool checkpoint_should_save(const CheckpointManager& mgr) {
    if (!mgr.enabled) return false;
    return (time(nullptr) - mgr.last_chk_time) >= mgr.interval_secs;
}

bool checkpoint_save(CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     const int*   h_spin,
                     const float* h_E,
                     const int*   h_family,
                     const int*   h_O,
                     float U,
                     const int64_t* out_pos) {
    if (!mgr.enabled) return false;
    (void)out_pos;  /* Output positions belong to the current int format. */

    char path_tmp[CHECKPOINT_PATH_CAPACITY];
    char path_cur[CHECKPOINT_PATH_CAPACITY];
    char path_prev[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".tmp",      path_tmp,  sizeof(path_tmp));
    chk_path(mgr, ".bin",      path_cur,  sizeof(path_cur));
    chk_path(mgr, ".prev.bin", path_prev, sizeof(path_prev));

    ensure_dir(mgr.chk_dir);

    /* ── Pack spins ──────────────────────────────────────────────────────── */
    size_t total_spins = (size_t)R * (size_t)N;
    size_t nw          = n_spin_words_needed(total_spins);
    uint32_t* packed = (uint32_t*)malloc(nw * sizeof(uint32_t));
    if (!packed) {
        fprintf(stderr, "[chk] out of memory during checkpoint\n");
        return false;
    }
    pack_spins(h_spin, total_spins, packed);

    /* ── Compute payload sizes ───────────────────────────────────────────── */
    size_t spin_bytes   = nw * sizeof(uint32_t);
    size_t E_bytes      = (size_t)R  * sizeof(float);
    size_t family_bytes = (size_t)R  * sizeof(int);
    size_t O_bytes      = (size_t)R  * sizeof(int);
    size_t total        = spin_bytes + E_bytes + family_bytes + O_bytes;

    /* ── Build payload buffer and compute checksum ───────────────────────── */
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) { free(packed); return false; }
    uint8_t* p = buf;
    memcpy(p, packed,   spin_bytes);   p += spin_bytes;
    memcpy(p, h_E,      E_bytes);      p += E_bytes;
    memcpy(p, h_family, family_bytes); p += family_bytes;
    memcpy(p, h_O,      O_bytes);
    free(packed);

    uint32_t csum = crc32_checksum(buf, total);

    /* ── Build header ────────────────────────────────────────────────────── */
    CheckpointHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, CHECKPOINT_MAGIC, 8);
    hdr.version      = CHECKPOINT_VERSION;
    hdr.L  = L; hdr.N = N; hdr.R = R;
    hdr.nSteps = nSteps; hdr.seed = seed;
    hdr.D    = D; hdr.heat = heat;
    hdr.U    = U;
    hdr.step_count   = mgr.step_count;
    hdr.timestamp    = (int64_t)time(nullptr);
    hdr.n_spin_words = nw;
    hdr.checksum     = csum;

    /* ── (1) Write to .tmp ───────────────────────────────────────────────── */
    FILE* fp = fopen(path_tmp, "wb");
    if (!fp) {
        fprintf(stderr, "[chk] cannot open %s for writing\n", path_tmp);
        free(buf); return false;
    }
    bool ok = (fwrite(&hdr, sizeof(hdr), 1, fp) == 1) &&
              (fwrite(buf, 1, total, fp) == total);
    fflush(fp);
    fclose(fp);
    free(buf);
    if (!ok) {
        fprintf(stderr, "[chk] write error to %s\n", path_tmp);
        remove(path_tmp);
        return false;
    }

    /* ── (2) Rotate: cur → prev, tmp → cur ──────────────────────────────── */
    /* Remove old prev (it's safe now: .tmp is complete) */
    if (file_exists(path_prev)) remove(path_prev);
    /* Atomically rename cur → prev */
    if (file_exists(path_cur)) (void)RENAME_ATOMIC(path_cur, path_prev);
    /* Atomically rename tmp → cur */
    if (!RENAME_ATOMIC(path_tmp, path_cur)) {
        fprintf(stderr, "[chk] rename %s → %s failed\n", path_tmp, path_cur);
        return false;
    }

    mgr.last_chk_time = time(nullptr);
    printf("[chk] Checkpoint saved: %s  (U=%.4f  steps=%lld  payload %.1f MB)\n",
           path_cur, U, (long long)mgr.step_count, total / 1048576.0);
    return true;
}

bool checkpoint_load(const CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     int*   h_spin,
                     float* h_E,
                     int*   h_family,
                     int*   h_O,
                     float& U,
                     int64_t& step_count,
                     int64_t* out_pos_out) {
    if (!mgr.enabled) return false;
    (void)out_pos_out;  /* Output positions belong to the current int format. */

    /* Fool protection: if previous run completed normally, start fresh */
    char path_done[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".done", path_done, sizeof(path_done));
    if (file_exists(path_done)) {
        printf("[chk] Found .done marker — previous run completed normally. Starting fresh.\n");
        printf("[chk] Delete %s to re-run from a checkpoint.\n", path_done);
        return false;
    }

    /* Try current checkpoint, then previous (for partial-write recovery) */
    char path_cur[CHECKPOINT_PATH_CAPACITY];
    char path_prev[CHECKPOINT_PATH_CAPACITY];
    char path_tmp[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".bin",      path_cur,  sizeof(path_cur));
    chk_path(mgr, ".prev.bin", path_prev, sizeof(path_prev));
    chk_path(mgr, ".tmp",      path_tmp,  sizeof(path_tmp));

    /* If a .tmp exists, the last save was interrupted; .cur (if it exists)
     * is the last complete save.  .tmp might also be complete — try .cur first. */
    if (file_exists(path_tmp) && !file_exists(path_cur)) {
        /* Unusual: .tmp exists but no .cur → try .tmp as the current */
        printf("[chk] WARNING: found .tmp but no .bin — attempting .tmp as current\n");
        if (try_load_file(path_tmp, L, N, R, nSteps, seed, D, heat,
                          h_spin, h_E, h_family, h_O, U, step_count))
            return true;
    }
    if (try_load_file(path_cur, L, N, R, nSteps, seed, D, heat,
                      h_spin, h_E, h_family, h_O, U, step_count))
        return true;
    if (try_load_file(path_prev, L, N, R, nSteps, seed, D, heat,
                      h_spin, h_E, h_family, h_O, U, step_count))
        return true;

    printf("[chk] No valid checkpoint found. Starting from scratch.\n");
    return false;
}

void checkpoint_mark_done(const CheckpointManager& mgr) {
    if (!mgr.enabled) return;
    char path_done[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".done", path_done, sizeof(path_done));
    FILE* fp = fopen(path_done, "w");
    if (fp) {
        time_t now = time(nullptr);
        fprintf(fp, "completed %s", ctime(&now));
        fflush(fp);
#ifndef _WIN32
        (void)fsync(fileno(fp));
#endif
        fclose(fp);
        sync_checkpoint_directory(mgr.chk_dir);
        printf("[chk] Run completed. Wrote %s\n", path_done);
    }
}

/* ── Int-energy overloads (BaxterWu) ──────────────────────────────────────── */

static bool write_chk_file_int(const char* path,
                                int L, int N, int R, int nSteps, int seed,
                                float D, int heat, int U, int64_t step_count,
                                const int* h_spin, const int* h_E,
                                const int* h_family, const int* h_O,
                                const void* h_rng_state, size_t rng_state_bytes,
                                uint64_t resampling_rng_state,
                                uint64_t resampling_rng_stream,
                                const int64_t* out_pos) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[chk] Cannot open %s: %s\n", path, strerror(errno)); return false; }

    int nw = (int)n_spin_words_needed((size_t)R * (size_t)N);
    uint32_t* packed = (uint32_t*)malloc((size_t)nw * sizeof(uint32_t));
    if (!packed) { fclose(f); return false; }
    pack_spins(h_spin, (size_t)R * (size_t)N, packed);

    CheckpointHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, CHECKPOINT_MAGIC, 8);
    hdr.version      = CHECKPOINT_VERSION;
    hdr.L = L; hdr.N = N; hdr.R = R;
    hdr.nSteps = nSteps; hdr.seed = seed;
    hdr.D          = D;
    hdr.heat       = heat;
    hdr.U          = (float)U;
    hdr.step_count = step_count;
    hdr.timestamp  = (int64_t)time(nullptr);
    hdr.n_spin_words = nw;
    hdr.energy_type  = CHK_ENERGY_INT;
    for (int _i=0;_i<3;_i++) hdr.out_pos[_i] = out_pos ? out_pos[_i] : -1;
    hdr.rng_state_bytes = static_cast<int64_t>(rng_state_bytes);
    hdr.resampling_rng_state = resampling_rng_state;
    hdr.resampling_rng_stream = resampling_rng_stream;
    hdr.checksum   = 0;

    uint32_t csum = crc32_update(0xffffffffU, &hdr, sizeof(hdr));
    csum = crc32_update(csum, packed,  (size_t)nw * 4);
    csum = crc32_update(csum, h_E,     (size_t)R * sizeof(int));
    csum = crc32_update(csum, h_family,(size_t)R * sizeof(int));
    csum = crc32_update(csum, h_O,     (size_t)R * sizeof(int));
    if (rng_state_bytes > 0)
        csum = crc32_update(csum, h_rng_state, rng_state_bytes);
    hdr.checksum = csum ^ 0xffffffffU;

    bool ok = true;
    ok = ok && (fwrite(&hdr,    sizeof(hdr),    1,  f) == 1);
    ok = ok && (fwrite(packed,  sizeof(uint32_t),(size_t)nw, f) == (size_t)nw);
    ok = ok && (fwrite(h_E,     sizeof(int),    (size_t)R,  f) == (size_t)R);
    ok = ok && (fwrite(h_family,sizeof(int),    (size_t)R,  f) == (size_t)R);
    ok = ok && (fwrite(h_O,     sizeof(int),    (size_t)R,  f) == (size_t)R);
    if (rng_state_bytes > 0)
        ok = ok && (fwrite(h_rng_state, 1, rng_state_bytes, f) == rng_state_bytes);
    if (fflush(f) != 0) ok = false;
#ifndef _WIN32
    if (ok && fsync(fileno(f)) != 0) ok = false;
#endif
    if (fclose(f) != 0) ok = false;
    free(packed);
    if (!ok) fprintf(stderr, "[chk] Write error to %s\n", path);
    return ok;
}

static bool try_load_chk_int(const char* path,
                              int L, int N, int R, int nSteps, int seed,
                              float D, int heat,
                              int* h_spin, int* h_E, int* h_family, int* h_O,
                              int& U, int64_t& step_count,
                              void* h_rng_state, size_t rng_state_bytes,
                              uint64_t& resampling_rng_state,
                              uint64_t& resampling_rng_stream) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    CheckpointHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return false; }

    if (memcmp(hdr.magic, CHECKPOINT_MAGIC, 8) != 0 || hdr.version != CHECKPOINT_VERSION) {
        fprintf(stderr, "[chk] %s: bad magic/version\n", path);
        fclose(f); return false;
    }
    if (hdr.energy_type != CHK_ENERGY_INT) {
        fprintf(stderr, "[chk] %s: energy type mismatch (expected int)\n", path);
        fclose(f); return false;
    }
    bool params_ok = (hdr.L == L && hdr.N == N && hdr.R == R
                   && hdr.nSteps == nSteps && hdr.seed == seed
                   && hdr.heat == heat
                   && fabs((double)(hdr.D - D)) < 1e-5);
    if (!params_ok) { fprintf(stderr, "[chk] %s: params mismatch\n", path); fclose(f); return false; }
    if (hdr.rng_state_bytes < 0
        || static_cast<uint64_t>(hdr.rng_state_bytes) != rng_state_bytes
        || (rng_state_bytes > 0 && h_rng_state == nullptr)) {
        fprintf(stderr, "[chk] %s: RNG state size mismatch\n", path);
        fclose(f); return false;
    }

    int nw = hdr.n_spin_words;
    if (nw != (int)n_spin_words_needed((size_t)R * (size_t)N)) { fclose(f); return false; }

    uint32_t* packed = (uint32_t*)malloc((size_t)nw * sizeof(uint32_t));
    if (!packed) { fclose(f); return false; }

    bool ok = true;
    ok = ok && (fread(packed,   sizeof(uint32_t),(size_t)nw, f) == (size_t)nw);
    ok = ok && (fread(h_E,      sizeof(int),     (size_t)R,  f) == (size_t)R);
    ok = ok && (fread(h_family, sizeof(int),     (size_t)R,  f) == (size_t)R);
    ok = ok && (fread(h_O,      sizeof(int),     (size_t)R,  f) == (size_t)R);
    if (rng_state_bytes > 0)
        ok = ok && (fread(h_rng_state, 1, rng_state_bytes, f) == rng_state_bytes);
    fclose(f);
    if (!ok) { free(packed); return false; }

    /* Verify checksum */
    uint32_t saved = hdr.checksum; hdr.checksum = 0;
    uint32_t csum  = crc32_update(0xffffffffU, &hdr, sizeof(hdr));
    csum = crc32_update(csum, packed,  (size_t)nw * 4);
    csum = crc32_update(csum, h_E,     (size_t)R * sizeof(int));
    csum = crc32_update(csum, h_family,(size_t)R * sizeof(int));
    csum = crc32_update(csum, h_O,     (size_t)R * sizeof(int));
    if (rng_state_bytes > 0)
        csum = crc32_update(csum, h_rng_state, rng_state_bytes);
    csum ^= 0xffffffffU;
    if (csum != saved) {
        fprintf(stderr, "[chk] %s: checksum mismatch (corrupted)\n", path);
        free(packed); return false;
    }
    unpack_spins(packed, (size_t)R * (size_t)N, h_spin);
    free(packed);

    U          = (int)hdr.U;
    step_count = hdr.step_count;
    resampling_rng_state = hdr.resampling_rng_state;
    resampling_rng_stream = hdr.resampling_rng_stream;

    char tstr[32]; time_t ts = (time_t)hdr.timestamp;
    strftime(tstr, sizeof(tstr), "%Y-%m-%d %H:%M:%S", localtime(&ts));
    printf("[chk] Loaded %s  (saved %s  U=%d  step=%lld)\n",
           path, tstr, U, (long long)step_count);
    return true;
}

/* Public int-energy overloads */

bool checkpoint_save(CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     const int* h_spin, const int* h_E,
                     const int* h_family, const int* h_O, int U,
                     const void* h_rng_state, size_t rng_state_bytes,
                     uint64_t resampling_rng_state,
                     uint64_t resampling_rng_stream,
                     const int64_t* out_pos) {
    if (!mgr.enabled) return true;

    char path_tmp[CHECKPOINT_PATH_CAPACITY];
    char path_cur[CHECKPOINT_PATH_CAPACITY];
    char path_prev[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".tmp",      path_tmp,  sizeof(path_tmp));
    chk_path(mgr, ".bin",      path_cur,  sizeof(path_cur));
    chk_path(mgr, ".prev.bin", path_prev, sizeof(path_prev));

    printf("[chk] Writing checkpoint to %s ...\n", path_tmp); fflush(stdout);
    if (!write_chk_file_int(path_tmp, L, N, R, nSteps, seed, D, heat, U,
                             mgr.step_count, h_spin, h_E, h_family, h_O,
                             h_rng_state, rng_state_bytes,
                             resampling_rng_state, resampling_rng_stream,
                             out_pos)) {
        fprintf(stderr, "[chk] Checkpoint write FAILED; keeping old checkpoint.\n");
        return false;
    }
    if (file_exists(path_prev) && remove(path_prev) != 0) {
        fprintf(stderr, "[chk] cannot remove old previous checkpoint: %s\n",
                strerror(errno));
        return false;
    }
    if (file_exists(path_cur) && !RENAME_ATOMIC(path_cur, path_prev)) {
        fprintf(stderr, "[chk] cannot rotate current checkpoint: %s\n",
                strerror(errno));
        return false;
    }
    if (!RENAME_ATOMIC(path_tmp, path_cur)) {
        fprintf(stderr, "[chk] rename failed: %s\n", strerror(errno));
        return false;
    }
    sync_checkpoint_directory(mgr.chk_dir);
    mgr.last_chk_time = time(nullptr);
    printf("[chk] Checkpoint saved  (U=%d  step=%lld)\n", U, (long long)mgr.step_count);
    return true;
}

bool checkpoint_load(const CheckpointManager& mgr,
                     int L, int N, int R, int nSteps, int seed, float D, int heat,
                     int* h_spin, int* h_E, int* h_family, int* h_O,
                     int& U, int64_t& step_count,
                     void* h_rng_state, size_t rng_state_bytes,
                     uint64_t& resampling_rng_state,
                     uint64_t& resampling_rng_stream,
                     int64_t* out_pos_out) {
    if (!mgr.enabled) return false;

    char path_done[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".done", path_done, sizeof(path_done));
    { FILE* f = fopen(path_done, "r"); if (f) { fclose(f);
        printf("[chk] .done file found — starting fresh.\n"); return false; } }

    char path_cur[CHECKPOINT_PATH_CAPACITY];
    char path_prev[CHECKPOINT_PATH_CAPACITY];
    char path_tmp[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".bin",      path_cur,  sizeof(path_cur));
    chk_path(mgr, ".prev.bin", path_prev, sizeof(path_prev));
    chk_path(mgr, ".tmp",      path_tmp,  sizeof(path_tmp));

    if (try_load_chk_int(path_cur,  L, N, R, nSteps, seed, D, heat,
                         h_spin, h_E, h_family, h_O, U, step_count,
                         h_rng_state, rng_state_bytes,
                         resampling_rng_state, resampling_rng_stream)) {
        if (out_pos_out) {
            FILE* fp = fopen(path_cur, "rb");
            if (fp) {
                CheckpointHeader _hdr;
                if (fread(&_hdr, sizeof(_hdr), 1, fp) == 1)
                    for (int _i=0;_i<3;_i++) out_pos_out[_i] = _hdr.out_pos[_i];
                fclose(fp);
            }
        }
        return true;
    }
    printf("[chk] Primary checkpoint invalid or missing; trying previous...\n");
    if (try_load_chk_int(path_prev, L, N, R, nSteps, seed, D, heat,
                         h_spin, h_E, h_family, h_O, U, step_count,
                         h_rng_state, rng_state_bytes,
                         resampling_rng_state, resampling_rng_stream)) {
        if (out_pos_out) {
            FILE* fp = fopen(path_prev, "rb");
            if (fp) {
                CheckpointHeader _hdr;
                if (fread(&_hdr, sizeof(_hdr), 1, fp) == 1)
                    for (int _i=0;_i<3;_i++) out_pos_out[_i] = _hdr.out_pos[_i];
                fclose(fp);
            }
        }
        return true;
    }
    printf("[chk] Previous checkpoint invalid or missing; trying temporary...\n");
    if (try_load_chk_int(path_tmp, L, N, R, nSteps, seed, D, heat,
                         h_spin, h_E, h_family, h_O, U, step_count,
                         h_rng_state, rng_state_bytes,
                         resampling_rng_state, resampling_rng_stream)) {
        if (out_pos_out) {
            FILE* fp = fopen(path_tmp, "rb");
            if (fp) {
                CheckpointHeader _hdr;
                if (fread(&_hdr, sizeof(_hdr), 1, fp) == 1)
                    for (int _i=0;_i<3;_i++) out_pos_out[_i] = _hdr.out_pos[_i];
                fclose(fp);
            }
        }
        return true;
    }
    printf("[chk] No valid checkpoint found — starting fresh.\n");
    return false;
}

/* ── checkpoint_exists ────────────────────────────────────────────────────── */

bool checkpoint_exists(const CheckpointManager& mgr) {
    if (!mgr.enabled) return false;

    /* Presence and validity are intentionally separate.  Even a one-byte
     * generation is a candidate set that must fail loudly if no fallback
     * validates; it must never authorize fresh-output truncation. */
    char path_cur[CHECKPOINT_PATH_CAPACITY];
    char path_prev[CHECKPOINT_PATH_CAPACITY];
    char path_tmp[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".bin",      path_cur,  sizeof(path_cur));
    chk_path(mgr, ".prev.bin", path_prev, sizeof(path_prev));
    chk_path(mgr, ".tmp",      path_tmp,  sizeof(path_tmp));

    const char* _paths[3] = {path_cur, path_prev, path_tmp};
    for (int _pi = 0; _pi < 3; _pi++) { const char* p = _paths[_pi];
        FILE* f = fopen(p, "rb");
        if (!f) continue;
        fclose(f);
        return true;
    }
    return false;
}

bool checkpoint_is_done(const CheckpointManager& mgr) {
    if (!mgr.enabled) return false;
    char path_done[CHECKPOINT_PATH_CAPACITY];
    chk_path(mgr, ".done", path_done, sizeof(path_done));
    FILE* file = fopen(path_done, "r");
    if (!file) return false;
    fclose(file);
    return true;
}
