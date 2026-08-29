# MCPA 2D Ising migration

This branch now has an authoritative model target for the conventional
nearest-neighbor Ising model on a periodic square lattice:

\[
H = -\sum_{\langle ij\rangle} s_i s_j, \qquad s_i\in\{-1,+1\}.
\]

The historical `2DIsing` branch name was misleading: its old simulator was a
spin-1 Blume-Capel prototype. That source, old Visual Studio project, cached
notebook, reference tables, and cluster instruction file remain recoverable in
Git history but are intentionally absent from the maintained tree. New work
must use the `mcpa_ising2d` library. Its pure model contract is in
`include/mcpa/ising2d_model.hpp`; its first stochastic CUDA engine interface is
in `include/mcpa/ising2d_cuda.hpp`.

## Model contract

- The lattice is an `L x L` square torus in row-major order, with `L >= 3`.
- Every spin is exactly `-1` or `+1`.
- The energy counts each horizontal and vertical nearest-neighbor bond once.
- The exact minimum is `-2 L^2`.
- The exact maximum is `2 L^2` for even `L` and `2 L^2 - 4 L` for odd `L`,
  where the periodic antiferromagnetic state is frustrated.
- Cooling is a strict ceiling walk: a candidate is allowed only when `E < U`.
- Heating is a strict floor walk: a candidate is allowed only when `E > U`.
- The model API supplies initial sentinels one integer outside the exact
  spectrum, so the first strict constraint admits every state.

## CUDA engine contract

`DevicePopulationView` and `DeviceEngineView` are non-owning, explicitly typed
production memory views. Spins are replica-major, tracked energies use signed
64-bit `Energy`, accepted flips use unsigned 64-bit `FlipCount`, and generator
storage is the concrete CUDA `curandStatePhilox4_32_10_t` type. One CUDA thread
owns each replica throughout a kernel call.

The RNG-consumption contract is exact and part of this API:

1. `initialize_philox` calls `curand_init(seed, replica, 0, state)` once for
   each replica. It consumes no `curand()` output words.
2. `initialize_population` copies that state to a thread-local value and calls
   `curand()` exactly `L^2` times, in increasing row-major site order. The low
   bit maps `0` to `-1` and `1` to `+1`. It writes full energy, zero flips, and
   the advanced state.
3. `equilibrate` makes exactly `L^2 * sweeps` attempts per replica and consumes
   exactly one `curand()` word per attempt. `word % L^2` selects the site. The
   only Ising proposal is the deterministic sign flip, so there is no proposal
   draw. Rejection still consumes the site-selection word.
4. A cooling proposal is accepted only for `E_candidate < U`; a heating
   proposal only for `E_candidate > U`. Each call replaces `flip_counts[r]`
   with that call's accepted count and always stores the advanced Philox state.
5. `compute_energies` consumes no RNG and does not alter flip counts.

Changing draw count, draw order, low-bit mapping, modulo site selection, or
state write-back is a trajectory compatibility break. CUDA tests implement a
separate reference kernel and compare spins, tracked energies, flip counts, and
every byte of every Philox state for open, frozen, and mixed cooling/heating
trajectories.

Replica resampling has a separate checkpoint-ready PCG32 state consisting of
the two exact 64-bit words `state` and `stream`; it never calls `rand()`.
Seeding and every parent draw update that explicit value. An ordinary shell
uses exactly one PCG32 word per culled child. A terminal full cull and a
no-next-shell result consume no parent words. Cooling selects `max(E < U)` and
then culls `E >= U`; heating selects `min(E > U)` and culls `E <= U`. The host
API distinguishes `ok`, `terminal_full_cull`, and `no_next_shell` and preserves
the full signed-64-bit shell value and exact unsigned cull count.

Physical replica copying is a two-kernel gather/commit through a complete
typed scratch population. This immutable snapshot prevents an in-place child
write from corrupting a source used by another child. Spins and energies are
copied; Philox streams and per-call flip counters stay attached to destination
slots. Thus resampling does not alter the equilibration RNG-consumption
contract above.

## Functional simulator and output

The production entry point is:

```bash
./build/main_2d_ising seed L blocks threads nSteps heat \
    [checkpoint_dir|none] [checkpoint_every_shells]
```

`heat` is exactly `0` for cooling or `1` for heating. `L >= 3`; blocks,
threads, and `nSteps` are positive; their products and the GPU launch limits
are checked before allocation. `blocks * threads` defines the replica count.
The unsigned decimal seed initializes both Philox and the independent host
PCG. Any parse, range, CUDA, directory, allocation, or required-file failure
stops the run with a nonzero exit status.

Checkpointing is disabled when the optional argument is omitted or is `none`.
An enabled directory is created recursively; the positive interval defaults to
one completed shell. A checkpoint is taken only after safe replica copying and
output flushing, when GPU spins/energies, host families/order, the strict
boundary, completed-shell count, Philox states, and PCG state all describe the
same offspring population.

`MCPA_OUTPUT_ROOT` selects the root (default `./datasets`). The simulator
recursively creates its `2DIsing` child and writes exactly three common files:

- `{prefix}_main.txt` retains `E`, culling factor, and family concentration as
  its first three columns, then records exact integer `nCull`, a 17-digit
  culling fraction, status, per-shell wall timings, and population genealogy.
  Its first data row also records the GPU name, compute capability, total
  memory, free memory immediately before and after the run allocations, and
  CUDA driver/runtime versions; later rows contain `NA` in those seven fields,
  keeping the table rectangular without repeating run metadata.
- `{prefix}_agg_stats.txt` records exact shell and population count, flip-count
  sum/mean/rate, Ising `|M|` and `M^2`, and pre-resampling shell genealogy.
- `{prefix}_detailed_stats.txt` records shell, replica, family, flips, `M`,
  `|M|`, and `M^2` for the smallest matching replica indices.

`MCPA_DETAILED_CAP` is a validated nonnegative integer (default 100). Its value
and the literal policy `first_replica_indices_at_shell` are written on every
detailed row; detailed output is capped deterministically and is never a
random sample. A successful walk ends immediately after writing exactly one
`terminal_full_cull` row with `nCull=R` and `X=1`. An unexpected
`no_next_shell` before that terminal event is a failed run, preventing an
apparently complete but truncated dataset.

## Checkpoint contract

The model-specific version-2 payload stores `Spin` as exact signed bytes,
`Energy` as signed 64-bit integers, and preserves the complete run identity:
direction, `L`, `N`, `R`, sweeps, unsigned seed, detailed cap, and concrete
Philox-state byte size. It also stores families, sorting permutation, strict
boundary, completed-shell count, all three exact output byte offsets, both
PCG32 words, and every byte of every Philox state.

CRC32 covers the complete serialized payload. Save writes and `fsync`s a
temporary generation, rotates current to retained previous, promotes the
temporary file, and `fsync`s the directory. Load considers current, previous,
and a completed temporary fallback, choosing the valid generation with the
greatest completed-shell count. It rejects identity mismatch, invalid spin or
family domains, a non-permutation order, an even PCG stream, impossible
boundary/counts, wrong array sizes, and any tracked energy that differs from a
full square-torus bond recomputation.

The opaque Philox payload is byte-exact but remains a CUDA/cuRAND ABI object:
continuation assumes a compatible toolkit representation, and a different
`sizeof(curandStatePhilox4_32_10_t)` is rejected. Durable rotation currently
targets the project's POSIX/Linux environment (`fsync` plus atomic `rename`).
Recovery is shell-aligned; work inside an unfinished equilibration shell is
intentionally replayed from the preceding checkpoint.

On resume, each of the three output tables is required to exist and is
truncated to its stored byte offset before append. A clean terminal full cull
writes a synced identity-bearing `.done` marker. A matching completed rerun
returns success before GPU discovery, directory creation, or output opening,
so it cannot overwrite a completed dataset; a mismatched done identity is an
error.

## Build and test

The supported migration toolchain is GCC 11 with CUDA 12.4/nvcc and C++17.
The test entry point configures and builds the authoritative target:

```bash
./test.sh cpu
./test.sh gpu
./test.sh checkpoint
./test.sh all
./test.sh list
```

Each contract runs separately and prints its verdict immediately. Interactive
terminals show **PASSED** in green, **FAILED** in red, and **SKIPPED** in
yellow; redirected output stays plain. Set `MCPA_COLOR=always|never` to
override automatic colour detection. Full captured diagnostics are replayed
only for a failed contract.

`cpu` runs exact model and resampling oracles, `gpu` runs independent CUDA
trajectory/copy tests plus bounded production processes, `checkpoint` runs its
focused category, `all` runs everything, and `list` prints the one-line CTest
names. Successful test runs suppress configure/build chatter and print one
final verdict line per named contract; a failure reveals the captured CMake,
build, or CTest diagnostics. GPU and process test executables exit
with code 77 when no NVIDIA device is accessible; CTest marks that result as
skipped. The CPU suite includes the complete 512-state `L=3` DOS, constructed
states, local flip deltas, geometry, spectrum/sentinels, PCG replay,
terminal/no-next semantics, and signed-64-bit shell extremes. The direct GPU
suite covers initialization, energy, open/frozen/mixed trajectories, immutable
snapshot copying, and complete cooling/heating `L=3` processes with exact
three-file and unique-terminal-row checks. Checkpoint gates cover CPU
roundtrip, CRC corruption, current/previous rotation, temporary recovery, done
identity, semantic rejection, direct-GPU cooling/heating continuation, and
real SIGKILL/restart. The restart tests use deterministic zero timing columns
so all three resumed tables can be compared byte-for-byte with uninterrupted
references; normal production runs continue to record measured wall times.
