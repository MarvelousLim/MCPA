# Monte-Carlo Population Annealing
A PhD work with Lev Shur, Natan Rose and John Machta.

## 2DPotts_v3 structure

- `lib/mem_estimate.{h,cpp}` — shared setup-memory estimator for all model branches.
- `lib/potts_lib.{h,cu}` — Potts model logic, CUDA kernels, and resampling.
- `lib/potts_output.{h,cpp}` — versioned three-file statistics contract.
- `main/main.cpp` — command-line parsing, allocation, simulation loop, and cleanup.
- `tests/test_cpu_*.cpp` — independent model, output, and checkpoint contracts.
- `tests/test_cuda_*.cu` — independent CUDA trajectory and restart oracles.

Other model branches can reuse the same `MemEstimate` API by supplying their
own lattice / aux / RNG byte budgets to `estimate_setup_memory`.

Run the simulation as:

```text
main_potts seed L blocks threads nSteps q heat
```

Here `heat=0` selects cooling and `heat=1` selects heating.

## Reproducible CMake build and CPU contracts

The migration build uses CMake 3.22+, C++17, CUDA 12.4, and GCC 11 by
default. It preserves the production representation (`char` spins and runtime
`q`) and does not alter the Potts proposal sequence, resampling law, endpoint,
or leading output columns.

```bash
./test.sh cpu       # configure, build, and run all named CPU contracts
./test.sh gpu       # bitwise CUDA trajectory contracts; skips without a GPU
./test.sh checkpoint # one-line checkpoint/restart contracts
./test.sh all       # CPU and CUDA contracts
./test.sh list      # list the registered one-line contracts
cmake --build build --target main_potts
```

Each contract prints its verdict immediately. Interactive terminals show
**PASSED** in green, **FAILED** in red, and **SKIPPED** in yellow; redirected
output stays plain. Set `MCPA_COLOR=always|never` to override automatic colour
detection. Full captured diagnostics are replayed only for a failed contract.

Override `MCPA_BUILD_DIR`, `MCPA_CUDA_ARCHITECTURES`, `MCPA_BUILD_JOBS`,
`CUDACXX`, or `CXX` when required by the local toolchain.

Simulation output defaults to `./datasets`; set `MCPA_OUTPUT_ROOT` to choose a
different root. The executable creates `2DPotts/` below that root and stops
before CUDA setup if it cannot be created.

## Version 3 production output

Each run produces exactly three rectangular, whitespace-delimited files:

- `_main.txt` starts with the common `E culling_factor
  replica_family_avg_sq` columns, then records exact `nCull`, a 17-digit
  culling fraction, terminal status, `nSteps`, equilibration wall time,
  genealogy, and GPU metadata. Here `E` is the newly selected strict energy
  shell. For a normal row, genealogy is measured on the full post-resampling
  offspring labels; on the terminal full-cull row no offspring population
  exists, so it is the unchanged pre-resampling population. GPU metadata is
  populated on the first data row only and later rows contain `NA`, without
  changing the column count.
- `_agg_stats.txt` has the same `E` key and records only replicas whose tracked
  energy equals that shell. Its exact `population` therefore equals `nCull`.
  It contains accepted-flip sum/mean/rate, the first four moments of
  `m=(q*max_color_count/N-1)/(q-1)`, and pre-resampling family diagnostics.
- `_detailed_stats.txt` records shell, replica index, the family label measured
  before resampling, accepted flips, and scalar order for that same exact shell
  subset. Selection scans replica indices in ascending order and writes the
  first matches: `MCPA_DETAILED_CAP` defaults to `100`, `0` disables rows, and
  `-1` writes all shell replicas.

In both genealogy scopes, `replica_family_avg_sq` is the Simpson concentration
`sum_f (n_f/population)^2`; its reciprocal is the effective-family count.
All three files therefore join directly on `E`. Aggregate population equals
the main-file `nCull`, and detailed row count at each energy is
`min(MCPA_DETAILED_CAP,nCull)` (`nCull` for `-1`). A no-next-shell condition
writes no misleading data row; a successful walk has one terminal row. The
obsolete four split tables, `time.txt`, and the
per-shell packed spin-sample files are not produced. Raw byte spins remain in
memory and checkpoints, but detailed output replaces the former sample-file
explosion with bounded observables.

The CPU suite uses independent right/down bond sums rather than the production
neighbor helper. It fixes the exact `L=2` and `L=3` density of states for
`q=2,3,4`, checks the byte-spin domain, periodic geometry, full and local
energies, the affine `q=2` Ising mapping, and strict cooling/heating shell
selection. Each named contract is also registered separately with CTest.

Strict walks begin at integer sentinels outside the complete Potts spectrum.
Resampling distinguishes a missing next shell from a terminal full cull; a
full-cull row is written once and the run stops before applying an invalid
zero-survivor population.

Parent selection uses an explicit two-word PCG state instead of libc `rand()`.
The uniform-with-replacement survivor law is unchanged, while historical and
migrated seeded trajectories are intentionally not bitwise identical. Exact
state replay is tested so the generator can be persisted in checkpoints.

The CUDA suite checks initialization and equilibration against independent
reference kernels for `q=2,3,4`. It compares every spin, tracked energy,
accepted-flip count, and Philox-state byte after the exact production draw sequence, including
self-proposals, in cooling/heating open, frozen, and mixed regimes for all
three runtime `q` values. Counting accepted updates writes no RNG state and
does not add a draw. CUDA tests return skip code 77 when no GPU is accessible.

## Checkpoint/restart

Checkpointing is optional for both cooling and heating runs:

```text
main_potts seed L blocks threads nSteps q heat \
    checkpoint_dir [checkpoint_shell_interval]
```

Use `none` as the checkpoint directory to retain checkpoint-free execution.
When a directory is supplied without an interval, the default is one copied
shell. Snapshots are taken only after the synchronized replica-copy operation,
so spins, tracked energies, family labels, order, both RNG streams, and `U` all
describe the population entering the next strict shell.

Potts format `MCPA2DP3`, version 3, preserves each runtime-`q` spin as its
original byte. It stores exact integer energies, family/order arrays, `U`, the
completed-shell count, the detailed-cap identity, offsets for exactly three
live files, every opaque Philox byte, and both 64-bit words of the PCG32 state.
CRC32 covers the complete header
and payload. Save uses a synced `.tmp`, rotates `.bin` to `.prev.bin`, and keeps
both generations; restart checks current, previous, then a complete temporary
file. A `.done` marker makes a completed rerun exit successfully before opening
or changing outputs, while a set of checkpoint files with no valid generation
fails fast before opening outputs.

Loading also applies model semantics after CRC validation: parameters including
runtime `q` and heating direction must match, spins must lie in `[0,q)`, every
energy must equal an independent periodic right/down bond sum, families must be
valid, order must be a permutation, output offsets must be nonnegative, and the
PCG stream must be odd.

`./test.sh checkpoint` runs CPU round-trip, corruption/fallback, rotation,
temporary recovery, done-marker and semantic-validation contracts; direct RTX
continuations for `q=2,3,4`; and bounded real cooling and heating
SIGKILL/restarts whose three files must match uninterrupted trajectories byte
for byte. Process tests set `MCPA_DETERMINISTIC_TIMINGS=1` so timing fields are
zero for that byte-identity oracle; normal production runs write measured wall
times.
