# Microcanonical Population Annealing: 1D Ising branch

The maintained CUDA simulator is `src/ising1d_simulation.cu`. It evolves Ising
spins `sigma_i in {-1,+1}` on a one-dimensional periodic ring with

```text
H = -sum_i sigma_i sigma_(i+1).
```

For a ring of `N` spins with `w` domain walls, `w` is even and
`E=-N+2w`. The exact degeneracy is `g(E)=2*C(N,w)` for even `w`.
The two uniform states are the ground states at `E=-N`.

Obsolete notebooks, figures, IDE metadata, and alternate CUDA experiments were
removed from the maintained tree; their provenance remains available in Git
history.

## Build

Requirements: CMake 3.22 or newer, CUDA Toolkit 12.4, and a C++17 host
compiler. The default Linux wrapper selects CUDA 12.4 and GCC 11.

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
cmake --build build --target main_1d_ising
```

## Exact CPU tests

```bash
./test.sh cpu
./test.sh gpu
./test.sh checkpoint
./test.sh all
./test.sh list
```

Each oracle is a separately named CTest contract. The suite compares exhaustive
and analytic densities of states for odd and even rings, checks the `2^N` state
count and twofold ground state, verifies even wall parity and `E=-N+2w`, checks
local flip deltas against full recomputation, tests periodic neighbors, and
checks strict cooling-shell selection.

Successful configure/build chatter is suppressed by `test.sh`; each selected
contract runs separately and emits its verdict immediately. Interactive
terminals show **PASSED** in green, **FAILED** in red, and **SKIPPED** in
yellow; redirected output stays plain. Set `MCPA_COLOR=always|never` to
override automatic colour detection. Full captured diagnostics are printed
only after a failure. `list` retains CTest's readable contract listing.

The CUDA suite uses small directly callable odd- and even-`N` kernel fixtures.
It checks initialization against an independent Philox reference, recomputes
every tracked energy from the ring configuration, and compares cooling spins,
energies, and every Philox state byte after the inherited one-draw-per-attempt
trajectory. A machine without an accessible NVIDIA GPU reports these tests as
CTest `Skipped` (return code 77), not failed.

## Inherited CLI and the L-to-N ambiguity

```bash
./build/main_1d_ising run_number N blocks threads nSteps
```

`main_1d_ising` is the canonical interface: `N` is the explicit number of ring
sites. All integer arguments are validated, `N>=3`, `blocks*threads` is checked
for overflow, and `threads` must not exceed 1024.

Historical scripts that supplied a linear `L` and expected `N=L^2` can use the
clearly named compatibility executable:

```bash
./build/main_1d_ising_legacy_l_squared run_number L blocks threads nSteps
```

For example, canonical `N=8` means an 8-spin ring, while compatibility `L=8`
means the inherited 64-spin ring. Neither executable represents a 2D lattice.

Both executables create `datasets/test` recursively and stop before simulation
if any output cannot be opened. Their RNG, resampling parent law, strict cooling
update, output schema, and checkpoint implementation are otherwise shared.

Parent selection uses an explicit PCG32 state seeded from `run_number`. Parents
are still drawn uniformly with replacement from the surviving replicas. This
replaces implementation-defined libc `rand()`, so runs compiled after this
change intentionally start a new seeded trajectory version; the full PCG state
is available for exact checkpoint continuation.

Cooling begins at the outside-spectrum ceiling `U=N+2`, so strict `E_new<U`
updates do not discard the positive-energy part of either the odd- or even-ring
DOS before the first measured shell. A completed or fully culled walk writes
exactly one final `culling_factor=1` row and terminates immediately.

## Output format version 3

Each run owns exactly three tab-separated files. This intentionally replaces
the historical six split tables plus `time.txt`; no silent reader compatibility
is claimed.

- `_main.txt` starts with `E`, `culling_factor`, and
  `replica_family_avg_sq`, then records exact `nCull`, full-precision culling,
  status, `nSteps`, shell equilibration seconds, and compact post-resampling
  genealogy. Rectangular GPU metadata appears on the first data row only; later
  rows contain `NA` in those columns.
- `_agg_stats.txt` records the exact zero-based shell index, selected population,
  accepted-flip sum/mean/rate, `M`, `|M|`, and `M^2` sums and means, and family
  diagnostics calculated from the pre-resampling family snapshot.
- `_detailed_stats.txt` records shell, energy, replica, pre-resampling family,
  flips, `M`, `|M|`, `M^2`, the total matching population, detailed cap, and
  literal sampling policy. Its deterministic `first_matching_replica_indices`
  policy scans replica indices in ascending order and stops after the requested
  number of members of the selected energy shell.

Set `MCPA_DETAILED_CAP` to choose the maximum matching rows per shell: the
default is `100`, `0` writes only the header, and `-1` includes all matches.
Other negative values and
malformed integers fail before output creation. The cap is part of checkpoint
identity and cannot change during a resumed run.

## Checkpoint/restart

Checkpointing is opt-in and uses a deterministic copied-shell interval:

```bash
./build/main_1d_ising run_number N blocks threads nSteps checkpoint_dir interval
./build/main_1d_ising_legacy_l_squared run_number L blocks threads nSteps checkpoint_dir interval
```

Passing `none` in place of `checkpoint_dir` keeps checkpointing disabled. The
default interval is one copied shell when a directory is supplied without the
final argument. A snapshot is written only after `updateReplicas` has completed,
at the population boundary entering the next cooling shell.

Format version `MCPA1DI1/3` stores one bit per validated `{-1,+1}` spin, exact
integer energies, family labels and energy order, the current `U`, completed
shell count, detailed cap, offsets for exactly three live output files, the entire opaque
Philox byte array, and both 64-bit PCG32 state words. CRC32 covers the header and full
payload. Each save is synced through `.tmp`, then rotates `.bin` to
`.prev.bin`; loading tries current, previous, then a complete temporary file.
An invalid checkpoint set fails fast. After clean completion, a `.done` marker
makes later invocations exit successfully without opening, changing, or
restarting the completed output files.

`./test.sh checkpoint` runs the CPU round-trip, CRC corruption, generation
rotation, temporary-file recovery and done-marker contracts plus a direct CUDA
continuation test. With an accessible GPU it also kills a real simulator after
a committed checkpoint, rejects a changed detailed cap without touching outputs,
and resumes it. Aggregate and detailed files and every deterministic main-table
field must match the uninterrupted run. `equilibrate_seconds` and one-time GPU
free-memory observations are intentionally not byte-stable across processes.
The test also proves completed reruns are no-ops and invalid current, previous,
and temporary generations preserve all three outputs.
