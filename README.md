# Microcanonical Population Annealing: 2D Blume-Capel

CUDA/C++ implementation of MCPA for spin-1 Blume-Capel variables on a periodic
square lattice. The simulator keeps the exact integer energy parts
`E_J=-sum_<ij> sigma_i sigma_j` and `E_delta=sum_i sigma_i^2`, with a decimal
crystal field represented internally as a reduced rational number.

## Build

Requirements are CMake 3.22 or newer, CUDA Toolkit 12.4, and a C++17 compiler.
The default Linux test wrapper uses `/usr/local/cuda-12.4/bin/nvcc` and
`/usr/bin/g++-11`; override them with `CUDACXX` and `CXX` when needed.

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
cmake --build build --target main_bc
```

## CPU oracle tests

```bash
./test.sh cpu
./test.sh gpu
./test.sh checkpoint
./test.sh all
./test.sh list
```

Each contract prints its verdict immediately. Interactive terminals show
**PASSED** in green, **FAILED** in red, and **SKIPPED** in yellow; redirected
output stays plain. Set `MCPA_COLOR=always|never` to override automatic colour
detection. Full captured diagnostics are replayed only for a failed contract.

The named CTest cases exhaust all `3^9=19683` states of the physical `L=3`
torus and check the joint `(E_J,E_delta,M)` density of states, exact symmetry
and occupancy identities, constructed configurations, rational-D arithmetic,
local update terms, strict shell selection, proposal domain, and periodic
geometry. These tests link the production CUDA library but make no GPU calls.
The CUDA suite independently reproduces initialization-energy checks and the
two-draw Philox update loop in cooling and heating, then requires bitwise-equal
spins, energy parts, flip counts, and complete RNG state.

The checkpoint category covers the BC-specific CRC-32 format, current/previous
retention, temporary-file recovery, done markers, semantic and parameter
validation, completed-run no-op behavior, invalid-set fail-fast behavior, and
complete Philox plus PCG continuation. On a GPU it also kills
and restarts a real process, requiring aggregate/detail bytes and every main
column except nondeterministic wall-clock timing to match an uninterrupted run.

## Run

```bash
./build/main_bc seed L blocks threads nSteps heat D [checkpoint_dir] [checkpoint_hours] [detail_cap]
```

Here `heat=0` selects cooling and `heat=1` selects heating. Checkpoint hours may
be any finite nonnegative decimal. `0` saves at every completed shell, which is
useful for recovery tests; the default is 0.25 hours (15 minutes), independent
of how often a user logs into the cluster.

`detail_cap` defaults to 100. `0` disables detailed rows, `-1` writes every
replica matching the shell, and a positive value writes the deterministic
replica-index prefix and labels it `prefix_first_N`. Detailed rows retain the
joint `(E_J,E_delta,M,family_id)` evidence and report the full matching count.
Main rows include shell equilibration time and pre/post family concentration;
GPU name, compute capability, CUDA runtime/driver versions, total memory, and
free memory immediately before and after simulation GPU setup are populated
once; later rows use explicit `NA` cells.

Checkpoints use a Blume-Capel-specific versioned payload at the clean boundary
after resampling. It stores packed `-1/0/+1` spins, exact `E_J` and `E_delta`,
the rational crystal field and detail cap, genealogy and ordering, shell/output positions,
every Philox byte, and both PCG words. Resume validates CRC, parameters, spin
domain, energy parts, family/order ranges, and ordering permutation before
opening output files in append mode.

Validated runtime bounds are `L>=3`, positive blocks/threads/nSteps,
`threads<=1024`, `heat` exactly 0 or 1, and products that fit signed 32-bit
model indexing. Output directories are created recursively before files open;
the run stops before GPU allocation if arguments or output paths are invalid.

Parent selection uses an explicit PCG state rather than libc `rand()`. The
survivor set and uniform-with-replacement rule are unchanged, but seeded runs
from the historical branch are not trajectory-identical. The complete two-word
state is exposed for checkpoint persistence and has an exact replay test.

## Cluster launcher policy

Submit the complete human-approved array and let Slurm decide concurrency; do
not add an array `%N` throttle merely to imitate local one-GPU execution. Small
`L` jobs should not request a high-memory GPU type. A large Blume--Capel state
can request one only after its spin, energy-part, statistics, RNG, and
checkpoint memory estimate justifies it.

Use the shortest credible walltime because shorter jobs queue faster, while
retaining the cluster's 14-day maximum. Every tuple must use its own checkpoint
and output directory. A repeated tuple should no-op on a matching `.done`,
resume a validated current/previous/tmp generation, fail without truncating on
invalid state, and never silently replace orphan output. These scheduling rules
are shared engineering practice; they do not transfer Baxter--Wu lattice,
energy, observable, or symmetry assumptions into Blume--Capel.
