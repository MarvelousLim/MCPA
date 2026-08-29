# Stage 5: precomputed equilibration neighbor table

Decision: **accept as an opt-in, trajectory-preserving local-GPU policy**.  The
CMake switch `MCPA_BW_NEIGHBOR_TABLE` remains `OFF` by default, so production
behavior does not change automatically.  On the local RTX 4060 mixed workload,
two order-reversed comparisons reproduced gains above the 2% gate in cooling
and heating.

`block_sweep` was not evaluated.  Structured site traversal changes attempted
site order and therefore violates the current physics hard stop even if its
marginal site frequencies look similar.

## Candidate

The baseline calls `SLF(j, params)` at every randomly selected site.  That
function computes six periodic neighbor indices, including integer division and
remainder operations.  For a fixed lattice these six indices never change.

The candidate allocates one device table of `N` `neiborsIndexes` records and
populates entry `j` by calling the existing `SLF(j, params)` exactly once.  The
equilibration loop then loads the same six indices from that read-only table.
The table is cached for the current `(L,N)` and rebuilt if the lattice changes.
It occupies `24N` bytes: 31,104 bytes for the benchmark's `L=36` lattice and
about 3.0 MiB for `L=363`.

This path exists only when configured with:

```text
-DMCPA_BW_GROUPED_ATTEMPTS=OFF
-DMCPA_BW_NEIGHBOR_TABLE=ON
```

CMake rejects simultaneous grouped-attempt and neighbor-table builds so the
two mechanisms cannot contaminate each other's measurements.  With both
options `OFF`, the original Stage-1 data structure and kernel are compiled.

The table is attached only to the by-value device descriptor passed to
`equilibrate`.  Full energy recomputation, statistics, Fourier diagnostics,
resampling, checkpoints, and output continue to call their existing code and
do not consult the table.

## Trajectory-equivalence gate

The opt-in build passed the CPU suite and the complete CUDA suite.  The CUDA
reference comparison covers cooling and heating under open/all-accepted,
frozen/all-rejected, and mixed conditions.  It required byte-for-byte equality
of:

- all spins;
- tracked energies;
- the complete `replicaStatistics` structure, including flip count and the
  newly added diagnostic fields;
- stored Philox state.

All six direction/regime cases passed, as did the separate diagnostic kernels
and the partial-`curand4` tail test.  Thus the candidate retains the random
stream and consumption, attempted sites and order, acceptance decisions, and
all resulting state.

## Timing protocol

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- Driver: 610.47
- CUDA compiler: 12.4.131; host compiler: GCC 11.4.0
- Build: `RelWithDebInfo`, CUDA architecture setting `70`
- Workload: `L=36`, `R=131072`, `nSteps=2`, 512 threads per block
- Regime: deterministic mixed fixture
- Each invocation: correctness check, 3 warmups, then 10 timed samples
- Runs local, sequential, and independently capped at 55 seconds

The baseline and candidate were separate builds of the same worktree.  The
baseline had grouped attempts and the neighbor table disabled.  Correctness
initializes the candidate's table before warmups and timing; the one-time table
construction is therefore excluded from kernel throughput, as intended for a
table reused over an entire MCPA walk.

The second comparison reversed binary order to check warm-state bias.  Speedup
is baseline kernel median divided by neighbor-table kernel median.

| direction | batch order | acceptance | baseline median (ms) | table median (ms) | baseline MAD (ms) | table MAD (ms) | speedup |
|:---|:---|---:|---:|---:|---:|---:|---:|
| cooling | baseline, table | 0.429767 | 436.846573 | 426.368332 | 0.301804 | 0.120132 | 1.024576x |
| cooling | table, baseline | 0.429767 | 436.527649 | 426.409271 | 1.276611 | 0.188110 | 1.023729x |
| heating | baseline, table | 0.487604 | 439.193954 | 429.536774 | 0.312546 | 0.201324 | 1.022483x |
| heating | table, baseline | 0.487604 | 439.027023 | 429.293915 | 0.525650 | 0.306152 | 1.022672x |

Averaging the two batch medians gives 436.687111 ms versus 426.388801 ms
for cooling, a `1.024152x` throughput speedup.  Heating gives 439.110488 ms
versus 429.415345 ms, a `1.022578x` speedup.  Acceptance is identical between
builds and across repeated batches.

Open and frozen timing runs were not added.  Strict trajectory tests already
cover both endpoints, while the relevant mixed regime cleared the gate in both
directions and reproduced under reversed run order.  The bounded search stopped
there.

## Decision

**Accept the neighbor table as a reproducible opt-in policy on the local RTX
4060.**  Its 2.3--2.4% mixed-workload gain is modest but appears in both
directions, exceeds the predefined gate, and is well separated from within-run
dispersion.  Keep `MCPA_BW_NEIGHBOR_TABLE=OFF` as the production default until
an explicit production rollout is requested.  Other GPUs require their own
short confirmation.

No simulation outputs or existing datasets were written, and no cluster work
was submitted.  Population size, `nSteps`, measurements, diagnostics,
resampling, and output formats were unchanged.
