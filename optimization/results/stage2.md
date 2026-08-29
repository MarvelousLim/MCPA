# Stage 2: heating/cooling specialization

Decision: **rejected as neutral**. The accepted stage-1 kernel remains the
production implementation.

## Baseline behavior and suspected bottleneck

After Stage 1, each thread already kept its energy, flip count, and Philox state
locally. One small run-time choice remained inside every attempted update: use
`suggested_energy < U` during cooling or `suggested_energy > U` during heating.

The direction is the same for every replica in one program launch. It therefore
does not split a warp into different heating and cooling paths, but the compiled
kernel can still contain a direction load and conditional selection on every
attempt. Stage 2 tested whether removing that small repeated decision was worth
maintaining two compiled kernels.

## Exact code-level idea

The temporary candidate compiled separate cooling and heating versions of the
same equilibration loop. The cooling version contained only the `< U` test and
the heating version only the `> U` test. The host wrapper selected the version
once, before launching the GPU kernel.

The experiment was option-gated so the accepted Stage-1 dynamic kernel and the
specialized candidate could be built from the same source tree and compared as
separate binaries. The specialization and its build switch were removed after
the negative timing decision.

## Why it might help

Specialization lets the compiler remove a direction-dependent operation from a
very frequently repeated loop. The plausible gain was modest because the
direction was already launch-uniform: this was not a cure for warp divergence,
only a test of whether eliminating a small amount of per-attempt control work
measurably improved throughput.

## What stayed scientifically identical

Only the expression of the already chosen inequality changed. Both builds kept:

- the same Philox states and `curand4` draw schedule;
- the same random sites and attempted-spin order;
- the same number of attempts per replica;
- the same local energies and strict heating/cooling boundaries;
- the same accepted flips, energy tracking, and statistics;
- the same equilibration length, population, resampling, and output code.

Thus a valid candidate had to reproduce the exact trajectory, not merely the
same averages.

## Benchmark fixtures

The benchmark distinguishes three acceptance behaviors:

- **Open:** the boundary is outside the physical energy spectrum in the
  accepting direction, so every proposal is accepted.
- **Frozen:** the boundary rejects every proposal.
- **Mixed:** both accepted and rejected proposals occur, and the benchmark
  requires the aggregate acceptance fraction to remain between 5% and 95%.

Open and frozen are synthetic endpoint workloads rather than thermodynamic
ensembles. The retained cooling-mixed fixture starts from a ground-state branch
with isolated spacing-three defects. Its energy is checked as `-2N/3`, and
accepted and rejected proposals coexist within each replica. It gave 42.98%
acceptance.

The retained heating-mixed fixture is more artificial: it alternates frozen
ground-state replicas with open, globally inverted ceiling-state replicas and
uses `U=0`. Its 48.76% acceptance is an aggregate hardware workload. It is not
a critical ensemble, does not demonstrate within-replica mixing, and is not
evidence about the physical branch-trapping problem. An earlier heating fixture
was rejected before measurement because it accepted 99.97% of proposals; the
retained fixture explicitly satisfies the 5--95% gate.

## Correctness gate

Both option states passed the full CPU/CUDA test suite. For cooling and heating,
the CUDA test exercised open, frozen, and mixed regimes against an independent,
readable reference update loop. It required bitwise equality of:

- every spin;
- every tracked replica energy;
- the complete replica-statistics records;
- every Philox state byte.

The tests also checked that open accepted all attempts, frozen accepted none,
and mixed stayed inside the required 5--95% interval. Timing began only after
these gates passed.

## Timing result

The benchmark used an **NVIDIA GeForce RTX 4060 Laptop GPU**, CUDA 12.4.131,
GCC 11.4, `L=36`, `R=131072`, two sweeps, 256 threads, ten repetitions per
batch, and three alternating dynamic/specialized batches.

| direction | dynamic batch median (ms) | specialized batch median (ms) | speedup | throughput gain |
|:---|---:|---:|---:|---:|
| cooling | 525.644928 | 524.845551 | 1.00152x | 0.152% |
| heating | 531.878784 | 531.125763 | 1.00142x | 0.142% |

In plain terms, specialization saved about 0.80 ms from a roughly 526 ms
cooling kernel and about 0.75 ms from a roughly 532 ms heating kernel. That is
only about one to two parts in a thousand, small enough to be practically
neutral for this code.

The individual batch medians (ms), in execution order within each direction,
were `521.903992, 526.001312, 525.644928` (cooling dynamic),
`524.434418, 524.845551, 525.238159` (cooling specialized),
`531.877411, 531.968506, 531.878784` (heating dynamic), and
`531.125763, 531.189270, 530.595276` (heating specialized).

No spill warning was emitted. Raw alternating-batch measurements are in
`stage2_heat_specialization.jsonl`.

## Decision

**Reject as neutral.** The gains of 0.152% for cooling and 0.142% for heating
are far below the predeclared 2% practical threshold. Removing a launch-uniform
direction test is not worth the extra template, host dispatch, and build-policy
complexity.

The mixed benchmark and broader bitwise tests are retained because they
strengthen all later optimization experiments. The accepted Stage-1 kernel
remains the production implementation.
