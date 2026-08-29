# Stage 1: register state and synchronization cleanup

Decision: **accepted**. This is the production equilibration baseline for later
experiments.

## Baseline behavior and suspected bottleneck

Each GPU thread evolves one Baxter--Wu replica. For every attempted spin flip,
the original source repeatedly addressed the replica energy and Philox random
state through device-memory arrays. It also wrote the running flip count during
the loop. After the kernel launch, the wrapper contained more than one full-GPU
synchronization around its error check.

In physical terms, none of those operations creates new Monte Carlo samples;
they are bookkeeping and launch coordination. In hardware terms, repeated
state traffic and redundant barriers can consume time even though the local
spin update is unchanged.

## Exact code-level idea

The candidate made two related bookkeeping changes:

1. Load the replica energy and Philox state into thread-local variables before
   the update loop, keep the flip count local, and write all three back once
   after the loop.
2. Keep one launch-error check and one completion barrier, but remove the
   duplicate wrapper synchronization.

No spin representation, neighbor lookup, or Monte Carlo rule was changed.

## Why it might help

Thread-local scalar state normally resides in GPU registers, which are much
closer to the executing thread than device memory. Reusing those values avoids
source-level array accesses on every attempt. Removing a redundant global
barrier also avoids making the CPU wait twice for the same completed work.

## What stayed scientifically identical

The candidate retained all parts that define the stochastic process:

- the same Philox seed, state, and `curand4` consumption schedule;
- the same random site from the same random integer;
- exactly `N * nSteps` attempted flips per replica, in the same order;
- the same local-energy calculation and strict microcanonical inequality;
- the same accepted spin flips, tracked energy, and flip count;
- the same heating/cooling choice, population, resampling, and output logic.

This is therefore intended as an implementation optimization, not a modified
Monte Carlo algorithm.

## Benchmark fixtures

Stage 1 used two deliberately extreme fixtures starting from a ground-state
branch:

- **Open:** the energy boundary lies outside the accessible spectrum, so every
  proposed flip is accepted. This stresses energy/spin writes.
- **Frozen:** the boundary rejects every proposed flip. This measures the loop
  when it performs the proposal calculation but does not update spins.
- **Mixed:** accepted and rejected proposals coexist. This fixture was added
  only after Stage 1 and was not part of the Stage-1 timing claim.

Open and frozen are diagnostic ceilings, not equilibrium ensembles.

## Correctness gate

The CUDA bitwise comparison passed before timing. Starting from the same state
and random stream, the reference and candidate produced identical spins,
tracked energies, replica statistics, and Philox state. Bitwise equality is the
appropriate gate here: a merely close answer could hide a changed trajectory.

## Timing result

The local smoke benchmark used an **NVIDIA GeForce RTX 4060 Laptop GPU** with
`L=36`, `R=8192`, `nSteps=1`, 256 threads, and 7 timed repeats.

| regime | reference median | candidate median | speedup |
|---|---:|---:|---:|
| open | 7.271584 ms | 5.718016 ms | 1.271697x (27.17%) |
| frozen | 4.542720 ms | 3.867936 ms | 1.174456x (17.45%) |

For the open fixture, the candidate saved 1.553568 ms per timed kernel, making
the elapsed kernel time about 21% shorter and the attempted-update throughput
27.17% higher. For the frozen fixture, it saved 0.674784 ms, making kernel time
about 15% shorter and throughput 17.45% higher.

Speedup is `reference / candidate`; the percentage in parentheses is throughput
improvement, `(speedup - 1) * 100`, rather than percent reduction in elapsed
time.

## Decision

**Accept.** The gain is substantial in both endpoint workloads, and the strict
bitwise gate shows that it did not alter the sampled trajectory. These numbers
establish a local Stage-1 baseline; they are not a cross-node or cross-GPU
performance claim.
