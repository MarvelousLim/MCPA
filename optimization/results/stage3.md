# Stage 3: GPU block-size experiment

Decision: **use 512 threads per block for production-shaped runs on the local
RTX 4060 Laptop GPU**. This is a run-configuration result, not a source-code or
Monte Carlo algorithm change. A different cluster GPU should be checked
separately before adopting the same setting there.

## What was varied

One GPU thread evolves one replica. Threads are scheduled in groups called
blocks. The accepted Stage-1 kernel was launched with 128, 256, or 512 threads
per block; the number of blocks was recomputed so the replica population stayed
exactly `R=131072`:

| threads per block | blocks | replicas |
|---:|---:|---:|
| 128 | 1024 | 131072 |
| 256 | 512 | 131072 |
| 512 | 256 | 131072 |

No production source was edited. Block size changes how the GPU schedules the
same independent replica threads. It does not change which thread owns a
replica or the work performed by that thread.

## Why block size might matter

A block contains an integer number of 32-thread warps. Changing its size can
change how many blocks and warps reside on a streaming multiprocessor at once,
which in turn changes the GPU's ability to hide memory and instruction latency.
Larger is not automatically better: register use, block limits, and the GPU
model can reverse the result. This therefore has to be measured rather than
inferred from the thread count alone.

## What stayed scientifically identical

Every comparison used the same accepted Stage-1 kernel, seed, fixture,
population, lattice, and equilibration length. In particular, the experiment
kept:

- `L=36`, `N=1296`, `R=131072`, and `nSteps=2`;
- 339738624 attempted flips per timed kernel;
- the same Philox streams and random-number consumption;
- the same attempted-spin order within every replica;
- the same local-energy calculation and acceptance inequality;
- the same accepted flips, tracked energies, and replica statistics;
- all resampling and output code untouched.

This stage changes launch geometry only; it does not change equilibration or
the physical experiment.

## Benchmark fixture

The primary workload was the deterministic **mixed** regime, in which both
accepted and rejected proposals occur:

- Cooling used isolated spacing-three defects within every replica. Its
  measured acceptance was 0.429767 (42.9767%).
- Heating used a synthetic aggregate mixture of frozen ground-state and open
  ceiling-state replicas. Its measured acceptance was 0.487604 (48.7604%).
  This is useful for accepted/rejected GPU traffic, but it is not a critical
  ensemble and says nothing about physical branch mixing.

Open (all accepted) and frozen (all rejected) endpoint fixtures were not added.
The mixed result was already clear in both directions and reproduced in the
alternating confirmation, so more GPU runs were not needed for this bounded
decision.

## Correctness gate

The benchmark's correctness comparison remained enabled in every invocation.
Before timing, it ran the candidate and independent readable reference from the
same spins and Philox state, then required byte-for-byte equality of spins,
tracked energies, complete replica statistics, and Philox state. All ten
benchmark invocations passed. No `--skip-check` run was used.

Acceptance was identical at every block size, providing an additional simple
check that launch geometry did not change the update process.

## Hardware and build

- GPU: **NVIDIA GeForce RTX 4060 Laptop GPU**, 8188 MiB
- Driver: 610.47
- Initial state: 0 MiB reported used, 0% utilization, 41 degrees Celsius
- CUDA compiler: CUDA 12.4 (`Build cuda_12.4.r12.4/compiler.34097967_0`)
- Host compiler: GCC 11.4.0
- Build: `RelWithDebInfo`, CMake CUDA architecture setting `70`
- Benchmark preset: `production`
- Seed: 173
- Timed samples per invocation: 10, after 3 warmups
- Hard command timeout: 55 seconds; no invocation exceeded it

All runs were local and sequential. No cluster work was submitted.

## Initial three-size sweep

The table reports the CUDA-event kernel median, median absolute deviation
(MAD), p10--p90 interval, attempted-update throughput, and acceptance from each
ten-sample invocation.

| direction | threads | median (ms) | MAD (ms) | p10--p90 (ms) | attempts/s | acceptance |
|:---|---:|---:|---:|:---|---:|---:|
| cooling | 128 | 452.921371 | 0.127975 | 452.634326--453.049960 | 750105085.359244 | 0.429767 |
| cooling | 256 | 453.377045 | 0.156647 | 453.061050--453.536777 | 749351181.292141 | 0.429767 |
| cooling | 512 | 435.557358 | 0.209000 | 435.261652--435.947729 | 780008919.434429 | 0.429767 |
| heating | 128 | 458.095001 | 0.094269 | 457.971198--458.425415 | 741633554.382138 | 0.487604 |
| heating | 256 | 458.511948 | 0.090988 | 458.417194--458.673880 | 740959152.219943 | 0.487604 |
| heating | 512 | 439.212158 | 0.344177 | 438.905746--440.611105 | 773518259.125420 | 0.487604 |

The 128- and 256-thread results are practically indistinguishable. The
512-thread result is visibly faster in both directions.

## Alternating confirmation

Because the first sweep ran 512 threads last, the follow-up alternated back to
256 and then to 512 after the GPU was already active. The same separation
remained:

| direction | threads | median (ms) | MAD (ms) | p10--p90 (ms) | attempts/s | acceptance |
|:---|---:|---:|---:|:---|---:|---:|
| cooling | 256 | 453.282089 | 0.070297 | 453.156940--453.405920 | 749508158.538923 | 0.429767 |
| cooling | 512 | 435.630066 | 0.556702 | 435.253732--436.642435 | 779878733.310328 | 0.429767 |
| heating | 256 | 458.308960 | 0.167847 | 458.117603--458.509436 | 741287327.284539 | 0.487604 |
| heating | 512 | 439.425079 | 0.441284 | 439.012573--440.229282 | 773143454.865765 | 0.487604 |

Averaging the two batch medians gives 453.329567 ms versus 435.593712 ms for
cooling, a `1.040717x` throughput speedup (4.0717%) and a 3.912% reduction in
kernel time. Heating gives 458.410454 ms versus 439.318618 ms, a `1.043458x`
throughput speedup (4.3458%) and a 4.165% reduction in kernel time.

Relative to the single 128-thread batches, the corresponding 512-thread
speedups are `1.039779x` for cooling and `1.042740x` for heating.

## Decision

**Accept 512 threads as the local RTX 4060 setting for this production-shaped
workload.** The roughly 4% gain appears in both cooling and heating, is much
larger than the within-batch dispersion, survives the alternating check, and
passes the strict trajectory-equivalence gate.

This does not justify hard-coding 512 into the production kernel or assuming it
is optimal on cluster nodes. Thread count should remain a run parameter. The
cluster can retain its existing setting until a short, hardware-local check is
explicitly requested.
