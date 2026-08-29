# Stage 4: explicit groups of four attempts

Decision: **reject as neutral**.  Keep the accepted Stage-1 loop as the
production default.  The grouped implementation remains available only through
the opt-in CMake switch `MCPA_BW_GROUPED_ATTEMPTS=ON`; the switch defaults to
`OFF`, so this experiment does not change production behavior.

## Candidate

The Stage-1 kernel requests four Philox values at a time but iterates over
individual attempts, selecting a cached value with `k & 3` and requesting a new
vector when that expression is zero.  Stage 4 tested whether expressing the
same work as an outer `curand4` loop and an explicitly unrolled, ordered inner
loop of four attempts removes useful loop-control work.

The last group is handled explicitly.  If the attempt count is not divisible
by four, the kernel still consumes one complete `curand4` result and uses only
the required leading values.  This matches the historical Philox-state advance.
Within every group, lanes are processed in order `0,1,2,3`; a later attempt
therefore sees every accepted spin and energy change from earlier attempts.

The independent readable reference kernel was not changed.  The production
source selects the experiment at compile time:

```text
-DMCPA_BW_GROUPED_ATTEMPTS=ON   # experiment
-DMCPA_BW_GROUPED_ATTEMPTS=OFF  # production default
```

## Scientific invariants and correctness

The candidate retains the same lattice, population, sweep count, seed, Philox
streams, number of random vectors, random sites, attempted-site order,
acceptance inequalities, spin changes, tracked energies, flip counts, and
stored RNG state.  Resampling, measurements, diagnostics, and output were not
modified.

The grouped build passed the CPU suite and full CUDA suite.  The CUDA trajectory
test compares the candidate against the independent reference for all six
direction/regime cases:

- cooling and heating;
- open/all-accepted, frozen/all-rejected, and mixed acceptance;
- byte-for-byte spins, energies, complete replica statistics, and Philox state.

All six passed.  A new `L=3`, `nSteps=1` test performs nine attempts per replica,
so its last `curand4` group uses one value and discards three.  It passed in both
directions, explicitly covering tail consumption.  The pre-existing diagnostic
tests also passed.

## Timing protocol

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- Driver: 610.47
- Initial GPU state: 0 MiB used, 0% utilization, 41 degrees Celsius
- CUDA compiler: 12.4.131; host compiler: GCC 11.4.0
- Build: `RelWithDebInfo`, CUDA architecture setting `70`
- Workload: `L=36`, `R=131072`, `nSteps=2`, 512 threads per block
- Each result: 10 timed samples after 3 warmups
- Correctness comparison enabled in every invocation
- Runs local and sequential; each GPU command capped at 55 seconds

The performance baseline was a separately compiled default/Stage-1 binary, not
the deliberately readable reference kernel used as the correctness oracle.
Speedup below is baseline median divided by grouped median.

| direction | regime | acceptance | baseline median (ms) | grouped median (ms) | baseline MAD (ms) | grouped MAD (ms) | speedup |
|:---|:---|---:|---:|---:|---:|---:|---:|
| cooling | open | 1.000000 | 518.247498 | 518.217560 | 0.648193 | 0.180389 | 1.000058x |
| cooling | frozen | 0.000000 | 357.885239 | 353.739883 | 0.495102 | 0.340225 | 1.011719x |
| cooling | mixed | 0.429767 | 435.527206 | 433.371033 | 0.305588 | 0.390045 | 1.004975x |
| heating | open | 1.000000 | 518.046661 | 517.994476 | 0.901031 | 0.219208 | 1.000101x |

Cooling covered every acceptance endpoint and the relevant mixed workload.
Heating open independently reproduced the neutral result.  A heating frozen
baseline batch was started (357.527924 ms median), but the grouped partner and
the remaining heating mixed pair were deliberately not run: the completed
evidence was already below the 2% acceptance gate, so the bounded-stop rule
ended the experiment rather than accumulating redundant GPU repetitions.

## Decision

The open workload is unchanged within roughly 0.01%.  Cooling mixed improves by
about 0.50%, and the most favorable completed case, cooling frozen, improves by
about 1.17%.  None reaches the required 2% threshold, and the mixed result is
small relative to the maintenance cost of a second loop body.

**Reject grouped attempts as a production optimization.**  The opt-in switch
is retained to make the negative experiment reproducible, but remains `OFF` by
default.  No resampling, `nSteps`, `R`, measurement, diagnostic, output, or
cluster setting changed.
