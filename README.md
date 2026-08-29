# Microcanonical Population Annealing (MCPA)

A computational physics project implementing MCPA. This checkout and README
describe the active 2D Baxter--Wu branch; other model branches retain distinct
spin, energy, geometry, and output contracts and are not interchangeable.

For the map connecting simulator source, local experiment directories, analysis,
and the article workbench, start at
`../Analytics/2DBaxterWu/README.md#project-map--start-here`.

Cross-model consolidation is being prepared outside this repository. No
wholesale branch merge is planned; every model keeps an explicit physics and
representation contract.

## Authors

PhD work with Lev Shchur, Natan Rose, and John Machta.

## Overview

This project simulates the 2D Baxter-Wu model using Microcanonical Population Annealing (MCPA), a powerful algorithm for equilibrium and nonequilibrium systems. MCPA combines a population of replicas with sequential microcanonical energy ceilings.

The algorithm maintains a population of R replicas and progressively lowers (or raises) the energy threshold U to sample the phase space. At each temperature step, replicas are equilibrated via local spin-flip updates, then resampled based on their Boltzmann weights.

## Algorithm

### Baxter-Wu Model

The 2D Baxter-Wu model is defined on an L x L square lattice with periodic boundary conditions. Each site (i,j) has 6 triangular interactions with its neighbors. The energy of spin s_i with neighbors is:

```
E = -s_i * (s_diag_left * s_up + s_diag_left * s_left +
           s_diag_right * s_down + s_diag_right * s_right +
           s_down * s_left + s_up * s_right)
```

Spins are ±1 and the system exhibits a continuous phase transition.

### MCPA procedure

1. Initialize R replicas with random spin configurations
2. For each energy threshold U:
   - Equilibrate all replicas via Metropolis spin flips
   - Compute culling factor X (fraction to resample)
   - Resample: duplicate low-energy replicas, discard high-energy ones
   - Compute statistics: magnetization, polarization, family size
3. Adjust U and repeat until convergence

### Key Equations

**Local Energy:**
```
E_loc(s) = -s * (n_diag_left*n_up + n_diag_left*n_left +
              n_diag_right*n_down + n_diag_right*n_right +
              n_down*n_left + n_up*n_right)
```

**Culling Factor:**
```
X = nCull / R
```
where nCull is the number of replicas below energy threshold U.

**Family Size Parameter:**
```
rho_t = (1/R^2) * sum_i n_i^2
```
where n_i is the size of family i.

## Building

### Prerequisites

- CMake 3.22 or newer
- CUDA Toolkit 12.4 (`/usr/local/cuda-12.4/bin/nvcc` by default)
- GCC 11 (`/usr/bin/g++-11` by default)
- An NVIDIA GPU for CUDA tests and simulations; CPU logic tests compile and run
  without GPU access

### Tests and build

```bash
./test.sh          # named correctness checks, then an informational GPU benchmark
./test.sh correctness # correctness only, without timing
./test.sh cpu      # fast CPU suite only
./test.sh gpu      # GPU suite; reported as Skipped when no GPU is accessible
./test.sh checkpoint # checkpoint integrity, recovery boundaries, and resume
./test.sh benchmark smoke --regime mixed # timing + bitwise gate only
./test.sh list     # show registered CTest cases
```

Every registered contract prints one verdict as soon as it finishes. Verdicts
are coloured in an interactive terminal (green **PASSED**, red **FAILED**, and
yellow **SKIPPED**) but remain plain text when redirected; use
`MCPA_COLOR=always|never` to override detection. Failure diagnostics are shown
only for the contract that failed.

Override `MCPA_BUILD_DIR`, `MCPA_CUDA_ARCHITECTURES`, or `MCPA_BUILD_JOBS` when
needed. The defaults select CUDA 12.4, GCC 11, architecture 70, and two build
jobs. To build the simulation executable explicitly:

```bash
cmake --build build --target main_bw
```

### HSE cluster build and run

The MCPA checkout is the complete cluster bundle. Its required layout is:

```text
baxter_wu/
├── CMakeLists.txt
├── main/
├── lib/
├── tests/
├── scripts/
├── build/          generated and ignored
└── experiments/    generated cluster campaigns and ignored
```

There is no `Analytics/` dependency on the cluster. From `baxter_wu/` run:

```bash
bash scripts/compile.sh
bash scripts/cluster_smoke.sh --submit
# only after SMOKE PASS:
MCPA_R=32768 bash scripts/hse_meta_2DBaxterWu.sh --map
sbatch --export=ALL,MCPA_R=32768 scripts/hse_meta_2DBaxterWu.sh
```

Submit from the repository root or its `scripts/` child. `sbatch` executes a
private copy under `/var/spool/slurm`, so the batch scripts use
`SLURM_SUBMIT_DIR` as the repository anchor and resolve `build/` and
`experiments/` within that checkout. For an unusual submission directory, use
`--export=ALL,MCPA_REPO_DIR=/absolute/path/to/baxter_wu`. Both launchers accept
`--paths` to show the resolved paths without submitting anything.

`compile.sh` submits nothing. On the current HSE environment it switches the
default `gnu8` module to `gnu12/12.1`, then loads CUDA 12.4 and CMake 3.31.8;
the preferred local reference remains CUDA 12.4 with GCC 11. The Slurm scripts
leave memory selection to the `rocky` partition because explicit `--mem`
requests are rejected by the available GPU-node configuration.

The completed planning pilot used `L=18,36`, seeds `1,2`, both directions,
`nSteps=10`, and the human-selected `R=32768`. The current script is the next
size campaign: four tasks for one size, seeds `1,2`, and both directions. Its
default is `L=162`; `MCPA_L` may explicitly select the later `243` or `363`
stage. It records a deterministic hash sample of at most 1000 exact-shell
replicas by default. `MCPA_R` remains mandatory and is never selected or changed
by the script; it must be positive and divisible by the fixed 512-thread launch
width. `--map` prints all four mappings without running a GPU. The array has no
concurrency throttle: Slurm assigns one requested GPU to each task and decides
when tasks can run.

The V100 pilot took a mean 12.16 s at `L=18` and 229.27 s at `L=36`. At fixed
`R` and `nSteps` this is an empirical exponent 4.24, consistent enough with the
planning law `time ~ L^4 R nSteps` to use it as a first extrapolation. Anchoring
the fixed-exponent model to both pilot sizes predicts about 24 h for `L=162`,
5.1 d for `L=243`, and 25 d total for `L=363` at `R=32768,nSteps=10` on the
same V100 class. The default two-day request is therefore for `L=162`. Use a
seven-day request for the later stages:

```bash
sbatch --time=7-00:00:00 --export=ALL,MCPA_L=243,MCPA_R=32768 scripts/hse_meta_2DBaxterWu.sh
sbatch --constraint=type_e --time=7-00:00:00 --export=ALL,MCPA_L=363,MCPA_R=32768 scripts/hse_meta_2DBaxterWu.sh
```

`L=363` is expected to need roughly four seven-day allocations. Resubmit the
identical command only after each allocation ends: valid 15-minute checkpoints
resume the unfinished task, while `.done` makes already completed array elements
exit without launching the executable. Checkpointing makes a run longer than the
14-day Slurm ceiling possible; it does not reduce total GPU time. These estimates
must be recalibrated from `L=162` before treating the 243/363 numbers as firm.
At `L=363,R=32768`, the dominant 32-bit spin allocation is 16.09 GiB and the
remaining persistent device arrays add only about 10 MiB. A V100 32 GB should
therefore fit, but the production command deliberately constrains this largest
size to cHARISMa `type_e` nodes (A100 80 GB) for a larger safety margin. Do not
copy that constraint to the smaller-size arrays unless a measured allocation
requires it, because an unnecessary feature constraint narrows scheduling.

Every parameter tuple has an isolated run and checkpoint directory. On rerun,
a `.done` marker makes the launcher exit before `srun`, leaving outputs
unchanged after confirming that all three tables are nonempty. The detailed
limit is part of the run-directory name, so changing it cannot accidentally
reuse another output/checkpoint set. Current, previous, or temporary checkpoint
candidates are delegated to `main_bw`: a valid generation resumes after
truncating outputs to its saved offsets, while an invalid set fails without
modifying them. Outputs without a checkpoint are never silently overwritten;
the launcher stops unless the operator explicitly sets
`MCPA_ALLOW_FRESH_RESTART=1`.

## Usage

### Running the Simulation

```bash
./build/main_bw <seed> <L> <blocks> <threads> <nSteps> <heat> \
  [checkpoint_dir|none] [checkpoint_hours] [detailed_limit] \
  [frozen_nsteps_schedule|none]
```

`detailed_limit` defaults to `1000`; use `0` for a header-only detailed file,
`-1` for every replica matching an energy shell, or a positive value for that
many deterministic bottom-hash samples. Omitting `checkpoint_dir` enables
checkpointing in `./checkpoints` with a 15-minute interval. Passing literal
`none` is the only way to disable checkpointing and its completed-run marker.

Arguments:
- `seed`: Random seed for reproducibility
- `L`: Lattice dimension (N = L x L)
- `blocks`: Number of GPU blocks (replicas = blocks x threads)
- `threads`: Number of GPU threads per block
- `nSteps`: Number of Monte Carlo steps per replica per iteration
- `heat`: 0 for annealing (E decreasing), 1 for heating (E increasing)
- `checkpoint_dir|none`: `none` disables checkpoint creation and resume. A
  directory path enables them there; omitting the argument enables them in
  `./checkpoints`.
- `checkpoint_hours`: nonnegative save interval in hours when checkpointing is
  enabled; decimals are accepted and the default is 0.25. Zero checkpoints every
  clean shell and is intended for the crash-integration test, not production.
- `frozen_nsteps_schedule`: optional two-column `ceiling nSteps` table. It is
  an experimental, predeclared schedule applied equally to every replica at a
  given ceiling. Scheduled runs currently require `checkpoint_dir=none`; this
  prevents a schedule change from being hidden inside a resumed trajectory.

Checkpoint format 6 stores the aligned post-resampling population, energies,
family labels, sorting order, every Philox state, and the explicit host
resampling RNG. CRC-32 protects the complete payload; atomic rotation retains
current and previous generations and can recover a completed temporary file.
Cooling and heating tests require an interrupted continuation to match the
uninterrupted physical and RNG trajectory exactly. Timing and live GPU-memory
measurements are not trajectory state and may differ after restart.
Matching `.done` runs exit before opening outputs. If candidate generations
exist but none validate, the executable fails and preserves the existing
tables instead of silently starting a fresh trajectory.

Format-5 and older checkpoints are intentionally rejected. The explicit host
resampling generator also replaces libc `rand()`, so newly compiled runs do not
reproduce the historical parent-selection sequence even with the same seed;
the resampling rule itself is unchanged and new checkpoints preserve its full
state.

Example:
```bash
./build/main_bw 42 18 1 32 10 0 none
```
Runs on an 18x18 lattice with 32 replicas, 10 MC steps, cooling. Baxter--Wu
lattice sizes must be divisible by three.

### Performance experiments

`./test.sh benchmark smoke` runs a short bitwise-checked equilibration timing.
`./benchmark.sh` remains as a compatibility wrapper for existing experiment scripts.
Timing is informational: performance has no stable pass/fail threshold, while a
trajectory mismatch or invalid benchmark fixture is still a failure.
Use `./optimization/run.sh NAME candidate smoke` to record an experiment and
`./optimization/profile.sh NAME ncu smoke candidate` for a guided Nsight report.
See `optimization/README.md` for the experiment contract and preset ledger.

## Output Files

Simulation produces three output files per run:
- `{prefix}_main.txt` - entropy-walk/culling data, exact culling count, timing,
  compact post-resampling genealogy metrics, one run-level GPU record, and the
  actual equilibration ceiling, sweep count, all-population acceptance ratio,
  and fixed/frozen schedule policy
- `{prefix}_agg_stats.txt` - all-population moments, ordered-branch counts,
  pre-resampling family-cluster diagnostics, and three branch-invariant
  minimum-wavevector structure factors for reweighted correlation length
- `{prefix}_detailed_stats.txt` - Deterministic bottom-hash exact-shell sample

Legacy columns remain first and in their historical order; new columns are
appended. Detailed rows are the lowest platform-independent SplitMix64 keys of
`(run seed, direction, energy, replica index)`, so selection never depends on
an observable or family. Every row records the pre-resampling family ID,
replica ID, hash and rank, shell and sample populations, inclusion probability,
signed magnetic/polarization components, and all three minimum-wavevector
structure factors. The default is 1000 rows per shell; `-1` is the exact
all-matching mode for small-system validation. See
`../art4_baxterwu/articles/MCPA2_OBSERVABLE_DESIGN.md` for definitions,
normalizations, cost, and validation.

The three outputs collectively retain the ingredients rather than performing
the final analyses in C++: exact culling in `_main` supports Fisher/EPD zeros;
full-shell structure factors in `_agg_stats` support reweighted second-moment
correlation length; genealogy and clustered shell statistics support replica
correlation audits; and the unbiased signed detailed sample supports exploratory
Lee--Yang projections and cumulants. A cap of 1000 is an estimator, not an exact
magnetization histogram; use all-matching small systems and independent seeds
to validate it.

The output location is selected by the program/run environment; keep generated
datasets outside this source repository. `main_bw` recursively creates
`datasets/2DBaxterWu` in its working directory and, when checkpointing is
enabled, recursively creates the requested checkpoint directory. Failure to
create or open any required path stops the run before simulation begins.

The first data row of `_main.txt` records the GPU name, compute capability,
total memory, free memory before and after simulation setup, and CUDA driver and
runtime versions. Later rows contain `NA` in those run-level columns, keeping
the table rectangular without repeating the same information at every energy.
Memory values are stored as exact byte counts; CUDA version integers use the
CUDA API encoding (for example, `12040` means CUDA 12.4). The scientifically
reusable shell-level equilibration time remains the `equilibrate_seconds`
column. The final whole-run timing breakdown remains in the terminal or Slurm
log, and no fourth simulation output file is created.

## Project Structure

```
MCPA/
├── lib/
│   ├── baxterwu_lib.h    - Library header
│   └── baxterwu_lib.cu   - CUDA implementation
├── main/
│   └── main.cpp          - Baxter--Wu simulation entry point
├── tests/                - Modular CPU/CUDA doctest suites
├── scripts/              - Self-contained HSE build, smoke, and array launchers
├── benchmarks/           - Independent-reference equilibration benchmark
├── optimization/
│   ├── variants/         - Named policy ledger; no copied source variants
│   ├── results/          - Compact accepted/rejected decision records
│   └── artifacts/        - Generated profiler reports
├── build/                - Single generated working build tree
├── CMakeLists.txt
├── test.sh
└── README.md
```

Production source lives in `lib/` and `main/`. Tests, policy ledgers, and compact
decision Markdown are maintained evidence. Keep only the canonical `build/` in
routine use. A short-lived `build-NAME/` is appropriate when two compile-time
variants must coexist for a controlled comparison, but remove it after the
configuration and result have been recorded under `optimization/`. All
`build*/`, profiler artifacts, and raw benchmark JSONL are generated and may be
recreated. Local curated outputs belong under
`../Analytics/2DBaxterWu/experiments/`. The cluster array writes its isolated
campaign below this checkout's ignored `experiments/` directory; sync completed
outputs back to the analysis project rather than committing them here.

This layout is authoritative only for the Baxter--Wu branch today. The
consolidation plan ports build/test/checkpoint practices model by model while
preserving byte versus integer spins, integer versus rational-component
energies, runtime `q`/`D`, model observables, and legacy reader contracts.

## References

For the original Population Annealing algorithm:
- Machta, J. & Ellis, G. M. (2011). "Monte Carlo population annealing." J. Phys. A: Math. Theor. 44 095001.

For the Baxter-Wu model:
- Baxter, R. J. & Wu, F. Y. (1973). "Exact solution of the eight-vertex model." Phys. Rev. Lett. 30 1026.

For machine learning applied to phase transitions:
- Sukhoverkhova, D., Mozolenko, V., & Shchur, L. (2025). "Phase probabilities in first-order transitions using machine learning." Physical Review E, 112(4), 044128. https://doi.org/10.1103/PhysRevE.112.044128

For the Blume-Capel model and microcanonical population annealing:
- Mozolenko, V., & Shchur, L. (2024). "Blume-Capel model analysis with a microcanonical population annealing method." Physical Review E, 109(4), 045306. https://doi.org/10.1103/PhysRevE.109.045306

- Mozolenko, V., Fadeeva, M., & Shchur, L. (2024). "Comparison of the microcanonical population annealing algorithm with the Wang-Landau algorithm." Physical Review E, 110(4), 045301. https://doi.org/10.1103/PhysRevE.110.045301

## License

MIT License

Copyright (c) 2024 Lev Shchur, Viacheslav Mozolenko, Natan Rose, John Machta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
