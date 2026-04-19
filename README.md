# Monte-Carlo Population Annealing (MCPA)

A computational physics project implementing the Monte-Carlo Population Annealing algorithm for the 2D Baxter-Wu model.

## Authors

PhD work with Lev Shchur, Natan Rose, and John Machta.

## Overview

This project simulates the 2D Baxter-Wu model using Population Annealing Monte Carlo (PAMC), a powerful algorithm for equilibrium and nonequilibrium systems. Population Annealing combines parallel tempering-like parallelism with sequential Monte Carlo simulations.

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

### Population Annealing

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

- Visual Studio 2022
- CUDA Toolkit 13.2+
- Windows 10/11 x64

### Build Steps

1. Open `MCPA.sln` in Visual Studio
2. Select configuration (Debug/Release) and platform (x64)
3. Build solution (Ctrl+Shift+B)

The build outputs:
- `main/main.exe` - Main simulation executable
- `tests/tests.exe` - Test suite

## Usage

### Running the Simulation

```
main.exe <seed> <L> <blocks> <threads> <nSteps> <heat>
```

Arguments:
- `seed`: Random seed for reproducibility
- `L`: Lattice dimension (N = L x L)
- `blocks`: Number of GPU blocks (replicas = blocks x threads)
- `threads`: Number of GPU threads per block
- `nSteps`: Number of Monte Carlo steps per replica per iteration
- `heat`: 0 for annealing (E decreasing), 1 for heating (E increasing)
- `enable_timings`: 1 for run time calc to console

Example:
```
main.exe 42 16 1 16 10 0 1
```
Runs on 16x16 lattice with 16 replicas, 10 MC steps, cooling.

### Test Suite

Build and run `tests/tests.exe` to verify core functionality:

```
x64\Debug\tests.exe
```

Test coverage:
- Phase 1: Pure CPU logic (between, local_energy, swap, quicksort, SLF)
- Phase 2: CUDA setup (curand states, population initialization)
- Phase 3: Kernel energy (device energy calculation, equilibrate, statistics)
- Phase 4-5: Resample and integration (stubs)

## Output Files

Simulation produces three output files per run:
- `{prefix}_main.txt` - Energy, culling factor, family size vs U
- `{prefix}_agg_stats.txt` - Aggregated statistics
- `{prefix}_detailed_stats.txt` - Per-replica detailed statistics

Files are saved to `C:/Users/.../ASAV/Analytics/datasets/`

## Project Structure

```
MCPA/
├── lib/
│   ├── baxterwu_lib.h    - Library header
│   └── baxterwu_lib.cu   - CUDA implementation
├── main/
│   ├── main.cpp         - Main simulation
│   └── main.vcxproj    - Project file
├── tests/
│   ├── test_*.cpp/h    - Test implementations
│   └── tests.vcxproj   - Test project
├── MCPA.sln            - Solution file
└── README.md          - This file
```

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
