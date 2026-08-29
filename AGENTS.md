# AGENTS.md - Monte-Carlo Population Annealing (MCPA)

## Branch overview

- **2DBaxterWu** (current) — triangular lattice, 3-spin interactions, per-replica m_a/m_b/m_c/p stats
- **2DBlumeCapel** — square lattice BC model.  New files `lib/blumeCapel_lib.*` and `main/main_bc.cpp`
  port all 2DBaxterWu "fancy stuff" and add per-replica (e_j, e_delta) for field-mixing analysis.
  See `AGENTS.md` notes in that branch.

## 2DBlumeCapel: what was ported from 2DBaxterWu

| Feature | Status |
|---------|--------|
| Philox4-32 RNG | ✅ |
| Modular lib/main structure | ✅ |
| Cooling + heating modes | ✅ |
| 3-file output (main/agg/detailed_stats) | ✅ |
| Per-replica statistics | ✅ now stores e_j, e_delta, m |
| Float energy (D is real) | ✅ `float* E` |
| Square lattice 4-neighbour SLF | ✅ |
| BC local energy (bond + crystal-field) | ✅ |

## TODO for 2DBlumeCapel before running

1. Fix hardcoded output path (`outdir`) in `main/main_bc.cpp`
2. Create VS project files (copy from BW, rename sources)
3. Add unit tests for BC kernels
4. Validate at D=0 (Ising limit): T_c should match Ising T_c ≈ 2.269

---

## Project Overview

CUDA C++ physics simulation implementing Population Annealing for the 2D Baxter-Wu model.

## Build

- Configure with CMake 3.22+, CUDA 12.4, and a C++17 host compiler.
- Build the `main_bc` target, or run `./test.sh all` for the complete named suite.
- The default build output is `build/main_bc`.

## Running the Simulation

```
main/main.exe <seed> <L> <blocks> <threads> <nSteps> <heat> [timing]
```

Arguments:
- `seed`: Random seed (int)
- `L`: Lattice dimension (N = L × L)
- `blocks`: GPU blocks
- `threads`: GPU threads per block (total replicas = blocks × threads)
- `nSteps`: Monte Carlo steps per iteration
- `heat`: 0 = annealing (decreasing energy), 1 = heating (increasing energy)
- `timing` (optional): 1 = enable kernel-level timing

Example: `main/main.exe 42 16 1 16 10 0`

## Running Tests

```
tests/tests.exe
```

Tests verify: pure CPU logic, CUDA setup, kernel energy, resample (stub), integration (stub).

## Output Files

Written to `C:/Users/.../ASAV/Analytics/datasets/`:
- `{prefix}_main.txt` - Energy, culling factor, family size vs U
- `{prefix}_agg_stats.txt` - Aggregated statistics
- `{prefix}_detailed_stats.txt` - Per-replica statistics

## Architecture

```
lib/
  baxterwu_lib.h    - Header with all CUDA kernel declarations
  baxterwu_lib.cu   - CUDA implementation
main/
  main.cpp          - Entry point
tests/
  test_*.cpp       - Test files (Phase 1-3 implemented, 4-5 stubs)
```

## Key Implementation Details

- Energy bounds: `[-2*N - 2, 2*N + 2]` where N = L²
- Culling factor X = nCull / R (fraction of replicas below energy threshold U)
- Family size: ρ_t = (1/R²) × Σ n_i²
- Uses cuRAND for random number generation
- Unified memory with managed CUDA allocations
