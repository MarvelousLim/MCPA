# Equilibration experiments

This directory is the laboratory notebook for performance work. Production
sources remain in `lib/`; experiments are named policies and compile-time
switches, not copied `.cpp`/`.cu` snapshots. Git preserves every implementation
change, while the files here preserve *why and how it was measured*.

## Fast path

```bash
./optimization/run.sh baseline candidate smoke mixed cooling
./optimization/run.sh baseline candidate smoke mixed heating
./optimization/profile.sh baseline ncu smoke candidate
```

`run.sh` first runs the benchmark's bitwise comparison against the reference
loop, then appends the machine-readable benchmark line to
`results/<experiment>.jsonl` and one provenance row to `results/index.md`.
Generated JSONL and profiler reports are ignored; deliberately promote a small,
representative result to version control only when it supports a decision.

## Experiment contract

1. Start from an entry in `variants/policies.tsv`; add a row before coding a new
   policy.
2. Change one mechanism at a time. Keep the reference kernel unchanged.
3. Run `./test.sh` and a `smoke` comparison before larger presets.
4. Correctness means bitwise equality of spins, energy, statistics, and Philox
   state. A faster non-equivalent trajectory is a different algorithm.
5. Record the deterministic mixed regime in both directions; retain open and
   frozen as endpoint ceilings when acceptance behavior could matter.
6. Compare medians and dispersion on the same GPU; cluster and local timings are
   separate series because node model, clocks, load, and driver can differ.

`presets.tsv` documents the benchmark sizes. The local GPU is the default for
short tests; the cluster is reserved for final recalculation and hardware-specific
confirmation.

Use the normal `build/` tree for routine work. Controlled compile-time A/B
comparisons may temporarily use named `build-NAME/` trees, but those trees are
not evidence and should be removed after the CMake options, hardware, timings,
and decision have been written here. Never keep a build directory merely to
remember which candidate it represented.

## Suggested sequence

The register-state and synchronization stages are accepted. Heating/cooling
specialization was bitwise-correct but rejected as neutral; see
`results/stage2.md`. Grouped attempts were also bitwise-correct, including the
partial-`curand4` tail, but remained below the 2% performance gate; see
`results/stage4.md`. Structured block sweeps are now skipped because they change
attempt order. The neighbor-table candidate passed the bitwise gate and cleared
the 2% local mixed-workload threshold, but remains opt-in and default-off; see
`results/stage5.md`. Later byte-spin work remains a separate experiment and
requires separate authorization. Re-run bitwise checks after every stage.

Optimization decisions here are evidence for the Baxter--Wu kernel and the
recorded hardware only. Register-state handling and redundant synchronization
may inspire ports, but no policy is automatically enabled on another model
branch. Each destination needs its own independent reference loop, spin/energy
representation checks, both directions, relevant acceptance regimes, and fresh
timings.
