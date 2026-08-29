# Test suite

Run everything with `./test.sh`, correctness without timing with
`./test.sh correctness`, CPU-only checks with `./test.sh cpu`, or list the
human-readable contracts with `./test.sh list`. CUDA tests return CTest status
**Skipped** when no GPU is accessible; that is distinct from a failure.
Each contract runs separately and prints its final verdict immediately, so a
long suite never appears stuck behind buffered CTest output. Interactive
terminals show **PASSED** in green, **FAILED** in red, and **SKIPPED** in yellow;
redirected logs stay free of ANSI escapes. Set `MCPA_COLOR=always` or
`MCPA_COLOR=never` to override automatic colour detection. A failed contract
still expands a `FAILURE DETAILS` section with the doctest assertion and
captured parameters.

Checkpoint reliability has its own category: run `./test.sh checkpoint`. It
covers state and RNG round trips, corrupted/current/previous/temporary file
boundaries, exact cooling and heating continuation, and a real process killed
between checkpoint and completion. `./test.sh correctness` and `./test.sh`
still include this category.

Tests use doctest. Every literal `TEST_CASE("human-readable contract")` is
registered as a separate CTest line. Add CPU cases in `test_cpu_<topic>.cpp` or
CUDA cases in `test_cuda_<topic>.cu`; CMake discovers both files and cases
automatically. Keep each unit case focused (normally `L=3` or `L=6`) and use
`CAPTURE` so a failure reports the relevant parameters and assertion. The
cluster-style process interruption fixture uses bounded `L=18` so the process
cannot finish before the test kills it.

CUDA correctness is deliberately stricter than numerical agreement: the
equilibration reference checks spins, tracked energies, statistics, and Philox
state byte-for-byte. Any performance variant must pass this before timing.

The benchmark is not another physics test. It measures kernel and wall time,
reports spread and throughput, and uses a bitwise correctness gate before it
starts timing. `./test.sh` runs a small benchmark after the contracts, but never
turns a noisy speed difference into a correctness failure.

The exact oracle currently exhausts all `2^9 = 512` configurations at `L=3`.
It checks the full density of states, `g(E)=g(-E)`, and the first exact
cooling/heating culling fraction `4/512`. This is a sharp small-system contract;
it is not presented as a one-percent validation of an entire finite-`R` walk.

This suite is Baxter--Wu-specific even where its infrastructure is reusable.
When modernizing another MCPA branch, port the named runner and checkpoint
failure boundaries, but derive a new exact oracle, spin-domain checks, energy
representation checks, and reference trajectory for that model. A Baxter--Wu
bitwise pass cannot validate Blume--Capel, Potts, or Ising physics.
