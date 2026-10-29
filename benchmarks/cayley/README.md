# Cyclic Cayley implementation and benchmarks

The production free-ring propagator batches the Cartesian degrees of freedom
into a contiguous `(3 * Natoms, Nbeads)` view. It reuses the cyclic Thomas
factorization and workspace, caches reciprocal diagonal and mass-dependent
coefficients, and evaluates the same full Cayley map in midpoint form:

```text
M = I + (dt * omega_P)^2 L / 4
M q_mid = q + dt p / (2m)
q_new = 2 q_mid - q
p_new = p - dt m omega_P^2 L q_mid
```

The public Fortran/f2py interface still accepts `(3, Natoms, Nbeads)` arrays.
Explicit-shape sequence association provides the internal view without a copy.
The one-bead drift and two-bead doubled-edge cases have separate branches.
Factor and mass caches refresh when their parameters change. The module uses
process-local mutable state and is intended for serial calls within each worker.

From the repository root, build and run the Cayley regression tests:

```bash
make build python=python
PYTHONPATH=. python -m unittest -v tests.cayleytest
```

The recorded environment used Python 3.11, NumPy 1.24, and GNU Fortran 11.3.1.
The benchmark additionally requires the FFTW3 library and `fftw3.f03` header
(the script uses `/usr/include`). It compiles a fresh native library directly
from the production Fortran sources; it does not time Python calls per step.

## Reproduce the comparisons

Compare the current kernel with the original scalar cyclic implementation:

```bash
python benchmarks/cayley/run.py \
  --baseline-source benchmarks/cayley/baselines/scalar.f90 \
  --free-ring-only --trials 7 --sample-seconds 0.15
```

Compare the final flattened implementation with the intermediate three-dimensional
batched implementation and the supplied five-pass `stepB` reference:

```bash
python benchmarks/cayley/run.py \
  --baseline-source benchmarks/cayley/baselines/batched_3d.f90 \
  --step-b --free-ring-only --trials 7 --sample-seconds 0.15
```

Add `--native` to test `-march=native` on the execution host. Omit
`--free-ring-only` to also time two physical-potential kicks, the free ring, and
one new HHOO analytic PES/gradient evaluation. All backends use FP64 and the same
full Cayley mapping. FFTW uses cached, batched `FFTW_MEASURE` plans; its data
packing and unpacking are timed. Matrix construction, factorization and FFTW
planning are excluded. The dense controls use the repository's matrix-vector
transform layout, not optimized batched BLAS GEMM.

Each invocation writes a new `runs/` directory with its frozen sources, compiler
command, validation results, and raw timings. To compare 64-step nonlinear-PES
NVE trajectories after a run that includes a baseline:

```bash
python benchmarks/cayley/verify_trajectory.py RUN_DIRECTORY
```

## Recorded results

The measurements below used a Xeon Platinum 8176, one pinned logical CPU,
four H/H/O/O atoms, 300 K and a 0.5 fs step. Values are wall-clock medians of
seven randomized-order trials, in microseconds per complete free-ring step.
Frequency and other processes on the shared host were not controlled.

| Build | Beads | Previous 3D batch | Supplied stepB | Final 2D midpoint | FFTW |
|---|---:|---:|---:|---:|---:|
| Generic | 32 | 1.472 | 1.591 | 1.305 | 2.762 |
| Generic | 128 | 5.845 | 6.205 | 5.171 | 8.556 |
| Generic | 512 | 23.165 | 24.260 | 20.833 | 45.574 |
| Native | 32 | 1.470 | 1.543 | 1.151 | 3.721 |
| Native | 128 | 6.619 | 5.811 | 4.254 | 9.456 |
| Native | 512 | 20.148 | 23.981 | 17.651 | 55.201 |

Generic and native runs used logical CPUs 111 and 108 respectively. Compare
backends within a run; cross-build absolute ratios also include host variability.
The native compiler emitted AVX2/FMA instructions, not AVX-512, for these kernels.
The production Makefile retains generic compilation flags.

Each final build passed 84 single-step parameter/backend checks, six 2000-step
free-ring energy/reversibility checks, and six short nonlinear-PES trajectory
comparisons. The repository Cayley tests also cover cache invalidation after
atom-count, mass, temperature, bead-count and time-step changes.

Evidence directories retain the original JSON and CSV records:

- `evidence/batching/`: original scalar cyclic versus the intermediate batched
  kernel, including component timings with the analytic PES.
- `evidence/vector_generic/`: final generic build comparison shown above.
- `evidence/vector_native/`: final native build comparison shown above.

Absolute build paths in those historical JSON records identify the measurement
workspace. The commands above reproduce the benchmark from this repository.
Generated libraries, intermediate module files and task outputs are excluded.

These are free-ring/component benchmarks. They do not establish a full RPMD or
PMF speedup: the thermostat, reaction coordinate, constraints, bias, Python
callbacks and I/O are outside the timed scope. At 32 beads the analytic PES
dominates the combined kernel, and its repeated timing ranges overlap.
