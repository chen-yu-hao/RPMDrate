#!/usr/bin/env python3
"""Compare identical compiled PES kernels through three transport mechanisms.

No speed threshold is a correctness test. Startup is separate; warmed call
latency, raw round timings, numerical differences and optional full NVE step
cost are recorded. NumPy HHOO is reported separately, not treated as the same
implementation. Run from a real script (multiprocessing spawn).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from rpmdrate.drpmd_backends import PotentialSpec, SerialPotential, BeadParallelPotential
from rpmdrate.drpmd_shared import SharedMemoryPotential
from rpmdrate.drpmd_native_pes import NativePotential, build_hhoo


def geometry(beads, kernel):
    rng = np.random.default_rng(765)
    if kernel == "hhoo":
        xyz = np.array([[0., 0., 20.], [-0.65, 1.72, 0.], [0., 0., 0.], [2.52, 0., 0.]])
    else:
        xyz = np.zeros((4, 3))
    return [np.asfortranarray(xyz.T[:, :, None]+rng.normal(0, 0.015, (3, 4, beads))) for _ in range(8)]


def _output(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("hhoo", "harmonic"), default="hhoo")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--beads", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--nve-steps", type=int, default=64, help="0 disables dynamics timing")
    parser.add_argument("--output", type=Path, default=Path("pes-benchmark.json"))
    args = parser.parse_args(argv)
    if min(args.workers, args.calls, args.rounds, *args.beads) < 1 or args.nve_steps < 0:
        parser.error("counts must be positive; nve-steps may be zero")
    if args.output.exists():
        parser.error("output exists; choose a new path")
    os.environ["OMP_DYNAMIC"] = "FALSE"
    report = {"format": "RPMDrate.PES.benchmark.v1", "kernel": args.kernel,
              "workers": args.workers, "calls_per_round": args.calls, "rounds": args.rounds,
              "nve_steps": args.nve_steps, "python": sys.version, "numpy": np.__version__,
              "platform": platform.platform(), "git_commit": _output(["git", "rev-parse", "HEAD"]),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "compiler": _output(["gfortran" if args.kernel == "hhoo" else "gcc", "--version"]),
              "build_flags": "-O3 -fPIC -fopenmp -shared; Fortran also -frecursive; no fast-math",
              "notes": ["PES timing excludes worker startup; raw round times are retained.",
                        "native/process/shared_memory rows use the identical compiled kernel.",
                        "numpy_* rows use the existing vectorized NumPy implementation, not the compiled kernel.",
                        "Full NVE timing uses identical one-thread Cayley dynamics, includes initial force evaluation, and excludes file IO.",
                        "Microbenchmarks are machine-specific; no chemical rate/convergence or GPU performance is claimed."],
              "results": []}
    for path, key in (("/sys/fs/cgroup/cpu.max", "cpu_quota"), ("/proc/cpuinfo", "cpuinfo")):
        if Path(path).exists():
            text = Path(path).read_text()
            report[key] = next((line for line in text.splitlines() if "model name" in line), text.strip()) if key == "cpuinfo" else text.strip()
    with tempfile.TemporaryDirectory(prefix="drpmd-pes-bench-") as directory, threadpool_limits(limits=1, user_api="blas"):
        library = Path(directory)/"pes.so"
        model = None
        if args.kernel == "hhoo":
            build_hhoo(library)
            model = str(ROOT/"examples/H-H-O-O/pes/txt_1014")
        else:
            subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-fopenmp",
                            str(ROOT/"examples/native_pes/harmonic.c"), "-o", str(library)], check=True)
        spec = PotentialSpec("rpmdrate.drpmd_native_pes", initializer_kwargs={"library": str(library), "model": model})
        for beads in args.beads:
            frames = geometry(beads, args.kernel)
            factories = {
                "native_1": lambda: NativePotential(library, model, threads=1),
                f"process_{args.workers}": lambda: BeadParallelPotential(spec, args.workers),
                f"shared_memory_{args.workers}": lambda: SharedMemoryPotential(spec, args.workers, frames[0].shape),
            }
            if args.workers > 1:
                factories[f"native_{args.workers}"] = lambda: NativePotential(library, model, threads=args.workers)
            if args.kernel == "hhoo":
                numpy_spec = PotentialSpec("rpmdrate.drpmd_hhoo", initializer_kwargs={"weights_path": model})
                factories["numpy_1"] = lambda: SerialPotential(numpy_spec)
                factories[f"numpy_process_{args.workers}"] = lambda: BeadParallelPotential(numpy_spec, args.workers)
                factories[f"numpy_shared_memory_{args.workers}"] = lambda: SharedMemoryPotential(numpy_spec, args.workers, frames[0].shape)
            with ExitStack() as stack:
                backends, entries = {}, {}
                for name, factory in factories.items():
                    started = time.perf_counter()
                    backends[name] = stack.enter_context(factory())
                    entries[name] = {"beads": beads, "backend": name, "startup_seconds": time.perf_counter()-started,
                                     "raw_us_per_call": []}
                reference = backends["native_1"].evaluate(frames[0])
                for name, backend in backends.items():
                    current = backend.evaluate(frames[0])
                    for actual, expected in zip(current, reference):
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
                    entries[name]["max_energy_error_hartree"] = float(np.max(np.abs(current[0]-reference[0])))
                    entries[name]["max_gradient_error_hartree_per_bohr"] = float(np.max(np.abs(current[1]-reference[1])))
                    for i in range(8):
                        backend.evaluate(frames[i])
                order_rng = np.random.default_rng(991)
                names = list(backends)
                for _ in range(args.rounds):
                    for index in order_rng.permutation(len(names)):
                        name = names[index]
                        started = time.perf_counter_ns()
                        for call in range(args.calls):
                            backends[name].evaluate(frames[call % len(frames)])
                        duration = (time.perf_counter_ns()-started)/1000/args.calls
                        entries[name]["raw_us_per_call"].append(duration)
                nve_reference = None
                for name, backend in backends.items():
                    row = entries[name]
                    raw = row["raw_us_per_call"]
                    row["median_us_per_call"] = statistics.median(raw)
                    row["p10_us_per_call"], row["p90_us_per_call"] = [float(x) for x in np.percentile(raw, [10, 90])]
                    row["actual_native_threads"] = getattr(backend, "last_threads_used", None)
                    row["worker_pids"] = list(getattr(backend, "worker_pids", []))
                    if args.nve_steps:
                        from rpmdrate.drpmd_core import Dynamics, AMU, KB, TIME_FS, propagate, sample_momenta
                        masses = np.array([1., 1., 16., 16.])*AMU
                        momenta = sample_momenta(masses, beads, 1/(KB*300), np.random.default_rng(811))
                        dynamics = Dynamics(masses, 300, beads, 0.02/TIME_FS, backend, threads=1, backend="python")
                        times = []
                        for _ in range(3):
                            started = time.perf_counter()
                            result = propagate(dynamics, frames[0], momenta, args.nve_steps)
                            times.append((time.perf_counter()-started)/args.nve_steps*1e6)
                            if result.status == "failed":
                                raise RuntimeError(result.error)
                        if nve_reference is None:
                            nve_reference = result
                        np.testing.assert_allclose(result.q, nve_reference.q, rtol=0, atol=5e-11)
                        np.testing.assert_allclose(result.p, nve_reference.p, rtol=0, atol=5e-10)
                        row["nve_raw_us_per_step"] = times
                        row["nve_median_us_per_step"] = statistics.median(times)
                        row["nve_max_q_error_bohr"] = float(np.max(np.abs(result.q-nve_reference.q)))
                        row["nve_max_p_error_au"] = float(np.max(np.abs(result.p-nve_reference.p)))
                    report["results"].append(row)
                    print(f"P={beads:4d} {name:24s} PES={row['median_us_per_call']:10.2f} us/call "
                          f"NVE={row.get('nve_median_us_per_step', 0):10.2f} us/step", flush=True)
        report["threadpools"] = threadpool_info()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("BENCHMARK_JSON=" + json.dumps(report, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
