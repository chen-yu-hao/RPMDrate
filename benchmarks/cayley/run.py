"""Compare the production cyclic Cayley with dense and batched FFTW references.

Run from any directory with Python/NumPy, GNU Fortran and FFTW development files.
Each invocation keeps a new source snapshot, build, checks, and raw timings.
"""

import argparse
import csv
import ctypes as ct
import ctypes.util
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("dense_repository_cayley", "dense_cached_cayley", "fftw_cayley", "cyclic_production_cayley")
MASSES = np.array([1837.15, 1837.15, 29156.95, 29156.95])
TEMPERATURE = 300.0
BETA = 1.0 / (3.166811563e-6 * TEMPERATURE)
FS_PER_AU = 0.02418884326505


def choose_cpu():
    permitted = sorted(os.sched_getaffinity(0))

    def counters():
        values = {}
        for line in Path("/proc/stat").read_text().splitlines():
            fields = line.split()
            if fields[0].startswith("cpu") and fields[0][3:].isdigit():
                counter = list(map(int, fields[1:9]))
                values[int(fields[0][3:])] = (sum(counter), counter[3] + counter[4])
        return values

    start = counters()
    time.sleep(0.25)
    end = counters()
    loads = {}
    for cpu in end:
        total = end[cpu][0] - start[cpu][0]
        idle = end[cpu][1] - start[cpu][1]
        loads[cpu] = (total - idle) / max(1, total)
    sibling_load = {}
    for cpu in permitted:
        siblings = []
        for token in Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip().split(","):
            bounds = list(map(int, token.split("-")))
            siblings.extend(range(bounds[0], bounds[-1] + 1))
        sibling_load[cpu] = sum(loads[item] for item in siblings)
    cpu = min(permitted, key=lambda item: (sibling_load[item], loads[item], -item))
    os.sched_setaffinity(0, {cpu})
    return cpu, loads[cpu], permitted


def build(destination, baseline_source=None, step_b=False, native=False):
    snapshot = destination / "source"
    snapshot.mkdir()
    sources = [ROOT / "rpmdrate" / name for name in
               ("_surface.f90", "_math.f90", "blas_lapack.f90", "_main.f90")]
    sources += [ROOT / "examples/H-H-O-O/pes/hhoo_1014_analytic.f90"]
    for source in sources + [HERE / "backends.f90", Path(__file__)]:
        shutil.copy2(source, snapshot / source.name)

    main = (snapshot / "_main.f90").read_text()
    name = "free_ring_polymer_normal_mode_step"
    start = main.index("    subroutine " + name + "(")
    end = main.index("    end subroutine " + name, start)
    reference = main[start:end] + "    end subroutine " + name + "\n"
    reference = reference.replace(name, "dense_repository_cayley")
    for old, new in (
        ("cos_wt = dcos(wt)", "cos_wt = (1d0-wt*wt/4d0)/(1d0+wt*wt/4d0)"),
        ("sin_wt = dsin(wt)", "sin_wt = wt/(1d0+wt*wt/4d0)"),
    ):
        if reference.count(old) != 1:
            raise RuntimeError("The repository normal-mode reference has changed: " + old)
        reference = reference.replace(old, new)
    (snapshot / "dense_repository.f90").write_text(
        "module dense_repository_reference\n"
        "use system, only: CC, ICC, mass, dt, beta, pi\nimplicit none\ncontains\n"
        + reference + "end module\n"
    )
    build_dir = destination / "build"
    build_dir.mkdir()
    paths = [snapshot / source.name for source in sources]
    paths += [snapshot / "dense_repository.f90"]
    if baseline_source is not None:
        original = baseline_source.read_text()
        (snapshot / "_main_before.f90").write_text(original)
        header = original[original.index("module system"):original.index("contains")]
        baseline = header.replace("module system", "module system_before", 1) + "contains\n"
        routines = ["cayley_prepare", "cayley_solve_tridiagonal", "cayley_solve_cyclic", "free_ring_polymer_cayley_step"]
        if "    subroutine free_ring_polymer_cayley_nd(" in original:
            routines.append("free_ring_polymer_cayley_nd")
        for routine in routines:
            start = original.index("    subroutine " + routine + "(")
            end = original.index("    end subroutine " + routine, start)
            baseline += original[start:end] + "    end subroutine " + routine + "\n"
        baseline += "end module system_before\n"
        (snapshot / "cyclic_before.f90").write_text(baseline)
        paths.append(snapshot / "cyclic_before.f90")
        backend = (snapshot / "backends.f90").read_text()
        backend = backend.replace("    implicit none", "    use system_before, only: beta_before => beta, dt_before => dt, &\n"
                                  "        mass_before => mass, prepare_before => cayley_prepare, &\n"
                                  "        step_before => free_ring_polymer_cayley_step\n    implicit none", 1)
        backend = backend.replace("        call cayley_prepare(n)", "        call cayley_prepare(n)\n"
                                  "        beta_before = beta_in\n        dt_before = dt_in\n"
                                  "        mass_before(1:4) = masses\n        call prepare_before(n)", 1)
        backend = backend.replace("        case default", "        case(4)\n            call step_before(p, q, 4, n)\n        case default", 1)
        (snapshot / "backends.f90").write_text(backend)
    if step_b:
        reference_path = HERE / "step_b_reference.f90"
        shutil.copy2(reference_path, snapshot / reference_path.name)
        paths.append(snapshot / reference_path.name)
        backend = (snapshot / "backends.f90").read_text()
        backend = backend.replace("    implicit none", "    use step_b_reference, only: prepare_step_b, step_b\n    implicit none", 1)
        backend = backend.replace("        call cayley_prepare(n)", "        call cayley_prepare(n)\n        call prepare_step_b(4, n)", 1)
        backend = backend.replace("        case default", "        case(5)\n            call step_b(p, q, 12, n)\n        case default", 1)
        (snapshot / "backends.f90").write_text(backend)
    paths.append(snapshot / "backends.f90")
    command = ["gfortran", "-O3", "-funroll-loops", "-fPIC", "-shared", "-ffree-line-length-none",
               "-I/usr/include", "-J" + str(build_dir), "-I" + str(build_dir)]
    if native:
        command.append("-march=native")
    command += list(map(str, paths)) + ["-lfftw3", "-o", str(build_dir / "kernels.so")]
    result = subprocess.run(command, cwd=build_dir, capture_output=True, text=True)
    (destination / "build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    return ct.CDLL(str(build_dir / "kernels.so")), command


def fft_reference(p, q, step):
    n = q.shape[2]
    omega2 = (2.0 * n / BETA * np.sin(np.pi * np.arange(n // 2 + 1) / n))**2
    denominator = 1.0 + step**2 * omega2 / 4.0
    a = (1.0 - step**2 * omega2 / 4.0) / denominator
    b = step / denominator
    pp, qq = np.fft.rfft(p, axis=2), np.fft.rfft(q, axis=2)
    pp_new = a * pp - MASSES[None, :, None] * omega2 * b * qq
    qq_new = b * pp / MASSES[None, :, None] + a * qq
    return np.fft.irfft(pp_new, n=n, axis=2), np.fft.irfft(qq_new, n=n, axis=2)


def initial_state(n, rng, zero_centroid_momentum=False):
    template = np.array([[-0.7, 0, 0], [0.7, 0, 0], [0, 3, -1.4], [0, 3, 1.4]]).T
    q = np.asfortranarray(template[:, :, None] + rng.normal(0, 0.02, (3, 4, n)))
    p = rng.normal(size=(3, 4, n)) * np.sqrt(MASSES[None, :, None] * n / BETA)
    if zero_centroid_momentum:
        p -= p.mean(axis=2, keepdims=True)
    return np.asfortranarray(p), q


def validate(lib):
    rng = np.random.default_rng(73029)
    checks = []
    for n in (1, 2, 3, 5, 32, 128, 512):
        for step_fs in (0.1, 0.5):
            step = step_fs / FS_PER_AU
            lib.prepare_case(n, BETA, step, MASSES)
            p0, q0 = initial_state(n, rng)
            pref, qref = fft_reference(p0, q0, step)
            for method, label in enumerate(METHODS):
                p, q = p0.copy(order="F"), q0.copy(order="F")
                lib.single_step(n, method, p, q)
                dp = float(np.max(np.abs(p - pref)))
                dq = float(np.max(np.abs(q - qref)))
                dv = float(np.max(np.abs((p - pref) / MASSES[None, :, None])))
                centroid_error = float(np.max(np.abs(
                    q.mean(axis=2) - q0.mean(axis=2) - step * (p0 / MASSES[None, :, None]).mean(axis=2)
                )))
                passed = bool(np.isfinite(p).all() and np.isfinite(q).all()
                              and dq < 5e-10 and dv < 5e-10 and centroid_error < 5e-10)
                checks.append(dict(beads=n, dt_fs=step_fs, method=label, max_abs_dp=dp,
                                   max_abs_dq=dq, max_abs_dv=dv, centroid_error=centroid_error, passed=passed))
                if not passed:
                    raise AssertionError(checks[-1])

    def energy(p, q, n):
        return (np.sum(p**2 / (2 * MASSES[None, :, None]))
                + np.sum(0.5 * MASSES[None, :, None] * (n / BETA)**2
                         * (q - np.roll(q, 1, axis=2))**2))

    n, count, step = 32, 2000, 0.5 / FS_PER_AU
    p0, q0 = initial_state(n, rng)
    long_checks = []
    for method, label in enumerate(METHODS):
        lib.prepare_case(n, BETA, step, MASSES)
        p, q = p0.copy(order="F"), q0.copy(order="F")
        e0 = energy(p, q, n)
        lib.advance(n, method, count, p, q)
        energy_error = abs(energy(p, q, n) - e0) / abs(e0)
        lib.prepare_case(n, BETA, -step, MASSES)
        lib.advance(n, method, count, p, q)
        reverse_q = float(np.max(np.abs(q - q0)))
        reverse_v = float(np.max(np.abs((p - p0) / MASSES[None, :, None])))
        passed = bool(energy_error < 5e-8 and reverse_q < 5e-7 and reverse_v < 5e-9)
        long_checks.append(dict(method=label, beads=n, steps_each_direction=count,
                                relative_energy_error=float(energy_error), reverse_q=reverse_q,
                                reverse_v=reverse_v, passed=passed))
        if not passed:
            raise AssertionError(long_checks[-1])
    return dict(single_step=checks, long_free_ring=long_checks, all_passed=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--sample-seconds", type=float, default=0.12)
    parser.add_argument("--baseline-source", type=Path, help="Also time a frozen pre-optimization cyclic routine")
    parser.add_argument("--step-b", action="store_true", help="Compare the supplied stepB with the frozen baseline and production kernel")
    parser.add_argument("--native", action="store_true", help="Benchmark a host-specific -march=native build")
    parser.add_argument("--free-ring-only", action="store_true", help="Time only the free-ring kernels")
    args = parser.parse_args()
    if args.step_b and args.baseline_source is None:
        parser.error("--step-b requires --baseline-source for the paired comparison")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = HERE / "runs" / run_id
    destination.mkdir(parents=True)
    cpu, recent_load, previous_affinity = choose_cpu()
    print(json.dumps(dict(output=str(destination), pinned_cpu=cpu)), flush=True)
    global METHODS
    if args.baseline_source is not None:
        METHODS += ("cyclic_before",)
    if args.step_b:
        METHODS += ("user_step_b",)
    lib, command = build(destination, args.baseline_source, args.step_b, args.native)
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    lib.prepare_case.argtypes = [ct.c_int, ct.c_double, ct.c_double, array]
    lib.single_step.argtypes = [ct.c_int, ct.c_int, array, array]
    lib.advance.argtypes = [ct.c_int, ct.c_int, ct.c_int, array, array]
    lib.measure.argtypes = [ct.c_int] * 4 + [array] * 4 + [ct.POINTER(ct.c_double)] * 2
    for name in ("prepare_case", "single_step", "advance", "measure"):
        getattr(lib, name).restype = None
    checks = validate(lib)
    (destination / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("All single-step, centroid, reversibility, and free-ring energy checks passed.", flush=True)
    weights = ROOT / "examples/H-H-O-O/pes/txt_1014"
    shutil.copy2(weights, destination / "source/txt_1014")
    os.chdir(destination / "source")
    rng = np.random.default_rng(20261029)
    raw = []
    summary = []
    for step_fs in (0.1, 0.5):
        for n in (32, 128, 512):
            lib.prepare_case(n, BETA, step_fs / FS_PER_AU, MASSES)
            p0, q0 = initial_state(n, rng, zero_centroid_momentum=True)
            p, q = np.empty_like(p0, order="F"), np.empty_like(q0, order="F")
            for with_pes in ((0,) if args.free_ring_only else (0, 1)):
                scope = "free_ring_continuous" if with_pes == 0 else "kick_ring_PES_kick_reset"
                repetitions = {}
                for method in range(len(METHODS)):
                    wall, cpu_time = ct.c_double(), ct.c_double()
                    lib.measure(n, method, with_pes, 7, p0, q0, p, q, ct.byref(wall), ct.byref(cpu_time))
                    repetitions[method] = max(5, min(200000, round(args.sample_seconds / max(wall.value, 1e-9))))
                for trial in range(args.trials):
                    for method in rng.permutation(len(METHODS)):
                        wall, cpu_time = ct.c_double(), ct.c_double()
                        reps = repetitions[int(method)]
                        lib.measure(n, int(method), with_pes, reps, p0, q0, p, q, ct.byref(wall), ct.byref(cpu_time))
                        if not np.isfinite(p).all() or not np.isfinite(q).all():
                            raise AssertionError("Non-finite final benchmark state")
                        raw.append(dict(beads=n, dt_fs=step_fs, scope=scope, trial=trial+1,
                                        method=METHODS[method], repetitions=reps,
                                        wall_us=wall.value*1e6, cpu_us=cpu_time.value*1e6))
                row = dict(beads=n, dt_fs=step_fs, scope=scope, methods={})
                for label in METHODS:
                    samples = [sample for sample in raw if sample["beads"] == n
                               and sample["dt_fs"] == step_fs and sample["scope"] == scope
                               and sample["method"] == label]
                    wall_samples = [sample["wall_us"] for sample in samples]
                    cpu_samples = [sample["cpu_us"] for sample in samples]
                    row["methods"][label] = dict(median_us=float(np.median(wall_samples)),
                                                 min_us=min(wall_samples), max_us=max(wall_samples),
                                                 median_cpu_us=float(np.median(cpu_samples)))
                summary.append(row)
                print(json.dumps(row), flush=True)

    fftw = ct.CDLL(ctypes.util.find_library("fftw3"))
    version = (ct.c_char * 256).in_dll(fftw, "fftw_version").value.decode()
    report = dict(host=platform.node(), platform=platform.platform(), cpu_model=next(
        line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")), pinned_cpu=cpu, cpu_load_before_pin=recent_load,
        prior_affinity=previous_affinity, compiler=subprocess.check_output(["gfortran","--version"],text=True).splitlines()[0],
        fftw_version=version, fftw_planner="FFTW_MEASURE; 24 batched r2c/c2r transforms",
        compile_command=command, native_build=args.native, temperature_K=TEMPERATURE, atom_order=["H","H","O","O"],
        masses_au=MASSES.tolist(), precision="FP64", threads=1, trials=args.trials,
        source_commit=(subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
                       if (ROOT / ".git").exists() else None),
        source_snapshot=str(destination/"source"), all_validation_passed=checks["all_passed"],
        notes=["The cyclic backend calls the frozen production routine from _main.f90.",
               "All backends implement the same full Cayley map; the exact exponential propagator is not timed.",
               "Dense repository style changes only modal cos/sin factors to Cayley coefficients.",
               "Dense cached precomputes the same coefficients; both dense methods retain repository matmul transforms, not BLAS GEMM.",
               "Initialization, matrix/factor preparation and FFTW planning are excluded.",
               "Free-ring timing advances continuous native loops; packing and unpacking for FFTW are included.",
               "PES timing resets to the same p/q, reuses the initial gradient, evaluates one new PES/gradient, and includes two kicks.",
               "PES timing excludes thermostat, reaction coordinate, bias, constraint, Python callbacks and file I/O; it is not a full RPMD/PMF speedup.",
               "CPU affinity is fixed but CPU frequency and other users' activity are not controlled."],
        results=summary)
    (destination/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    with (destination/"raw_timings.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw)
    print("RESULT_DIRECTORY=" + str(destination), flush=True)


if __name__ == "__main__":
    main()
