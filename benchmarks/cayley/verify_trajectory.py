"""Compare short HHOO NVE trajectories using the paired benchmark library."""
import ctypes as ct
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    run = Path(sys.argv[1]).resolve()
    report = json.loads((run / "results.json").read_text())
    masses = np.array(report["masses_au"])
    beta = 1.0 / (3.166811563e-6 * report["temperature_K"])
    lib = ct.CDLL(str(run / "build/kernels.so"))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    lib.prepare_case.argtypes = [ct.c_int, ct.c_double, ct.c_double, array]
    lib.single_step.argtypes = [ct.c_int, ct.c_int, array, array]
    lib.get_potential_.argtypes = [array, array, array] + [ct.POINTER(ct.c_int)] * 3
    for name in ("prepare_case", "single_step", "get_potential_"):
        getattr(lib, name).restype = None
    os.chdir(run / "source")
    n, steps = 32, 64
    rng = np.random.default_rng(151)
    template = np.array([[-.7, 0, 0], [.7, 0, 0], [0, 3, -1.4], [0, 3, 1.4]]).T
    q0 = np.asfortranarray(template[:, :, None] + rng.normal(0, .01, (3, 4, n)))
    p0 = np.asfortranarray(rng.normal(size=q0.shape) * np.sqrt(masses[None, :, None] * n / beta))
    methods = [(4, "before"), (3, "optimized"), (2, "fftw")]
    if "user_step_b" in report["results"][0]["methods"]:
        methods.append((5, "step_b"))
    checks = []
    for dt_fs in (.1, .5):
        step = dt_fs / .02418884326505
        lib.prepare_case(n, beta, step, masses)
        states = {}
        for method, label in methods:
            p, q = p0.copy(order="F"), q0.copy(order="F")
            values = np.empty(n)
            gradient = np.empty(q.shape, order="F")
            atoms, beads, status = ct.c_int(4), ct.c_int(n), ct.c_int()

            def evaluate():
                lib.get_potential_(q, values, gradient, ct.byref(atoms), ct.byref(beads), ct.byref(status))
                if status.value or not np.isfinite(values).all() or not np.isfinite(gradient).all():
                    raise AssertionError("HHOO PES evaluation failed")

            evaluate()
            for _ in range(steps):
                p -= .5 * step * gradient
                lib.single_step(n, method, p, q)
                evaluate()
                p -= .5 * step * gradient
            states[label] = (p.copy(), q.copy())
        p_ref, q_ref = states["before"]
        for _, label in methods[1:]:
            p, q = states[label]
            dq = float(np.max(np.abs(q - q_ref)))
            dv = float(np.max(np.abs((p - p_ref) / masses[None, :, None])))
            check = dict(method=label, beads=n, steps=steps, dt_fs=dt_fs,
                         max_abs_dq_bohr=dq, max_abs_dv_au=dv,
                         passed=bool(dq < 2e-9 and dv < 2e-9))
            checks.append(check)
            if not check["passed"]:
                raise AssertionError(check)
    result = dict(scope="Short nonlinear-PES NVE equivalence; no convergence or PES accuracy claim",
                  all_passed=True, checks=checks)
    (run / "pes_trajectory_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
