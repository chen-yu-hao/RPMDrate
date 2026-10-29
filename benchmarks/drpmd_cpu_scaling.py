"""Compare dRPMD HHOO serial(1) and bead-parallel(2,4) for 32 beads / 1 ps."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
from rpmdrate.drpmd import run
from rpmdrate.drpmd_integrator import _load_native


def hardware():
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    quota = None
    try:
        numerator, denominator = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if numerator != "max":
            quota = float(numerator) / float(denominator)
    except (OSError, ValueError):
        pass
    return {
        "platform": platform.platform(), "python": sys.version.split()[0],
        "numpy": np.__version__, "visible_cpus": os.cpu_count(),
        "affinity_cpus": affinity, "cgroup_v2_cpu_quota": quota,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
    }


def compare_states(reference, other):
    deltas = {}
    for stage in ("thermal_state.npz", "trajectory_000001_initial.npz"):
        with np.load(reference / stage, allow_pickle=False) as a, \
             np.load(other / stage, allow_pickle=False) as b:
            for component in ("q", "p"):
                key = stage + "/" + component
                deltas[key] = float(np.max(np.abs(a[component] - b[component])))
                if not np.allclose(a[component], b[component], rtol=2e-10, atol=2e-8):
                    raise AssertionError("Initial states differ: " + key + " " + str(deltas[key]))
    return deltas


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("examples/drpmd/hhoo_smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("bench-results-32beads-1ps"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--nbeads", type=int, default=32)
    parser.add_argument("--duration-ps", type=float, default=1.0)
    args = parser.parse_args()
    if args.repeats < 1 or args.nbeads < 1 or not 0 < args.duration_ps <= 10:
        parser.error("Invalid repetitions, beads or duration")
    native = _load_native()
    if native is None or not native.drpmd_has_openmp():
        raise RuntimeError("An OpenMP native Cayley build is required, no silent fallback")

    src = args.input.resolve()
    base = src.parent
    config = json.loads(src.read_text(encoding="utf-8"))
    if config["potential"].get("type") != "hhoo":
        raise ValueError("This controlled benchmark uses the HHOO PES")
    step_fs = float(config["dt_fs"])
    steps = round(1000.0 * args.duration_ps / step_fs)
    if steps < 1 or not math.isclose(step_fs * steps, 1000 * args.duration_ps, abs_tol=1e-9):
        raise ValueError("Duration must be an integer multiple of dt_fs")
    config.update({
        "nbeads": args.nbeads, "evolution_steps": steps,
        "trajectories": 1,
        "check_interval": steps + 1,  # force the full trajectory for comparable timings
        "save_every": steps,  # exclude high-frequency trace I/O overhead
    })
    config["potential"]["weights"] = str(
        (base / config["potential"]["weights"]).resolve()
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "effective_input.json").write_text(json.dumps(config, indent=2) + "\n")
    machine = hardware()
    (output / "hardware.json").write_text(json.dumps(machine, indent=2) + "\n")
    print("BENCHMARK_SETUP", json.dumps({
        "nbeads": args.nbeads, "duration_ps": args.duration_ps,
        "dt_fs": step_fs, "steps": steps, "repeats": args.repeats,
        "hardware": machine,
    }), flush=True)

    rows = []
    directories = []
    cpus = (1, 2, 4)
    for repetition in range(1, args.repeats + 1):
        order = cpus if repetition % 2 else tuple(reversed(cpus))
        for workers in order:
            mode = "serial" if workers == 1 else "bead"
            label = f"cpu{workers}_rep{repetition}"
            folder = output / label
            summary = run(config, base=base, mode=mode, workers=workers, output=folder)
            if not summary["all_trajectories_resolved"] and summary["counts"]["unresolved"] != 1:
                raise AssertionError("Unexpected non-benchmark outcome: " + label)
            with (folder / "trajectories.csv").open(newline="") as stream:
                results = list(csv.DictReader(stream))
            if len(results) != 1 or int(results[0]["steps"]) != steps:
                raise AssertionError("Did not complete exactly requested steps: " + label)
            manifest = json.loads((folder / "input.json").read_text())
            if not manifest["native_cayley"]:
                raise AssertionError("Native Cayley was not used: " + label)
            if workers > 1 and len(set(manifest["pes_worker_pids"])) != workers:
                raise AssertionError("Not all PES workers initialized: " + label)
            t = summary["timing_seconds"]
            row = {
                "repeat": repetition, "cores": workers, "mode": mode,
                "label": label, "steps": steps,
                "collision_seconds": t["collision_propagation_and_trace"],
                "total_wall_seconds": t["wall"],
                "thermal_seconds": t["thermal_propagation"],
                "energy_drift_per_bead_hartree": float(
                    results[0]["max_energy_drift_per_bead_hartree"]
                ), "status": results[0]["status"],
            }
            rows.append(row)
            directories.append((label, folder))
            print("BENCHMARK_RUN", json.dumps(row), flush=True)

    initial_state_differences = {
        label: compare_states(directories[0][1], path) for label, path in directories[1:]
    }
    baseline = statistics.median(
        row["collision_seconds"] for row in rows if row["cores"] == 1
    )
    totals = []
    for cores in cpus:
        values = [row for row in rows if row["cores"] == cores]
        times = [row["collision_seconds"] for row in values]
        median = statistics.median(times)
        totals.append({
            "cores": cores, "mode": "serial" if cores == 1 else "bead",
            "repeats": len(times), "median_collision_seconds": median,
            "min_collision_seconds": min(times),
            "max_collision_seconds": max(times),
            "median_wall_seconds": statistics.median(
                row["total_wall_seconds"] for row in values
            ),
            "speedup": baseline / median,
            "parallel_efficiency_pct": 100 * baseline / (median * cores),
        })

    save_csv(output / "raw_times.csv", rows)
    save_csv(output / "scaling.csv", totals)
    result = {
        "configuration": {"nbeads": args.nbeads, "duration_ps": args.duration_ps,
                          "steps": steps, "dt_fs": step_fs, "repeats": args.repeats},
        "hardware": machine,
        "metric": "collision_propagation_and_trace (seconds), including PES transport",
        "notes": [
            "Serial(1) uses in-process HHOO PES and native 1-thread Cayley.",
            "Bead(2/4) uses spawn PES workers and matching OpenMP Cayley threads.",
            "Same random seed; thermal/initial states must match within tolerance.",
            "Auto-stop intentionally disabled: check_interval > evolution_steps.",
            "Not a converged chemical rate or a benchmark across different hosts.",
        ],
        "raw": rows, "summary": totals,
        "initial_state_max_abs_differences": initial_state_differences,
    }
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    table = [
        "# HHOO 32 beads, 1 ps: 1/2/4 CPU scaling",
        "",
        "Each CPU configuration ran twice on the same Ubuntu runner. "
        "Collision propagation includes 10,000 0.1-fs steps and PES transfers.",
        "",
        "| Cores | Median collision time (s) | Speedup | Parallel efficiency | Median wall (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in totals:
        table.append(
            f"| {r['cores']} | {r['median_collision_seconds']:.3f} | "
            f"{r['speedup']:.3f}x | {r['parallel_efficiency_pct']:.1f}% | "
            f"{r['median_wall_seconds']:.3f} |"
        )
    table.extend(["", "Note: benchmark forces full propagation, no product classification.", ""])
    report = "\n".join(table)
    (output / "REPORT.md").write_text(report)
    print("BENCHMARK_REPORT_BEGIN\n" + report + "\nBENCHMARK_REPORT_END", flush=True)
    for item in totals:
        print("BENCHMARK_RESULT", json.dumps(item), flush=True)
    print("BENCHMARK_VALIDATED", json.dumps({
        "runs": len(rows), "steps_each": steps,
        "max_initial_q_abs": max(
            (x["trajectory_000001_initial.npz/q"] for x in initial_state_differences.values()),
            default=0.0),
        "max_initial_p_abs": max(
            (x["trajectory_000001_initial.npz/p"] for x in initial_state_differences.values()),
            default=0.0),
    }), flush=True)


if __name__ == "__main__":
    main()
