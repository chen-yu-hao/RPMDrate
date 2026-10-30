"""Compare production dynamic/static schedulers using known uneven task costs.

This is deliberately a synthetic scheduler benchmark, NOT HHOO speed or physics.
Full driver wall times include startup, thermalization, checkpoint/trace IO and
shutdown. Each completed task runs a short free-particle trajectory as well as
its prescribed delay. The same total collision costs are used for both modes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rpmdrate.drpmd import run
from tests.drpmd_paralleltest import tiny_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--slow-seconds", type=float, default=0.6)
    parser.add_argument("--fast-seconds", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("scheduler-benchmark.json"))
    args = parser.parse_args()
    if min(args.workers, args.trajectories, args.rounds) < 1 or min(args.slow_seconds, args.fast_seconds) <= 0:
        parser.error("Counts and delays must be positive")
    if args.output.exists():
        parser.error("Output already exists")
    quotient, remainder = divmod(args.trajectories, min(args.workers, args.trajectories))
    heavy = quotient + bool(remainder)
    cases = {"balanced": [args.fast_seconds]*args.trajectories,
             "imbalanced": [args.slow_seconds]*heavy + [args.fast_seconds]*(args.trajectories-heavy)}
    rows = []
    with tempfile.TemporaryDirectory(prefix="drpmd-scheduling-") as temporary:
        for name, costs in cases.items():
            config = tiny_config(args.trajectories)
            config["potential"] = {"type": "module", "module": "benchmarks.scheduling.workload",
                                   "initializer_kwargs": {"costs": costs, "chains": args.workers}}
            for repeat in range(args.rounds):
                # Alternate ordering to reduce systematic warm-cache/order bias.
                for scheduler in (("static", "dynamic") if repeat % 2 == 0 else ("dynamic", "static")):
                    directory = Path(temporary)/f"{name}-{repeat}-{scheduler}"
                    start = time.perf_counter()
                    result = run(config, base=ROOT, scheduler=scheduler, trajectory_workers=args.workers, output=directory)
                    elapsed = time.perf_counter()-start
                    if result["counts"]["failed"] or result.get("controller_error") or any(not r["ok"] for r in result["worker_reports"]):
                        raise RuntimeError(f"Benchmark failed: {result}")
                    counts = ([r["collision_tasks"] for r in result["worker_reports"]] if scheduler == "dynamic"
                              else [c["trajectories"] for c in result["parent_chains"]])
                    row = {"case": name, "round": repeat, "scheduler": scheduler, "wall_seconds": elapsed,
                           "collision_tasks_per_worker": counts, "counts": result["counts"]}
                    if scheduler == "dynamic":
                        row["timeline"] = [json.loads(line) for line in (directory/"scheduler.jsonl").read_text().splitlines()]
                    rows.append(row)
                    print(f"{name} {scheduler}: {elapsed:.3f} s; tasks={counts}", flush=True)
    summary = {}
    for name in cases:
        times = {s: statistics.median(r["wall_seconds"] for r in rows if r["case"] == name and r["scheduler"] == s)
                 for s in ("static", "dynamic")}
        summary[name] = {**times, "dynamic_speedup": times["static"]/times["dynamic"]}
    report = {"format": "RPMDrate.scheduler.benchmark.v1", "synthetic": True,
              "note": "Prescribed sleep per collision, not a PES/chemical speed benchmark. Full driver wall time, including startup and file IO. No guaranteed performance threshold.",
              "python": sys.version, "platform": platform.platform(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "workers": args.workers, "trajectories": args.trajectories, "costs_seconds": cases,
              "summary": summary, "raw": rows}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
