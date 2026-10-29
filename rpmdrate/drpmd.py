"""Public direct-RPMD API and CLI, with orthogonal trajectory parallelism.

Physical sampling and propagation live in ``drpmd_core``. CPU PES transport is
independent of the outer trajectory scheduler and does not change Cayley rings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import drpmd_core as _core
from .drpmd_core import *  # Preserve the existing public Python API.
from .drpmd_scheduler import normalize_input, run_ensemble


def __getattr__(name):
    # Preserve imports of existing private numerical helpers as well.
    return getattr(_core, name)


def run(config, *, base=None, mode="serial", workers=1, output="drpmd-results",
        thermal_state=None, trajectory_workers=None, gpu_devices=None,
        trajectories_per_gpu=None, pes_backend=None, dynamics_backend=None,
        scheduler=None, task_prefetch=None):
    """Run N collisions, optionally distributed over independent parent chains.

    ``workers`` remains the PES/OpenMP worker count PER trajectory in bead mode.
    ``trajectory_workers`` is the persistent worker count (and thermal-chain count).
    Parallel ensembles default to a dynamic task pool; scheduler="static" retains
    the legacy fixed partition. task_prefetch bounds ready/in-flight samples.
    GPU devices are logical CUDA indices after CUDA_VISIBLE_DEVICES filtering.
    ``pes_backend`` optionally overrides potential.cpu_backend: process,
    shared_memory or native. Native requires potential.native_library.
    Multiprocessing callers must use an ``if __name__ == '__main__'`` guard.
    With no parallel options, the existing single-chain output/API is preserved.
    """
    config, options = normalize_input(config)
    if dynamics_backend is not None:
        config["dynamics_backend"] = dynamics_backend
    if config.get("dynamics_backend", "auto") not in ("auto", "python", "native"):
        raise ValueError("dynamics_backend must be auto, python or native")
    if pes_backend is not None:
        if not isinstance(config.get("potential"), dict):
            raise ValueError("Input must contain a potential object")
        config["potential"] = {**config["potential"], "cpu_backend": pes_backend}
    if isinstance(config.get("potential"), dict):
        backend = _core._cpu_pes_backend(config["potential"])
        if mode == "gpu" and backend != "process":
            raise ValueError("cpu_backend is a CPU-only option; GPU mode uses its CUDA PES")
    explicit = {"trajectory_workers": trajectory_workers, "gpu_devices": gpu_devices,
                "trajectories_per_gpu": trajectories_per_gpu,
                "scheduler": scheduler, "task_prefetch": task_prefetch}
    options.update({key: value for key, value in explicit.items() if value is not None})
    if not options:
        return _core.run(config, base=base, mode=mode, workers=workers,
                         output=output, thermal_state=thermal_state)
    return run_ensemble(config, base=base, mode=mode, workers=workers, output=output,
                        thermal_state=thermal_state, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Direct RPMD: thermal parents -> unthermostatted Cayley collisions")
    parser.add_argument("input", type=Path, help="JSON input; atom indices are zero-based")
    parser.add_argument("--mode", choices=("serial", "bead", "gpu"), default="serial")
    parser.add_argument("--workers", type=int, default=1,
                        help="PES processes/OpenMP threads PER trajectory in bead mode")
    parser.add_argument("--pes-backend", choices=("process", "shared_memory", "native"),
                        help="CPU PES transport; native also requires potential.native_library")
    parser.add_argument("--dynamics-backend", choices=("auto", "python", "native"),
                        help="Native C integration blocks require the native CPU PES; auto selects when available")
    parser.add_argument("--trajectory-workers", type=int,
                        help="Persistent task workers and independent physical thermal chains")
    parser.add_argument("--scheduler", choices=("dynamic", "static"),
                        help="Dynamic task pool (parallel default) or legacy fixed chain partition")
    parser.add_argument("--task-prefetch", type=int,
                        help="Maximum queued collision states plus active sampling tasks (default 2 x workers)")
    parser.add_argument("--gpu-devices", type=int, nargs="+", metavar="ID",
                        help="Logical CUDA device IDs, e.g. 0 1; gpu mode only")
    parser.add_argument("--trajectories-per-gpu", type=int,
                        help="Concurrent parent-chain processes per selected GPU")
    parser.add_argument("--output", type=Path, default=Path("drpmd-results"),
                        help="New output directory; existing files are never overwritten")
    parser.add_argument("--thermal-state", type=Path,
                        help="Single-chain state, or a previous ensemble output directory")
    parser.add_argument("--trajectories", type=int, help="Override TOTAL collision trajectories, not per worker")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.input.read_text())
        if args.trajectories is not None:
            if not isinstance(config, dict):
                raise ValueError("Input must be a JSON object")
            config["trajectories"] = args.trajectories
        summary = run(config, base=args.input.resolve().parent, mode=args.mode, workers=args.workers,
                      output=args.output, thermal_state=args.thermal_state,
                      trajectory_workers=args.trajectory_workers, gpu_devices=args.gpu_devices,
                      trajectories_per_gpu=args.trajectories_per_gpu, pes_backend=args.pes_backend,
                      dynamics_backend=args.dynamics_backend, scheduler=args.scheduler, task_prefetch=args.task_prefetch)
    except KeyboardInterrupt:
        print("dRPMD interrupted; workers stopped and partial ensemble results retained", file=sys.stderr)
        return 130
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, FloatingPointError) as exc:
        print(f"dRPMD error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    failed = (summary["counts"]["failed"] or summary.get("controller_error") or
              any(not report["ok"] for report in summary.get("worker_reports", [])))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
