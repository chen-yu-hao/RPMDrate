"""Public direct-RPMD API and CLI, with orthogonal trajectory parallelism.

The previously validated single-chain implementation is preserved verbatim in
``drpmd_core``. Its physical sampling, propagator and PES interfaces are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import drpmd_core as _core
from .drpmd_core import *  # Preserve the existing public Python API.
from .drpmd_parallel import normalize_input, run_ensemble


def __getattr__(name):
    # Preserve imports of existing private numerical helpers as well.
    return getattr(_core, name)


def run(config, *, base=None, mode="serial", workers=1, output="drpmd-results",
        thermal_state=None, trajectory_workers=None, gpu_devices=None,
        trajectories_per_gpu=None):
    """Run N collisions, optionally distributed over independent parent chains.

    ``workers`` remains the PES/OpenMP worker count PER trajectory in bead mode.
    ``trajectory_workers`` is the number of independent parent-chain processes.
    GPU devices are logical CUDA indices after CUDA_VISIBLE_DEVICES filtering.
    Multiprocessing callers must use an ``if __name__ == '__main__'`` guard.
    With no parallel options, the existing single-chain output/API is preserved.
    """
    config, options = normalize_input(config)
    explicit = {"trajectory_workers": trajectory_workers, "gpu_devices": gpu_devices,
                "trajectories_per_gpu": trajectories_per_gpu}
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
    parser.add_argument("--trajectory-workers", type=int,
                        help="Concurrent independent parent chains, sharing the TOTAL trajectory count")
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
                      trajectories_per_gpu=args.trajectories_per_gpu)
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
