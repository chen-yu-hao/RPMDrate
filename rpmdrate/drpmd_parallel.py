"""Spawned independent thermal-parent chains for all direct-RPMD PES modes.

One process owns one parent chain and its serial/bead/CUDA backend. It retains
that parent while launching copies as collision children. No CUDA objects or
mutable Fortran state are passed between processes. The coordinator alone
writes aggregate results; each child process has a disjoint output directory.
"""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import csv
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import warnings
import zipfile

import numpy as np


def _int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def normalize_input(config):
    """Copy input, removing execution settings and resolving a time interval."""
    if not isinstance(config, dict):
        raise ValueError("Input must be a JSON object")
    config = dict(config)
    options = config.pop("parallel", {})
    if not isinstance(options, dict):
        raise ValueError("parallel must be an object")
    allowed = {"trajectory_workers", "gpu_devices", "trajectories_per_gpu"}
    if set(options) - allowed:
        raise ValueError(f"Unknown parallel keys: {sorted(set(options) - allowed)}")
    if any(value is None for value in options.values()):
        raise ValueError("parallel settings must not be null; omit unused settings")
    if "sampling_time_fs" in config:
        duration, dt = float(config.pop("sampling_time_fs")), float(config["dt_fs"])
        if not math.isfinite(duration) or not math.isfinite(dt) or duration <= 0 or dt <= 0:
            raise ValueError("sampling_time_fs and dt_fs must be finite and positive")
        ratio = duration / dt
        if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8):
            raise ValueError("sampling_time_fs must be a positive integer multiple of dt_fs")
        steps = int(round(ratio))
        if "sampling_steps" in config and _int(config["sampling_steps"], "sampling_steps") != steps:
            raise ValueError("sampling_steps and sampling_time_fs disagree")
        config["sampling_steps"] = steps
    return config, dict(options)


@dataclass(frozen=True)
class Chain:
    chain_id: int
    first_trajectory: int
    trajectories: int
    seed: int
    gpu_device: int | None = None

    @property
    def directory(self):
        return f"chains/chain_{self.chain_id:04d}"


def make_plan(total, *, mode="serial", workers=1, trajectory_workers=None,
              gpu_devices=None, trajectories_per_gpu=None, seed=20261029):
    """Pure resource/count planning; this function never initializes CUDA."""
    total, workers, seed = _int(total, "trajectories"), _int(workers, "workers"), _int(seed, "seed", 0)
    if mode not in ("serial", "bead", "gpu"):
        raise ValueError("mode must be serial, bead or gpu")
    if (mode != "bead" and workers != 1) or (mode == "bead" and workers < 2):
        raise ValueError("workers must be >= 2 in bead mode and exactly 1 in serial/gpu mode")
    requested = None if trajectory_workers is None else _int(trajectory_workers, "trajectory_workers")
    devices = []
    if mode == "gpu":
        devices = [0] if gpu_devices is None else gpu_devices
        if not isinstance(devices, (list, tuple)) or not devices:
            raise ValueError("gpu_devices must be a nonempty list of logical CUDA indices")
        devices = [_int(device, "GPU device", 0) for device in devices]
        if len(set(devices)) != len(devices):
            raise ValueError("gpu_devices must not contain duplicates")
        if trajectories_per_gpu is not None:
            capacity = len(devices) * _int(trajectories_per_gpu, "trajectories_per_gpu")
            if requested is not None and requested != capacity:
                raise ValueError("trajectory_workers must equal len(gpu_devices) * trajectories_per_gpu")
            requested = capacity
        elif requested is None:
            requested = len(devices)
    elif gpu_devices is not None or trajectories_per_gpu is not None:
        raise ValueError("gpu_devices and trajectories_per_gpu require mode='gpu'")
    requested = 1 if requested is None else requested
    active = min(total, requested)
    quotient, remainder = divmod(total, active)
    chains, first = [], 1
    for index in range(active):
        count = quotient + (index < remainder)
        chain_seed = int(np.random.SeedSequence(seed, spawn_key=(index,)).generate_state(1, dtype=np.uint64)[0])
        chains.append(Chain(index + 1, first, count, chain_seed,
                            devices[index % len(devices)] if devices else None))
        first += count
    return {"mode": mode, "workers_per_trajectory": workers,
            "requested_trajectory_workers": requested, "active_trajectory_workers": active,
            "total_trajectories": total, "root_seed": seed, "gpu_devices": devices,
            "active_slots_per_gpu": {str(device): sum(c.gpu_device == device for c in chains) for device in devices},
            "cpu_compute_slots": active * workers, "chains": chains}


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _states(source, count):
    if source is None:
        return [None] * count
    if isinstance(source, (list, tuple)):
        if len(source) != count:
            raise ValueError("thermal_state must provide exactly one state per active parent chain")
        paths = [Path(item).expanduser().resolve() for item in source]
    else:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            paths = [path / f"chains/chain_{index:04d}/thermal_state.npz" for index in range(1, count + 1)]
        elif count == 1:
            paths = [path]
        else:
            raise ValueError("Parallel reuse requires a previous ensemble directory or a list of per-chain states; a single shared state is not an independent ensemble")
    if len(set(paths)) != len(paths):
        raise ValueError("Use a distinct thermal-state file for each parent chain")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing thermal state: {path}")
    return [str(path) for path in paths]


def _preflight(config, base, plan, states):
    """Reject invalid physics/input before starting processes or creating output."""
    from . import drpmd_core as core
    if not isinstance(config.get("potential"), dict):
        raise ValueError("Input must contain a potential object")
    allowed = {"atoms", "masses_amu", "fragments", "temperature_K", "nbeads", "dt_fs",
               "initial_geometry_bohr", "thermalization_steps", "sampling_steps", "andersen_interval",
               "trajectories", "evolution_steps", "initial_separation_bohr", "bmax_bohr", "seed",
               "bond_cutoffs_bohr", "channels", "exit_distance_bohr", "check_interval", "persistence",
               "save_every", "electronic_weight", "potential", "description",
               "dynamics_backend", "native_block_steps", "classifier_backend"}
    if set(config) - allowed:
        raise ValueError(f"Unknown input keys: {sorted(set(config) - allowed)}")
    if config.get("dynamics_backend", "auto") not in ("auto", "python", "native"):
        raise ValueError("dynamics_backend must be auto, python or native")
    if _int(config.get("native_block_steps", 256), "native_block_steps") > np.iinfo(np.int32).max:
        raise ValueError("native_block_steps exceeds int32")
    if config.get("dynamics_backend") == "native":
        from .drpmd_engine import engine_library
        if plan["mode"] == "gpu" or core._cpu_pes_backend(config["potential"]) != "native" or engine_library() is None:
            raise ValueError("Native dynamics requires a native CPU PES and a rebuilt C kernel")
    masses = core._masses(config["masses_amu"]) * core.AMU
    fragments = core.validate_fragments(config["fragments"], len(masses))
    temperature = core._positive(config["temperature_K"], "temperature_K")
    beads = _int(config["nbeads"], "nbeads")
    core._positive(config["dt_fs"], "dt_fs")
    for key in ("sampling_steps", "andersen_interval", "evolution_steps"):
        _int(config[key], key)
    for key, default in (("check_interval", 10), ("persistence", 3), ("save_every", 100)):
        _int(config.get(key, default), key)
    thermal = _int(config["thermalization_steps"], "thermalization_steps", 0)
    if any(path is None for path in states) and thermal < config["andersen_interval"]:
        raise ValueError("New initial states require thermalization_steps >= andersen_interval")
    separation = core._positive(config["initial_separation_bohr"], "initial_separation_bohr")
    if separation <= core._positive(config["bmax_bohr"], "bmax_bohr"):
        raise ValueError("initial_separation_bohr must exceed bmax_bohr")
    core._positive(config.get("electronic_weight", 1.), "electronic_weight", zero=True)
    core.ConnectivityClassifier(config["atoms"], fragments, masses, config["bond_cutoffs_bohr"],
                                config["channels"], config["exit_distance_bohr"])
    classifier_backend = config.get("classifier_backend", "auto")
    if classifier_backend not in ("auto", "python", "native"):
        raise ValueError("classifier_backend must be auto, python or native")
    if classifier_backend == "native" and (plan["mode"] == "gpu" or
        config.get("dynamics_backend", "auto") == "python" or
        core._cpu_pes_backend(config["potential"]) != "native"):
        raise ValueError("Native product checks require native CPU dynamics and PES")
    kind = config["potential"].get("type", "module")
    if kind not in ("hhoo", "module", "torchscript"):
        raise ValueError(f"Unknown potential type: {kind}")
    if kind == "hhoo" and config["atoms"] != ["H", "H", "O", "O"]:
        raise ValueError("The HHOO model requires atoms in H,H,O,O order")
    if (kind == "module" and plan["mode"] == "gpu") or (kind == "torchscript" and plan["mode"] != "gpu"):
        raise ValueError("GPU mode requires an HHOO or TorchScript PES; TorchScript requires GPU mode")
    inserted = str(base) not in sys.path
    if inserted:
        sys.path.insert(0, str(base))
    try:
        provenance, identity = core._potential_provenance(config, base)
    finally:
        if inserted:
            sys.path.remove(str(base))
    if any(path is None for path in states):
        geometry = np.asarray(config["initial_geometry_bohr"], dtype=float)
        if geometry.shape != (len(masses), 3) or not np.isfinite(geometry).all():
            raise ValueError("initial_geometry_bohr must be a finite Natoms x 3 array")
    for path in states:
        if path is None:
            continue
        state = core.ThermalState.load(path)
        if (state.metadata.get("phase") != "thermal" or state.metadata.get("potential_identity") != identity or
            state.metadata.get("thermal_separation_bohr") != separation or state.temperature != temperature or
            state.nbeads != beads or state.fragments != fragments or not np.array_equal(state.masses, masses)):
            raise ValueError("Saved thermal state must have matching potential, temperature, beads, masses, fragments and separation")
    if plan["mode"] == "bead":
        from .drpmd_integrator import _load_native
        if _load_native() is None:
            raise RuntimeError("Build the OpenMP Cayley kernel before using bead mode")
    if plan["mode"] == "gpu":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("GPU mode requires CUDA-enabled PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no usable CUDA device is available")
        count = torch.cuda.device_count()
        if any(device >= count for device in plan["gpu_devices"]):
            raise ValueError(f"gpu_devices must index the {count} visible CUDA devices (CUDA_VISIBLE_DEVICES is respected)")
    return provenance, identity


def _chain_entry(config, base, mode, workers, output, state, chain, status_path, log_path):
    """Top-level spawn target, deliberately non-daemonic for nested PES pools."""
    if os.name == "posix":
        os.setsid()  # Cancellation can terminate this chain AND its bead workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.update({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OMP_NUM_THREADS": str(workers if mode == "bead" else 1),
                       "NUMEXPR_NUM_THREADS": "1"})
    report = {"chain_id": chain.chain_id, "pid": os.getpid(), "gpu_device": chain.gpu_device,
              "started_monotonic": time.monotonic(), "ok": False, "error": None}
    with open(log_path, "x") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            if mode == "gpu":
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                torch.cuda.set_device(chain.gpu_device)
                report["cuda_current_device"] = torch.cuda.current_device()
                report["cuda_device_name"] = torch.cuda.get_device_name(chain.gpu_device)
            from .drpmd_core import run
            local = dict(config, trajectories=chain.trajectories, seed=chain.seed)
            summary = run(local, base=base, mode=mode, workers=workers, output=output, thermal_state=state)
            report["ok"] = summary["trajectories"] == chain.trajectories and not summary["counts"]["failed"]
            if not report["ok"]:
                report["error"] = "Parent chain stopped early or contained failed collision trajectories"
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["traceback"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            report["finished_monotonic"] = time.monotonic()
            _json(status_path, report)
    if not report["ok"]:
        raise SystemExit(1)


def _stop(process):
    """Stop a chain's entire process group on POSIX, including bead PES workers."""
    if not process.is_alive():
        # A killed leader can leave its nested bead workers behind. Their POSIX
        # process group retains the original leader PID until its last member exits.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        process.join()
        return
    def send(sig):
        if os.name == "posix":
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, sig)
                    return
            except ProcessLookupError:
                return
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    send(signal.SIGTERM)
    process.join(timeout=5)
    if process.is_alive():
        send(getattr(signal, "SIGKILL", signal.SIGTERM))
        process.join()


_FLOAT_FIELDS = ("collision_energy_hartree", "speed_au", "impact_parameter_bohr", "azimuth_rad",
                 "initial_energy_per_bead_hartree", "final_energy_per_bead_hartree",
                 "max_energy_drift_per_bead_hartree")


def collect_chain(root, chain, report):
    """Recover completed children and retain every missing assignment as unknown."""
    directory = Path(root) / chain.directory
    table = {}
    path = directory / "trajectories.csv"
    if path.is_file():
        try:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    table[int(row["trajectory"])] = row
        except (OSError, ValueError, KeyError, csv.Error):
            # Per-child NPZ files below can still recover completed trajectories.
            pass
    records = []
    for local in range(1, chain.trajectories + 1):
        prefix = directory / f"trajectory_{local:06d}"
        initial, final = Path(str(prefix) + "_initial.npz"), Path(str(prefix) + "_final.npz")
        row = table.get(local)
        recovered = False
        if row is None and final.is_file():
            try:
                with np.load(final, allow_pickle=False) as data:
                    row = json.loads(str(data["metadata"].item()))
                recovered = True
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                row = None
        record = {"trajectory": chain.first_trajectory + local - 1, "chain_id": chain.chain_id,
                  "local_trajectory": local, "chain_seed": chain.seed, "gpu_device": chain.gpu_device,
                  "status": "failed", "channel": None, "steps": 0, "launched": initial.is_file(),
                  "error": report.get("error") or "Assigned trajectory has no completed result",
                  "recovered_from_checkpoint": recovered,
                  "trace_file": str(prefix.relative_to(root)) + ".csv",
                  "initial_state": str(initial.relative_to(root)) if initial.is_file() else None,
                  "final_state": str(final.relative_to(root)) if final.is_file() else None}
        record.update({field: None for field in _FLOAT_FIELDS})
        if row is not None:
            try:
                if row["status"] not in ("product", "nonreactive", "unresolved", "failed"):
                    raise ValueError("Unknown outcome in chain record")
                values = {key: float(row[key]) if row.get(key) not in (None, "") else None for key in _FLOAT_FIELDS}
                if any(value is not None and not math.isfinite(value) for value in values.values()):
                    raise ValueError("Nonfinite value in chain record")
                record.update(values)
                record.update(status=row["status"], channel=row.get("channel") or None,
                              steps=int(row.get("steps", 0)), error=row.get("error") or None)
            except (ValueError, TypeError, KeyError) as exc:
                record.update(status="failed", channel=None, error=f"Invalid saved result: {exc}")
        record["failure_stage"] = ("not_launched" if not record["launched"] else "collision_or_worker") if record["status"] == "failed" else None
        records.append(record)
    return records


def run_ensemble(config, *, base=None, mode="serial", workers=1, output="drpmd-results",
                 thermal_state=None, trajectory_workers=None, gpu_devices=None,
                 trajectories_per_gpu=None):
    """Launch exactly min(N,n) independent chains, assigning N collisions in total."""
    from . import drpmd_core as core
    started = time.perf_counter()
    base = Path.cwd() if base is None else Path(base).expanduser().resolve()
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Output directory already exists: {root}")
    plan = make_plan(config["trajectories"], mode=mode, workers=workers, trajectory_workers=trajectory_workers,
                     gpu_devices=gpu_devices, trajectories_per_gpu=trajectories_per_gpu,
                     seed=config.get("seed", 20261029))
    chains = plan["chains"]
    states = _states(thermal_state, len(chains))
    provenance, identity = _preflight(config, base, plan, states)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if plan["cpu_compute_slots"] > cpus:
        warnings.warn(f"Requested {plan['cpu_compute_slots']} CPU compute slots but only {cpus} are available", RuntimeWarning)
    root.mkdir(parents=True)
    for name in ("chains", "logs", "worker_status"):
        (root / name).mkdir()
    manifest = {**plan, "chains": [asdict(chain) for chain in chains],
                "format": "RPMDrate.dRPMD.ensemble.v1", "input": config, "base": str(base),
                "source_thermal_states": states, "potential_identity": identity, "provenance": provenance,
                "sampling_time_fs": config["sampling_steps"] * config["dt_fs"],
                "start_method": "spawn", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    _json(root / "input.json", manifest)
    context = mp.get_context("spawn")
    processes, controller_error, interrupted = [], None, False
    print(f"Starting {len(chains)} independent parent chains for {config['trajectories']} total collisions ({mode})", flush=True)
    try:
        for chain, state in zip(chains, states):
            process = context.Process(target=_chain_entry,
                args=(config, str(base), mode, workers, str(root / chain.directory), state, chain,
                      str(root / "worker_status" / f"chain_{chain.chain_id:04d}.json"),
                      str(root / "logs" / f"chain_{chain.chain_id:04d}.log")),
                name=f"drpmd-parent-{chain.chain_id}", daemon=False)
            process.start()
            processes.append((chain, process))
        remaining = {process.sentinel: (chain, process) for chain, process in processes}
        while remaining:
            for sentinel in wait(list(remaining)):
                chain, process = remaining.pop(sentinel)
                process.join()
                _json(root / "progress.json", {"finished_chains": len(chains) - len(remaining),
                                              "total_chains": len(chains), "last_chain": chain.chain_id,
                                              "last_exitcode": process.exitcode})
    except KeyboardInterrupt:
        controller_error, interrupted = "Interrupted by user", True
    except Exception as exc:
        controller_error = f"{type(exc).__name__}: {exc}"
    finally:
        for _, process in processes:
            _stop(process)
    reports, records = [], []
    exits = {chain.chain_id: process.exitcode for chain, process in processes}
    for chain in chains:
        status_path = root / "worker_status" / f"chain_{chain.chain_id:04d}.json"
        try:
            report = json.loads(status_path.read_text())
        except (OSError, ValueError):
            report = {"chain_id": chain.chain_id, "ok": False,
                      "error": controller_error or f"Worker exited without a report (exitcode={exits.get(chain.chain_id)})"}
        report["exitcode"] = exits.get(chain.chain_id)
        reports.append(report)
        records.extend(collect_chain(root, chain, report))
    records.sort(key=lambda row: row["trajectory"])
    with (root / "trajectories.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _json(root / "trajectories.json", records)
    summary = core.summarize(records, config["channels"], config["temperature_K"],
                             np.asarray(config["masses_amu"]) * core.AMU, config["fragments"],
                             config["bmax_bohr"], config.get("electronic_weight", 1.))
    summary.update({"requested_trajectories": config["trajectories"],
                    "launched_trajectories": sum(record["launched"] for record in records),
                    "not_launched_trajectories": sum(not record["launched"] for record in records),
                    "mode": mode, "workers": workers, "trajectory_workers": len(chains),
                    "requested_trajectory_workers": plan["requested_trajectory_workers"],
                    "gpu_devices": plan["gpu_devices"], "active_slots_per_gpu": plan["active_slots_per_gpu"],
                    "parent_chains": [asdict(chain) for chain in chains], "worker_reports": reports,
                    "failed_parent_chains": sum(not report["ok"] for report in reports),
                    "temperature_K": config["temperature_K"], "nbeads": config["nbeads"],
                    "dt_fs": config["dt_fs"], "sampling_time_fs": config["sampling_steps"] * config["dt_fs"],
                    "potential_identity": identity, "potential": config["potential"],
                    "timing_seconds": {"wall": time.perf_counter() - started},
                    "controller_error": controller_error, "output_directory": str(root),
                    "parallel_note": "Each process owns an independent thermal parent. CUDA uses one model/context per process, not cross-trajectory tensor batching.",
                    "uncertainty_note": "Missing assignments remain failed/unknown in the requested-N denominator. Independent RNG streams do not establish equilibration or decorrelation. Wilson intervals assume iid children and do not include within-chain autocorrelation."})
    _json(root / "summary.json", summary)
    if interrupted:
        raise KeyboardInterrupt
    return summary
