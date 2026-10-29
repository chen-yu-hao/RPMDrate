"""Dynamic trajectory tasks with persistent *physical* thermal parent chains.

A fixed-size spawn pool keeps PES/native/CUDA contexts alive. Thermal q/p and
thermostat clocks belong to chain checkpoints, not worker processes. A chain
has at most one sampling task in flight; any free worker may advance it. The
resulting immutable collision initial states enter a bounded global queue.
No RNG state is saved, transferred, or restored. Each sampling task derives
fresh independent thermal/launch streams from its identity and the input seed.
"""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import csv
from dataclasses import asdict
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import warnings

import numpy as np

from . import drpmd_core as core
from .drpmd_parallel import _int, _json, _preflight, _states, _stop, make_plan


def _prefix(root, task):
    return Path(root) / f"chains/chain_{task['chain_id']:04d}/trajectory_{task['local_trajectory']:06d}"


def _atomic_state(state, path):
    """One writer per chain/task; readers see only a complete NPZ checkpoint."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        state.save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def task_rng(seed, chain_id, local, domain):
    """Task-local streams, not a persistent RNG chain or restored RNG checkpoint."""
    return np.random.default_rng(np.random.SeedSequence(
        seed, spawn_key=(0x4452504D, int(chain_id), int(local), int(domain))))


class TaskQueue:
    """Pure scheduler: fixed chain quotas, exclusive parent advancement, bounded IO.

    A pending collision is counted until dispatched. Sampling reservations count
    against prefetch too, so fast producers cannot grow an unbounded queue.
    Work is assigned one task at a time; there are no per-worker collision quotas.
    """
    def __init__(self, chains, sources, prefetch):
        self.chains = {c.chain_id: c for c in chains}
        self.sources = dict(zip(self.chains, sources))
        self.next_local = {cid: 1 for cid in self.chains}
        self.sampling = set()
        self.disabled = {}
        self.ready = deque()
        self.inflight = {}
        self.prefetch = _int(prefetch, "task_prefetch")
        self.max_buffered = 0
        self.completed = set()

    def take(self, worker_id):
        if worker_id in self.inflight:
            raise RuntimeError("Worker already owns an active task")
        available = [cid for cid, c in self.chains.items()
                     if cid not in self.disabled and cid not in self.sampling
                     and self.next_local[cid] <= c.trajectories]
        task = None
        if available and len(self.ready) + len(self.sampling) < self.prefetch:
            # Fair production across chains; completion speed cannot change each
            # chain's planned contribution to the statistical ensemble.
            cid = min(available, key=lambda c: (self.next_local[c], c))
            c, local = self.chains[cid], self.next_local[cid]
            task = {"kind": "sample", "chain_id": cid, "local_trajectory": local,
                    "trajectory": c.first_trajectory + local - 1,
                    "source": self.sources[cid], "chain_seed": c.seed}
            self.sampling.add(cid)
        elif self.ready:
            task = dict(self.ready.popleft(), kind="collision")
        if task is not None:
            self.inflight[worker_id] = task
            self.max_buffered = max(self.max_buffered, len(self.ready) + len(self.sampling))
        return task

    def finish_sample(self, worker_id, parent_path):
        task = self.inflight.pop(worker_id)
        if task["kind"] != "sample":
            raise RuntimeError("Unexpected sampling completion")
        cid = task["chain_id"]
        self.sampling.remove(cid)
        self.sources[cid] = str(parent_path)
        self.next_local[cid] += 1
        self.ready.append(task)

    def finish_collision(self, worker_id):
        task = self.inflight.pop(worker_id)
        if task["kind"] != "collision" or task["trajectory"] in self.completed:
            raise RuntimeError("Duplicate or unexpected collision completion")
        self.completed.add(task["trajectory"])

    def fail(self, worker_id, error):
        task = self.inflight.pop(worker_id, None)
        if task and task["kind"] == "sample":
            self.sampling.remove(task["chain_id"])
            self.disabled[task["chain_id"]] = error
        elif task:
            self.completed.add(task["trajectory"])
        return task

    @property
    def exhausted(self):
        return not self.ready and not self.inflight and all(
            cid in self.disabled or self.next_local[cid] > c.trajectories
            for cid, c in self.chains.items())


class _Runtime:
    """One persistent backend, reusable for any chain or collision task."""
    def __init__(self, config, base, root, mode, workers, slot, identity, provenance, stack):
        self.config, self.root, self.slot = config, Path(root), slot
        self.identity, self.provenance = identity, provenance
        self.masses = core._masses(config["masses_amu"]) * core.AMU
        self.fragments = core.validate_fragments(config["fragments"], len(self.masses))
        self.temperature = config["temperature_K"]
        self.dt = config["dt_fs"] / core.TIME_FS
        self.potential = stack.enter_context(core._make_potential(config, Path(base), mode, workers))
        self.dynamics = core.Dynamics(self.masses, self.temperature, config["nbeads"], self.dt,
                                     self.potential, threads=workers if mode == "bead" else 1,
                                     backend=config.get("dynamics_backend", "auto"),
                                     block_steps=config.get("native_block_steps", 256))
        if self.dynamics.engine is not None:
            stack.callback(self.dynamics.engine.close)
        self.classifier = core.ConnectivityClassifier(
            config["atoms"], self.fragments, self.masses, config["bond_cutoffs_bohr"],
            config["channels"], config["exit_distance_bohr"])

    def sample(self, task):
        config, dynamics = self.config, self.dynamics
        thermal_rng = task_rng(config.get("seed", 20261029), task["chain_id"], task["local_trajectory"], 0)
        launch_rng = task_rng(config.get("seed", 20261029), task["chain_id"], task["local_trajectory"], 1)
        if task["source"]:
            state = core.ThermalState.load(task["source"])
            if (state.metadata.get("phase") != "thermal" or state.metadata.get("potential_identity") != self.identity
                    or state.temperature != self.temperature or state.nbeads != config["nbeads"]
                    or state.fragments != self.fragments or not np.array_equal(state.masses, self.masses)
                    or state.metadata.get("thermal_separation_bohr") != config["initial_separation_bohr"]):
                raise ValueError("Thermal checkpoint has mismatched physical parameters")
            if task["local_trajectory"] > 1 and (state.metadata.get("chain_id") != task["chain_id"]
                    or state.metadata.get("sample_index") != task["local_trajectory"] - 1):
                raise ValueError("Stale thermal parent checkpoint or overlapping sampling task")
            q, p = state.q, state.p
            clock = _int(state.metadata.get("thermal_steps", 0), "thermal_steps", 0)
        else:
            geometry = np.asarray(config["initial_geometry_bohr"], dtype=float)
            q = np.asfortranarray(np.repeat(geometry.T[:, :, None], config["nbeads"], axis=2))
            for frag, center in zip(self.fragments, (np.zeros(3), np.array([0., 0., config["initial_separation_bohr"]]))):
                q[:, list(frag), :] += (center - core.fragment_com(q, self.masses, frag))[:, None, None]
            p = core.sample_momenta(self.masses, config["nbeads"], dynamics.beta, thermal_rng)
            _, clock = dynamics.thermalize(q, p, self.fragments, config["thermalization_steps"],
                                          config["andersen_interval"], thermal_rng)
        # Imported first samples retain the existing no-hidden-rethermalization
        # contract. Subsequent tasks always advance the physical parent.
        if task["local_trajectory"] > 1 or not task["source"]:
            _, clock = dynamics.thermalize(q, p, self.fragments, config["sampling_steps"],
                                          config["andersen_interval"], thermal_rng, clock)
        self.classifier.validate_reactants(q)
        metadata = {"phase": "thermal", "chain_id": task["chain_id"],
                    "sample_index": task["local_trajectory"], "thermal_steps": clock,
                    "potential_identity": self.identity, "provenance": self.provenance,
                    "thermal_separation_bohr": config["initial_separation_bohr"],
                    "thermal_dt_fs": config["dt_fs"], "rng_policy": "task-local; no saved RNG state"}
        parent = core.ThermalState(q, p, self.temperature, self.masses, self.fragments, metadata)
        directory = _prefix(self.root, task).parent
        _atomic_state(parent, directory / "thermal_latest.npz")
        if task["local_trajectory"] == 1:
            _atomic_state(parent, directory / "thermal_state.npz")
        launch_q, launch_p, sample = core.launch_collision(
            parent, config["bmax_bohr"], config["initial_separation_bohr"], launch_rng)
        initial_metadata = {**sample, "phase": "collision_initial", "trajectory": task["trajectory"],
                            "chain_id": task["chain_id"], "local_trajectory": task["local_trajectory"],
                            "potential_identity": self.identity, "parent_thermal_steps": clock,
                            "sampling_worker": self.slot.chain_id, "sampling_gpu_device": self.slot.gpu_device}
        prefix = _prefix(self.root, task)
        _atomic_state(core.ThermalState(launch_q, launch_p, self.temperature, self.masses, self.fragments,
                                       initial_metadata), str(prefix) + "_initial.npz")
        receipt = {**initial_metadata, "parent_path": str(directory / "thermal_latest.npz")}
        # Commit marker is written LAST. A crash before this point never causes
        # a partly prepared initial condition to be dispatched.
        _json(str(prefix) + "_sample.json", receipt)
        return receipt

    def collision(self, task):
        prefix = _prefix(self.root, task)
        initial = core.ThermalState.load(str(prefix) + "_initial.npz")
        record = {"trajectory": task["trajectory"], "chain_id": task["chain_id"],
                  "local_trajectory": task["local_trajectory"], "worker_id": self.slot.chain_id,
                  "gpu_device": self.slot.gpu_device, "status": "failed", "channel": None,
                  "steps": 0, "error": None,
                  **{name: initial.metadata[name] for name in ("collision_energy_hartree", "speed_au",
                       "impact_parameter_bohr", "azimuth_rad", "parent_thermal_steps", "sampling_worker")},
                  "initial_energy_per_bead_hartree": None, "final_energy_per_bead_hartree": None,
                  "max_energy_drift_per_bead_hartree": None}
        with Path(str(prefix) + ".csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "time_fs", "ring_energy_per_bead_hartree"] +
                            [f"{a}{i}_{axis}_bohr" for i, a in enumerate(self.config["atoms"]) for axis in "xyz"])
            def observe(step, q, p, energy):
                writer.writerow([step, step * self.config["dt_fs"], energy, *q.mean(axis=2).T.ravel()])
            try:
                result = core.propagate(self.dynamics, initial.q, initial.p, self.config["evolution_steps"],
                                        self.classifier, self.config.get("check_interval", 10),
                                        self.config.get("persistence", 3), self.config.get("save_every", 100), observe)
                record.update(status=result.status, channel=result.channel, steps=result.steps, error=result.error,
                              initial_energy_per_bead_hartree=result.initial_energy_per_bead,
                              final_energy_per_bead_hartree=result.final_energy_per_bead,
                              max_energy_drift_per_bead_hartree=result.max_energy_drift_per_bead)
                _atomic_state(core.ThermalState(result.q, result.p, self.temperature, self.masses, self.fragments,
                              {**initial.metadata, **record, "phase": "collision_final"}), str(prefix) + "_final.npz")
            except Exception as exc:
                record.update(status="failed", channel=None, error=f"{type(exc).__name__}: {exc}")
        _json(str(prefix) + "_result.json", record)
        return record


def _worker(config, base, root, mode, workers, slot, identity, provenance, pipe):
    """Non-daemonic worker; native/CUDA/PES pools initialize once per process."""
    if os.name == "posix":
        os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.update({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                       "OMP_NUM_THREADS": str(workers if mode == "bead" else 1)})
    sys.path.insert(0, str(base))
    report = {"worker_id": slot.chain_id, "pid": os.getpid(), "gpu_device": slot.gpu_device,
              "ok": False, "error": None, "sample_tasks": 0, "collision_tasks": 0,
              "busy_seconds": 0., "started_monotonic": time.monotonic()}
    path = Path(root) / "worker_status" / f"worker_{slot.chain_id:04d}.json"
    try:
        with (Path(root) / "logs" / f"worker_{slot.chain_id:04d}.log").open("x") as log, \
                redirect_stdout(log), redirect_stderr(log), ExitStack() as stack:
            from threadpoolctl import threadpool_limits
            stack.enter_context(threadpool_limits(limits=1, user_api="blas"))
            if mode == "gpu":
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                torch.cuda.set_device(slot.gpu_device)
                report.update(cuda_current_device=torch.cuda.current_device(),
                              cuda_device_name=torch.cuda.get_device_name(slot.gpu_device))
            runtime = _Runtime(config, base, root, mode, workers, slot, identity, provenance, stack)
            report.update(dynamics_backend=runtime.dynamics.backend,
                          pes_worker_pids=list(getattr(runtime.potential, "worker_pids", ())))
            pipe.send(("ready", {**report}))
            while True:
                task = pipe.recv()
                if task is None:
                    report["ok"] = True
                    break
                start = time.perf_counter()
                payload = runtime.sample(task) if task["kind"] == "sample" else runtime.collision(task)
                report["busy_seconds"] += time.perf_counter() - start
                report["sample_tasks" if task["kind"] == "sample" else "collision_tasks"] += 1
                fatal = task["kind"] == "collision" and payload["status"] == "failed"
                pipe.send(("sampled" if task["kind"] == "sample" else "completed", payload))
                if fatal:
                    report["error"] = payload["error"] or "PES/trajectory failed; retiring backend"
                    break
            report["native_block_calls"] = runtime.dynamics.engine.calls if runtime.dynamics.engine else 0
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        try:
            pipe.send(("error", {"error": report["error"]}))
        except (OSError, EOFError):
            pass
    finally:
        report["finished_monotonic"] = time.monotonic()
        _json(path, report)
        pipe.close()


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _records(root, plan, errors):
    """The denominator is the requested N, never the number of returned tasks."""
    records = []
    for chain in plan["chains"]:
        for local in range(1, chain.trajectories + 1):
            task = {"trajectory": chain.first_trajectory + local - 1,
                    "chain_id": chain.chain_id, "local_trajectory": local}
            prefix = _prefix(root, task)
            sample = _read_json(str(prefix) + "_sample.json") or {}
            saved = _read_json(str(prefix) + "_result.json") or {}
            record = {**task, "worker_id": saved.get("worker_id"), "gpu_device": saved.get("gpu_device"),
                      "sampling_worker": sample.get("sampling_worker"), "parent_thermal_steps": sample.get("parent_thermal_steps"),
                      "status": "failed", "channel": None, "steps": 0,
                      "error": errors.get(task["trajectory"], "No completed result; not counted as nonreactive"),
                      "launched": Path(str(prefix) + ".csv").exists(),
                      "initial_state": str(prefix.relative_to(root)) + "_initial.npz" if sample else None,
                      "final_state": str(prefix.relative_to(root)) + "_final.npz" if Path(str(prefix) + "_final.npz").exists() else None,
                      "trace_file": str(prefix.relative_to(root)) + ".csv"}
            for name in ("collision_energy_hartree", "speed_au", "impact_parameter_bohr", "azimuth_rad",
                         "initial_energy_per_bead_hartree", "final_energy_per_bead_hartree", "max_energy_drift_per_bead_hartree"):
                record[name] = saved.get(name, sample.get(name))
            if saved:
                if saved.get("trajectory") != task["trajectory"] or saved.get("status") not in ("failed", "unresolved", "product", "nonreactive"):
                    record["error"] = "Invalid collision receipt"
                else:
                    for name in ("status", "channel", "steps", "error"):
                        record[name] = saved.get(name)
            record["failure_stage"] = ("collision_or_worker" if record["launched"] else "not_launched") if record["status"] == "failed" else None
            records.append(record)
    return records


def run_dynamic(config, *, base=None, mode="serial", workers=1, output="drpmd-results",
                thermal_state=None, trajectory_workers=None, gpu_devices=None,
                trajectories_per_gpu=None, task_prefetch=None):
    """A bounded dynamic pool; q/p continuity survives changes in executing worker."""
    started = time.perf_counter()
    root = Path(output).expanduser().resolve()
    base = Path.cwd() if base is None else Path(base).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Output directory already exists: {root}")
    plan = make_plan(config["trajectories"], mode=mode, workers=workers, trajectory_workers=trajectory_workers,
                     gpu_devices=gpu_devices, trajectories_per_gpu=trajectories_per_gpu, seed=config.get("seed", 20261029))
    chains = plan["chains"]
    limit = min(config["trajectories"], _int(task_prefetch, "task_prefetch") if task_prefetch is not None else 2 * len(chains))
    sources = _states(thermal_state, len(chains))
    # Dynamic restarts continue the LAST physical parent, not its first sample.
    if thermal_state is not None and not isinstance(thermal_state, (list, tuple)) and Path(thermal_state).expanduser().is_dir():
        sources = [str(Path(p).with_name("thermal_latest.npz")) if Path(p).with_name("thermal_latest.npz").is_file() else p for p in sources]
    provenance, identity = _preflight(config, base, plan, sources)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if plan["cpu_compute_slots"] > cpus:
        warnings.warn(f"Requested {plan['cpu_compute_slots']} compute slots; only {cpus} logical CPUs visible", RuntimeWarning)
    root.mkdir(parents=True)
    for name in ("chains", "logs", "worker_status"):
        (root / name).mkdir()
    for chain in chains:
        (root / chain.directory).mkdir()
    manifest = {**plan, "chains": [asdict(c) for c in chains], "input": config, "base": str(base),
                "format": "RPMDrate.dRPMD.dynamic.v1", "scheduler": "dynamic", "task_prefetch": limit,
                "source_thermal_states": sources, "potential_identity": identity, "provenance": provenance,
                "rng_policy": "fresh task-local streams; RNG state is never saved or transferred",
                "start_method": "spawn", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    _json(root / "input.json", manifest)
    queue = TaskQueue(chains, sources, limit)
    context = mp.get_context("spawn")
    processes, pipes, live, idle, stopping = {}, {}, set(), set(), set()
    errors, dispatches, reports = {}, {}, {}
    controller_error, interrupted = None, False
    print(f"Dynamic pool: {len(chains)} workers, {len(chains)} physical parent chains, {config['trajectories']} total collisions", flush=True)
    timeline = (root / "scheduler.jsonl").open("x")

    def event(kind, wid, task=None, **extra):
        timeline.write(json.dumps({"event": kind, "worker_id": wid, "time_seconds": time.perf_counter()-started,
                                   **({k: task[k] for k in ("kind", "trajectory", "chain_id", "local_trajectory")} if task else {}),
                                   **extra}, allow_nan=False) + "\n")
        timeline.flush()

    def lost(wid, message):
        task = queue.inflight.get(wid)
        if task:
            prefix = _prefix(root, task)
            receipt = _read_json(str(prefix) + ("_sample.json" if task["kind"] == "sample" else "_result.json"))
            if receipt and receipt.get("trajectory") == task["trajectory"]:
                if task["kind"] == "sample":
                    queue.finish_sample(wid, receipt["parent_path"])
                else:
                    queue.finish_collision(wid)
                event("recovered", wid, task)
            else:
                queue.fail(wid, message)
                errors[task["trajectory"]] = message
        reports.setdefault(wid, {"worker_id": wid, "pid": processes[wid].pid})
        reports[wid].update(ok=False, error=message)
        live.discard(wid)
        idle.discard(wid)
        event("worker_failed", wid, task, error=message)

    try:
        for slot in chains:
            parent, child = context.Pipe()
            process = context.Process(target=_worker,
                args=(config, str(base), str(root), mode, workers, slot, identity, provenance, child),
                name=f"drpmd-worker-{slot.chain_id}", daemon=False)
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            wid = slot.chain_id
            processes[wid], pipes[wid] = process, parent
            live.add(wid)
        while live:
            for wid in sorted(idle):
                task = queue.take(wid)
                if task is not None:
                    idle.remove(wid)
                    dispatches[task["trajectory"], task["kind"]] = wid
                    event("dispatch", wid, task)
                    try:
                        pipes[wid].send(task)
                    except (OSError, EOFError):
                        lost(wid, "Worker connection closed during dispatch")
            # No queued or future samples: idle workers cannot shorten the last
            # already-running trajectory. Stop them rather than busy-spinning.
            if queue.exhausted:
                for wid in sorted(idle):
                    try:
                        pipes[wid].send(None)
                        stopping.add(wid)
                    except (OSError, EOFError):
                        lost(wid, "Worker connection closed during shutdown")
                idle.clear()
            handles = [pipes[w] for w in live] + [processes[w].sentinel for w in live]
            if not handles:
                break
            ready = wait(handles, timeout=0.5)
            for wid in list(live):
                if pipes[wid] not in ready:
                    continue
                task = queue.inflight.get(wid)
                try:
                    tag, payload = pipes[wid].recv()
                except (OSError, EOFError):
                    if wid in stopping:
                        live.remove(wid)
                    else:
                        lost(wid, "Worker exited without a complete task reply")
                    continue
                if tag == "ready":
                    reports[wid] = payload
                    idle.add(wid)
                    event("ready", wid, pid=payload["pid"])
                elif tag == "sampled":
                    if task is None or task["kind"] != "sample" or payload.get("trajectory") != task["trajectory"]:
                        raise RuntimeError("Mismatched sampling task reply")
                    queue.finish_sample(wid, payload["parent_path"])
                    idle.add(wid)
                    event("sampled", wid, task)
                elif tag == "completed":
                    if task is None or task["kind"] != "collision" or payload.get("trajectory") != task["trajectory"]:
                        raise RuntimeError("Mismatched collision reply")
                    queue.finish_collision(wid)
                    event("completed", wid, task, status=payload["status"])
                    if payload["status"] == "failed":
                        stopping.add(wid)  # Worker retires its possibly poisoned backend.
                    else:
                        idle.add(wid)
                elif tag == "error":
                    lost(wid, payload["error"])
                else:
                    raise RuntimeError(f"Unknown pool message {tag!r}")
            # Drain messages before interpreting process death. A completed
            # receipt remains recoverable even if the final pipe reply was lost.
            for wid in list(live):
                if processes[wid].sentinel in ready:
                    if pipes[wid].poll():
                        continue
                    if wid in stopping:
                        live.remove(wid)
                    else:
                        lost(wid, f"Worker exited (code {processes[wid].exitcode})")
            _json(root / "progress.json", {"total_trajectories": config["trajectories"],
                "completed_collisions": len(queue.completed), "ready_collisions": len(queue.ready),
                "active_tasks": len(queue.inflight), "live_workers": len(live),
                "failed_parent_chains": len(queue.disabled)})
        if not queue.exhausted:
            controller_error = "All workers exited before the requested ensemble finished"
    except KeyboardInterrupt:
        controller_error, interrupted = "Interrupted by user", True
    except Exception as exc:
        controller_error = f"{type(exc).__name__}: {exc}"
    finally:
        for wid, process in processes.items():
            if wid in stopping and not interrupted and controller_error is None:
                process.join(timeout=5)  # Let Python/native contexts finish normal teardown.
            _stop(process)
            pipes[wid].close()
        timeline.close()
    records = _records(root, plan, errors)
    for row in records:
        if row["worker_id"] is None:
            row["worker_id"] = dispatches.get((row["trajectory"], "collision"))
    for slot in chains:
        wid = slot.chain_id
        saved = _read_json(root / "worker_status" / f"worker_{wid:04d}.json")
        report = saved or reports.get(wid) or {"worker_id": wid, "ok": False, "error": controller_error or "Worker did not start"}
        report["exitcode"] = processes[wid].exitcode if wid in processes else None
        if report["exitcode"] not in (None, 0):
            report.update(ok=False, error=report.get("error") or f"Worker exit code {report['exitcode']}")
        reports[wid] = report
    with (root / "trajectories.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _json(root / "trajectories.json", records)
    for chain in chains:
        subset = [r for r in records if r["chain_id"] == chain.chain_id]
        clock = max((r["parent_thermal_steps"] or 0 for r in subset), default=0)
        _json(root / chain.directory / "summary.json", {"chain_id": chain.chain_id,
              "requested_trajectories": chain.trajectories, "thermal_steps_at_last_sample": clock,
              "scheduler": "dynamic", "sampling_error": queue.disabled.get(chain.chain_id)})
    summary = core.summarize(records, config["channels"], config["temperature_K"],
                              np.asarray(config["masses_amu"]) * core.AMU, config["fragments"],
                              config["bmax_bohr"], config.get("electronic_weight", 1.))
    summary.update({"requested_trajectories": config["trajectories"], "scheduler": "dynamic", "task_prefetch": limit,
                    "max_buffered_tasks": queue.max_buffered, "mode": mode, "workers": workers,
                    "trajectory_workers": len(chains), "requested_trajectory_workers": plan["requested_trajectory_workers"],
                    "gpu_devices": plan["gpu_devices"], "active_slots_per_gpu": plan["active_slots_per_gpu"],
                    "parent_chains": [asdict(c) for c in chains], "worker_reports": list(reports.values()),
                    "failed_parent_chains": len(queue.disabled), "failed_workers": sum(not r["ok"] for r in reports.values()),
                    "launched_trajectories": sum(r["launched"] for r in records),
                    "not_launched_trajectories": sum(not r["launched"] for r in records),
                    "temperature_K": config["temperature_K"], "nbeads": config["nbeads"], "dt_fs": config["dt_fs"],
                    "sampling_time_fs": config["sampling_steps"] * config["dt_fs"],
                    "potential_identity": identity, "potential": config["potential"],
                    "timing_seconds": {"wall": time.perf_counter()-started}, "output_directory": str(root),
                    "controller_error": controller_error, "rng_policy": manifest["rng_policy"],
                    "parallel_note": "Parent q/p and thermostat clock persist independently of workers. Fixed per-chain sample counts; dynamic collision assignment. No RNG checkpoint. CUDA contexts remain pinned to worker devices.",
                    "uncertainty_note": "Every requested trajectory remains in the denominator; missing results are unknown/failed. Thermal correlations and convergence are not included in iid intervals."})
    _json(root / "summary.json", summary)
    if interrupted:
        raise KeyboardInterrupt
    return summary
