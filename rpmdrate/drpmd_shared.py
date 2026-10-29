"""Fixed-buffer, spawned-process bead PES transport.

Only (generation, status, pid) messages cross the pipes during evaluation. q, V
and dV/dq live in one shared-memory allocation owned by the trajectory process.
Each worker has an isolated legacy PES and a disjoint, fixed bead interval.
The public result owns its memory: the next call cannot invalidate a gradient
still needed by velocity Verlet/Cayley. This is reduced IPC, not zero-copy MD.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing.connection import wait
from multiprocessing.shared_memory import SharedMemory
import numbers
import os
import threading
import time

import numpy as np

from .drpmd_backends import (
    PotentialError, PotentialSpec, SerialPotential, _PotentialContext,
    _validated_result,
)


def _shape(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 3 or
        any(isinstance(i, bool) or not isinstance(i, numbers.Integral) or i < 1 for i in value) or
        value[0] != 3):
        raise ValueError("shape must be (3, positive Natoms, positive Nbeads)")
    return tuple(int(i) for i in value)


def _buffers(memory, shape):
    size = math.prod(shape)
    q = np.ndarray(shape, dtype=np.float64, order="F", buffer=memory.buf)
    energy = np.ndarray((shape[2],), dtype=np.float64, buffer=memory.buf, offset=8*size)
    gradient = np.ndarray(shape, dtype=np.float64, order="F", buffer=memory.buf,
                          offset=8*(size+shape[2]))
    return q, energy, gradient


def _worker(spec, name, shape, start, end, pipe):
    # Independent process/module state is required for non-reentrant legacy PES.
    memory = potential = limiter = None
    q = energy = gradient = None
    generation = 0
    try:
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
        memory = SharedMemory(name=name)
        q, energy, gradient = _buffers(memory, shape)
        potential = SerialPotential(spec)
        pipe.send(("ready", 0, os.getpid()))
        while True:
            message = pipe.recv()
            if message is None:
                break
            generation = message
            values, deriv = potential.evaluate(q[:, :, start:end])
            energy[start:end] = values
            gradient[:, :, start:end] = deriv
            pipe.send(("done", generation, os.getpid()))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            pipe.send(("error", generation, f"{type(exc).__name__}: {exc}"[:4096]))
        except (OSError, EOFError):
            pass
    finally:
        if potential is not None:
            potential.close()
        q = energy = gradient = None
        if memory is not None:
            memory.close()  # Only the owning trajectory unlinks the allocation.
        if limiter is not None:
            limiter.restore_original_limits()
        pipe.close()


class SharedMemoryPotential(_PotentialContext):
    """One long-lived process per bead interval; no ndarray queue payloads.

    shape is fixed for the lifetime of this backend. Calls are serialized within
    an instance; parallel trajectories must each own an instance. Startup and
    evaluation failures close every worker and unlink the allocation. The timeout
    covers a complete batch, not each worker separately. Increase timeout for
    expensive ab initio evaluations; it must be positive and finite.
    """

    def __init__(self, spec, workers, shape, startup_timeout=60.0, timeout=300.0):
        if not isinstance(spec, PotentialSpec):
            raise TypeError("SharedMemoryPotential requires an importable PotentialSpec")
        if isinstance(workers, bool) or not isinstance(workers, numbers.Integral) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.shape = _shape(shape)
        for name, value in (("startup_timeout", startup_timeout), ("timeout", timeout)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.spec, self.workers = spec, min(int(workers), self.shape[2])
        self.timeout = float(timeout)
        self.closed = False
        self.worker_pids = self.last_worker_pids = ()
        self._lock = threading.RLock()
        self._generation = 0
        self._processes, self._pipes = [], []
        self._memory = None
        self._q = self._energy = self._gradient = None
        try:
            self._memory = SharedMemory(create=True, size=8*(2*math.prod(self.shape)+self.shape[2]))
            self.shared_memory_name = self._memory.name
            self._q, self._energy, self._gradient = _buffers(self._memory, self.shape)
            context = mp.get_context("spawn")
            bounds = np.linspace(0, self.shape[2], self.workers+1, dtype=int)
            for start, end in zip(bounds[:-1], bounds[1:]):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(
                    spec, self._memory.name, self.shape, int(start), int(end), child))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._pipes.append(parent)
                self._processes.append(process)
            self.worker_pids = tuple(self._receive("ready", 0, float(startup_timeout)))
        except BaseException as exc:
            self._close(abort=True)
            if not isinstance(exc, Exception):
                raise
            raise PotentialError(f"Failed to start shared-memory PES workers: {exc}") from exc

    def _receive(self, tag, generation, timeout):
        pending = set(range(len(self._pipes)))
        deadline = time.monotonic()+timeout
        pids = [None]*len(self._pipes)
        while pending:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise PotentialError(f"Shared-memory PES {tag} timed out after {timeout:g} s")
            handles = [self._pipes[i] for i in pending] + [self._processes[i].sentinel for i in pending]
            ready = wait(handles, timeout=remaining)
            if not ready:
                raise PotentialError(f"Shared-memory PES {tag} timed out after {timeout:g} s")
            # Consume pipe data before death notifications: a dying worker may
            # have provided an actionable PES exception immediately before exit.
            for i in tuple(pending):
                if self._pipes[i] in ready:
                    try:
                        message = self._pipes[i].recv()
                    except EOFError as exc:
                        raise PotentialError(f"PES worker {i} exited without a result") from exc
                    if message[0] == "error":
                        raise PotentialError(f"PES worker {i}: {message[2]}")
                    if message[:2] != (tag, generation):
                        raise PotentialError(f"Unexpected shared-memory PES reply: {message}")
                    pids[i] = message[2]
                    pending.remove(i)
            for i in pending:
                if self._processes[i].sentinel in ready:
                    raise PotentialError(f"PES worker {i} exited (code {self._processes[i].exitcode})")
        return pids

    def evaluate(self, q):
        with self._lock:
            self._check_open()
            coordinates = np.asarray(q)
            if np.iscomplexobj(coordinates) or coordinates.shape != self.shape:
                raise ValueError(f"PES coordinates must be real and have fixed shape {self.shape}")
            coordinates = np.asarray(coordinates, dtype=np.float64, order="F")
            if not np.isfinite(coordinates).all():
                raise ValueError("PES coordinates must all be finite")
            np.copyto(self._q, coordinates)
            self._energy.fill(np.nan)
            self._gradient.fill(np.nan)
            self._generation += 1
            try:
                for pipe in self._pipes:
                    pipe.send(self._generation)
                self.last_worker_pids = tuple(self._receive("done", self._generation, self.timeout))
                return _validated_result((self._energy, self._gradient), self.shape)
            except BaseException as exc:
                self._close(abort=True)
                if not isinstance(exc, Exception):
                    raise
                raise PotentialError(f"Shared-memory PES evaluation failed: {exc}") from exc

    def _close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        for pipe in self._pipes:
            if not abort:
                try:
                    pipe.send(None)
                except (OSError, EOFError, BrokenPipeError):
                    pass
        for process in self._processes:
            if abort and process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join()
        for pipe in self._pipes:
            pipe.close()
        self._q = self._energy = self._gradient = None
        if self._memory is not None:
            self._memory.close()
            try:
                self._memory.unlink()
            except FileNotFoundError:
                pass
            self._memory = None

    def close(self):
        with self._lock:
            self._close()
