"""Checked Python boundary for native integration blocks (not a per-step loop).

The C loop calls the native C/Fortran PES function pointer directly. Python is
retained for input, RNG batches, custom classifiers, sparse output and scheduling.
Default product checks and persistence run inside C integration blocks.
No CUDA/Python PES is silently moved to CPU. Numerical arrays are atomic units.
"""
from __future__ import annotations

import ctypes as ct
import threading
import weakref

import numpy as np

from .drpmd_backends import PotentialError
from .drpmd_integrator import _load_native, _pointer, _positive_integer
from .drpmd_native_pes import NativePotential

_D = ct.POINTER(ct.c_double)
_I = ct.POINTER(ct.c_int32)


def _ints(array):
    return None if array is None else array.ctypes.data_as(_I)


def _doubles(array):
    return None if array is None else _pointer(array)


def engine_library():
    lib = _load_native()
    if lib is None or not hasattr(lib, "drpmd_engine_abi_version"):
        return None
    lib.drpmd_engine_abi_version.restype = ct.c_int32
    if lib.drpmd_engine_abi_version() != 1:
        raise RuntimeError("Unsupported native dynamics ABI; rebuild drpmd_integrator")
    lib.drpmd_engine_create.argtypes = [ct.c_int32, ct.c_int32, _D, ct.c_double,
                                       ct.c_double, ct.c_int32, ct.c_int32, _I]
    lib.drpmd_engine_create.restype = ct.c_void_p
    lib.drpmd_engine_destroy.argtypes = [ct.c_void_p]
    lib.drpmd_engine_destroy.restype = None
    lib.drpmd_engine_project.argtypes = [ct.c_void_p, _D, _I]
    lib.drpmd_engine_project.restype = ct.c_int32
    lib.drpmd_engine_start.argtypes = [ct.c_void_p, ct.c_void_p, _D, _D, _D, _D, _D, _I]
    lib.drpmd_engine_start.restype = ct.c_int32
    lib.drpmd_engine_advance.argtypes = [
        ct.c_void_p, ct.c_void_p, _D, _D, _D, _D, ct.c_int32, _I, _D, _D,
        ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double, _D, _D, _I, _I]
    lib.drpmd_engine_advance.restype = ct.c_int32
    return lib


def select_engine(dynamics, backend):
    if backend not in ("auto", "python", "native"):
        raise ValueError("dynamics_backend must be auto, python or native")
    if backend == "python":
        return None
    supported = type(dynamics.potential) is NativePotential
    lib = engine_library() if supported else None
    if backend == "native" and (not supported or lib is None):
        raise RuntimeError("Native dynamics requires NativePotential and a rebuilt kernel: "
                           "python -m rpmdrate.drpmd_integrator --build")
    return NativeEngine(dynamics, lib) if supported and lib is not None else None


class NativeEngine:
    """One immutable parameter set and independent C workspace per trajectory."""

    def __init__(self, dynamics, lib):
        self.dynamics = weakref.proxy(dynamics)
        self.potential = dynamics.potential  # Keep the PES DSO and model alive.
        self.lib = lib
        self.shape = (3, len(dynamics.masses), dynamics.nbeads)
        self.lock = threading.RLock()
        code = ct.c_int32()
        self.context = lib.drpmd_engine_create(
            self.shape[1], self.shape[2], _pointer(dynamics.masses), dynamics.beta,
            dynamics.dt, dynamics.integrator.threads, self.potential.threads, ct.byref(code))
        if not self.context:
            raise RuntimeError(f"Native dynamics initialization failed (status={code.value})")
        self._finalizer = weakref.finalize(self, lib.drpmd_engine_destroy, self.context)
        self.function = ct.cast(self.potential._lib.drpmd_pes_eval, ct.c_void_p)
        self.calls = 0
        self.classifier_checks = 0

    def _check(self, q, p):
        if not self._finalizer.alive:
            raise RuntimeError("Native dynamics engine is closed")
        for name, array in (("q", q), ("p", p)):
            if not isinstance(array, np.ndarray) or array.dtype != np.float64:
                raise TypeError(f"{name} must be an FP64 numpy array")
            if (array.shape != self.shape or not array.flags.f_contiguous or
                    not array.flags.aligned or not array.flags.writeable):
                raise ValueError(f"{name} must be writable, aligned, Fortran contiguous with shape {self.shape}")
        if np.shares_memory(q, p):
            raise ValueError("q and p must not share memory")
        self.potential._check_open()

    def _error(self, code, detail):
        if code == 7:
            return f"Native product classification failed (status={detail[1]})"
        return (f"Native propagation status={code}, bad_bead={detail[0]}, "
                f"PES_status={detail[1]}, PES_threads={detail[2]}")

    def start(self, q, p, group=None):
        with self.lock, self.potential._lock:
            self._check(q, p)
            if group is not None:
                raw = np.asarray(group)
                if raw.shape != (self.shape[1],) or raw.dtype.kind not in "iu" or not np.isin(raw, [0, 1]).all():
                    raise ValueError("Invalid native fragment partition")
                group = np.require(raw, dtype=np.int32, requirements=["C", "A"])
                rc = self.lib.drpmd_engine_project(self.context, _pointer(p), _ints(group))
                if rc:
                    raise ValueError("Invalid native fragment partition")
            values = np.empty(self.shape[2], dtype=np.float64)
            gradient = np.empty(self.shape, dtype=np.float64, order="F")
            energy = ct.c_double()
            detail = np.zeros(3, dtype=np.int32)
            rc = self.lib.drpmd_engine_start(self.context, self.function, _pointer(q), _pointer(p),
                                            _pointer(values), _pointer(gradient), ct.byref(energy), _ints(detail))
            self.potential.last_threads_used = int(detail[2])
            if rc:
                raise PotentialError(self._error(rc, detail))
            return values, gradient, energy.value

    def advance(self, q, p, values, gradient, steps, reference, energy, drift=0.,
                group=None, targets=None, refresh=None, first_refresh=0, interval=0,
                classifier=None, start_step=0, check_interval=10, persistence=3):
        steps = _positive_integer(steps, "native block steps")
        with self.lock, self.potential._lock:
            self._check(q, p)
            for name, a, shape in (("values", values, (self.shape[2],)), ("gradient", gradient, self.shape)):
                if (not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.shape != shape or
                        not a.flags.f_contiguous or not a.flags.aligned or not a.flags.writeable):
                    raise ValueError(f"Invalid native {name} buffer")
            arrays = (q, p, values, gradient)
            if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
                raise ValueError("Native state/result buffers must not overlap")
            if group is not None:
                group = np.require(group, dtype=np.int32, requirements=["C", "A"])
                targets = np.require(targets, dtype=np.float64, requirements=["C", "A"])
                if group.shape != (self.shape[1],) or targets.shape != (2, 3):
                    raise ValueError("Invalid native constraint arrays")
            count = 0
            if refresh is not None:
                refresh = np.require(refresh, dtype=np.float64, requirements=["C", "A"])
                if refresh.ndim != 4 or refresh.shape[1:] != self.shape[::-1]:
                    raise ValueError("Refresh blocks must have shape (draws, beads, atoms, 3)")
                count = len(refresh)
            if count:
                _positive_integer(count, "refresh count")
                first_refresh = _positive_integer(first_refresh, "first_refresh")
                interval = _positive_integer(interval, "interval")
            else:
                first_refresh = interval = 0
            value, maximum = ct.c_double(energy), ct.c_double(drift)
            done = ct.c_int32()
            detail = np.zeros(3, dtype=np.int32)
            if classifier is None:
                rc = self.lib.drpmd_engine_advance(
                    self.context, self.function, _pointer(q), _pointer(p), _pointer(values), _pointer(gradient),
                    steps, _ints(group), _doubles(targets), _doubles(refresh), first_refresh, interval, count,
                    reference, ct.byref(value), ct.byref(maximum), ct.byref(done), _ints(detail))
            else:
                if group is not None or refresh is not None:
                    raise ValueError("Product checks are only valid during unthermostatted collisions")
                if (isinstance(start_step, (bool, np.bool_)) or not isinstance(start_step, (int, np.integer)) or
                        not 0 <= start_step <= np.iinfo(np.int64).max-steps):
                    raise ValueError("start_step must be a nonnegative int64 with room for the block")
                check_interval = _positive_integer(check_interval, "check_interval")
                persistence = _positive_integer(persistence, "persistence")
                label = ct.c_int32(-1)
                with classifier.lock:
                    classifier._check_open()
                    before = classifier.checks
                    rc = self.lib.drpmd_engine_advance_classified(
                        self.context, self.function, _pointer(q), _pointer(p), _pointer(values), _pointer(gradient),
                        steps, reference, ct.byref(value), ct.byref(maximum), ct.byref(done), _ints(detail),
                        classifier.context, start_step, check_interval, persistence, ct.byref(label))
                    self.classifier_checks += classifier.checks-before
                self.calls += 1
                self.potential.last_threads_used = int(detail[2])
                return (done.value, value.value, maximum.value, self._error(rc, detail) if rc else None,
                        classifier.label(label.value))
            self.calls += 1
            self.potential.last_threads_used = int(detail[2])
            return done.value, value.value, maximum.value, self._error(rc, detail) if rc else None

    def thermalize(self, q, p, fragments, steps, interval, rng, clock):
        from .drpmd_core import CentroidCOMConstraints
        constraints = CentroidCOMConstraints(q, self.dynamics.masses, fragments)
        group = np.empty(self.shape[1], dtype=np.int32)
        for f, atoms in enumerate(constraints.fragments):
            group[list(atoms)] = f
        targets = np.array(constraints.targets, dtype=np.float64, order="C")
        values, gradient, energy = self.start(q, p, group)
        end = clock+steps
        max_draws = max(1, (32*1024*1024)//(8*q.size))
        while clock < end:
            n = min(end-clock, self.dynamics.block_steps, max_draws*interval)
            first = interval-clock % interval
            count = (n-first)//interval+1 if n >= first else 0
            # Batch calls to the same NumPy Generator retain the scalar API's
            # draw order, unlike replacing the RNG with an unrelated C generator.
            refresh = None
            if count:
                refresh = rng.normal(size=(count, *self.shape))
                refresh *= np.sqrt(self.dynamics.masses[None, None, :, None]*self.shape[2]/self.dynamics.beta)
                refresh = np.ascontiguousarray(refresh.transpose(0, 3, 2, 1))
            done, energy, _, error = self.advance(q, p, values, gradient, n, energy, energy,
                group=group, targets=targets, refresh=refresh, first_refresh=first, interval=interval)
            clock += done
            if error:
                raise PotentialError(f"Thermalization failed at step {clock+1}: {error}")
        return values, clock

    def close(self):
        with self.lock:
            self._finalizer()


def propagate_native(dynamics, q, p, steps, classifier, check_interval, persistence, save_every, observer):
    from .drpmd_core import TrajectoryResult
    from .drpmd_classifier import select_classifier
    engine = dynamics.engine
    native = select_classifier(dynamics, classifier)
    try:
        values, gradient, initial = engine.start(q, p)
        energy, drift, step = initial, 0.0, 0
        candidate, streak = None, 0
        if observer is not None:
            observer(0, q, p, energy)
        while step < steps:
            count = min(steps-step, dynamics.block_steps)
            # Native checks do NOT cut the C block. Only custom Python callbacks,
            # observation boundaries and the interrupt cap return control here.
            if classifier is not None and native is None:
                count = min(count, check_interval-step % check_interval)
            if observer is not None:
                count = min(count, save_every-step % save_every)
            label, resolved = None, False
            if native is not None:
                done, energy, drift, error, label = engine.advance(
                    q, p, values, gradient, count, initial, energy, drift,
                    classifier=native, start_step=step, check_interval=check_interval, persistence=persistence)
                resolved = label is not None
            else:
                done, energy, drift, error = engine.advance(q, p, values, gradient, count, initial, energy, drift)
            step += done
            if error:
                return TrajectoryResult("failed", None, step, initial, energy, drift, q, p, error)
            if native is None and classifier is not None and step % check_interval == 0:
                label = classifier(q, p)
                streak = streak+1 if label is not None and label == candidate else (1 if label else 0)
                candidate = label
                resolved = label is not None and streak >= persistence
            if observer is not None and (step % save_every == 0 or step == steps or resolved):
                observer(step, q, p, energy)
            if resolved:
                return TrajectoryResult("nonreactive" if label == "reactants" else "product",
                                        None if label == "reactants" else label, step, initial, energy, drift, q, p)
        return TrajectoryResult("unresolved", None, steps, initial, energy, drift, q, p)
    finally:
        if native is not None:
            native.close()
