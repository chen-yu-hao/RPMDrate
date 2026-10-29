"""Potential-energy backends for direct RPMD trajectories.

Coordinates are FP64 arrays with shape ``(3, atoms, beads)`` in bohr.  Every
backend returns ``(energy, gradient)`` in Hartree and Hartree/bohr, respectively.
The second array is **positive dV/dq**, not force.  A potential must evaluate
each bead independently; splitting the bead axis must not change its values.

The process backend isolates legacy Fortran module state.  It distributes only
PES evaluations: free-ring dynamics must retain the complete cyclic bead ring.
PyTorch is imported only when a :class:`TorchPotential` is constructed.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import importlib
import multiprocessing
import numbers
import os
from pathlib import Path

import numpy as np


class PotentialError(RuntimeError):
    """A potential could not provide valid energies and gradients."""


class BackendUnavailableError(PotentialError):
    """The explicitly requested execution backend is not available."""


@dataclass(frozen=True)
class PotentialSpec:
    """An importable CPU potential specification, safe to send to spawn workers.

    ``function`` is called as ``function(q, **kwargs)`` by default.  Set
    ``with_dimensions=True`` for an older f2py interface requiring
    ``function(q, natoms, nbeads, **kwargs)``.  Both ``(V, dVdq)`` and
    ``(V, dVdq, info)`` results are supported; every nonzero ``info`` is an error.

    ``initializer`` is called once in each backend process with
    ``initializer_kwargs``.  It must return ``None`` or a finite scalar zero
    status.  Use ``initializer=None`` when none is needed.
    Potential data paths should be absolute, so spawned processes can load the
    same data without changing the application's working directory.
    """

    module: str
    function: str = "get_potential"
    initializer: str | None = "initialize_potential"
    kwargs: dict = field(default_factory=dict)
    initializer_kwargs: dict = field(default_factory=dict)
    with_dimensions: bool = False

    def __post_init__(self):
        for name in ("module", "function"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("PotentialSpec.%s must be a nonempty string" % name)
        if self.initializer is not None and (
            not isinstance(self.initializer, str) or not self.initializer.strip()
        ):
            raise ValueError("initializer must be a nonempty function name or None")
        if not isinstance(self.kwargs, dict) or not isinstance(self.initializer_kwargs, dict):
            raise TypeError("potential and initializer kwargs must be dictionaries")
        if not isinstance(self.with_dimensions, bool):
            raise TypeError("with_dimensions must be boolean")
        if self.initializer is None and self.initializer_kwargs:
            raise ValueError("initializer_kwargs requires an initializer")
        # Keep later caller edits to their dictionaries from changing a running
        # backend or making serial/process initialization differ.
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "initializer_kwargs", dict(self.initializer_kwargs))


def _coordinates(q):
    array = np.asarray(q)
    if np.iscomplexobj(array):
        raise ValueError("PES coordinates must be real")
    if array.ndim != 3 or array.shape[0] != 3 or min(array.shape[1:]) < 1:
        raise ValueError("PES coordinates must have shape (3, atoms, beads), with positive sizes")
    try:
        array = np.array(array, dtype=np.float64, order="F", copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("PES coordinates must be numerical FP64 values") from exc
    if not np.isfinite(array).all():
        raise ValueError("PES coordinates must all be finite")
    return array


def _validate_status(status, operation="PES"):
    info = np.asarray(status)
    if info.ndim != 0 or info.dtype.kind not in "biuf" or not np.isfinite(info):
        raise PotentialError("%s info must be a finite scalar status" % operation)
    if info != 0:
        raise PotentialError("%s reported nonzero status info=%s" % (operation, info.item()))


def _validated_result(result, shape):
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise PotentialError("PES must return (energy, gradient) or (energy, gradient, info)")
    if len(result) == 3:
        _validate_status(result[2])
    energy, gradient = np.asarray(result[0]), np.asarray(result[1])
    if energy.shape != (shape[2],):
        raise PotentialError("PES energy shape %s does not match (%d,)" % (energy.shape, shape[2]))
    if gradient.shape != shape:
        raise PotentialError("PES gradient shape %s does not match %s" % (gradient.shape, shape))
    if energy.dtype.kind not in "biuf" or gradient.dtype.kind not in "biuf":
        raise PotentialError("PES energies and gradients must be real numerical arrays")
    energy = np.array(energy, dtype=np.float64, copy=True)
    gradient = np.array(gradient, dtype=np.float64, order="F", copy=True)
    if not np.isfinite(energy).all() or not np.isfinite(gradient).all():
        raise PotentialError("PES returned nonfinite energies or gradients")
    return energy, gradient


class _PotentialContext:
    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _check_open(self):
        if self.closed:
            raise PotentialError("PES backend is closed")

    def close(self):
        self.closed = True


class SerialPotential(_PotentialContext):
    """Call one imported or Python-callable potential in the current process."""

    def __init__(self, potential):
        self.closed = False
        self.spec = potential if isinstance(potential, PotentialSpec) else None
        self._kwargs = {}
        self._with_dimensions = False
        if self.spec is not None:
            spec = self.spec
            try:
                module = importlib.import_module(spec.module)
                self._potential = getattr(module, spec.function)
                if not callable(self._potential):
                    raise TypeError("%s.%s is not callable" % (spec.module, spec.function))
                if spec.initializer is not None:
                    initializer = getattr(module, spec.initializer)
                    if not callable(initializer):
                        raise TypeError("%s.%s is not callable" % (spec.module, spec.initializer))
                    status = initializer(**spec.initializer_kwargs)
                    if status is not None:
                        _validate_status(status, "PES initialization")
                self._kwargs = dict(spec.kwargs)
                self._with_dimensions = spec.with_dimensions
            except Exception as exc:
                raise PotentialError("Failed to initialize PES %s: %s" % (spec.module, exc)) from exc
        elif callable(potential):
            self._potential = potential
        else:
            raise TypeError("SerialPotential requires a PotentialSpec or callable")

    def evaluate(self, q):
        self._check_open()
        coordinates = _coordinates(q)
        try:
            if self._with_dimensions:
                result = self._potential(
                    coordinates, coordinates.shape[1], coordinates.shape[2], **self._kwargs
                )
            else:
                result = self._potential(coordinates, **self._kwargs)
        except Exception as exc:
            raise PotentialError("PES evaluation failed: %s" % exc) from exc
        return _validated_result(result, coordinates.shape)


# These values exist separately in each spawned worker process.  No potential
# instance or Fortran cache is shared between concurrent PES evaluations.
_WORKER_POTENTIAL = None
_WORKER_START_BARRIER = None
_WORKER_THREAD_LIMITER = None


def _initialize_worker(spec, barrier, startup_timeout):
    global _WORKER_POTENTIAL, _WORKER_START_BARRIER, _WORKER_THREAD_LIMITER
    # Do not create nested BLAS/OpenMP pools for each PES process.  Environment
    # variables handle libraries loaded after this initializer; threadpoolctl
    # also limits libraries already imported by Python's spawn bootstrap.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        pass
    else:
        _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    _WORKER_POTENTIAL = SerialPotential(spec)
    _WORKER_START_BARRIER = (barrier, startup_timeout)


def _worker_ready():
    barrier, timeout = _WORKER_START_BARRIER
    # One task must reach this barrier in each worker.  Thus even very fast
    # initializers cannot let one worker consume all the readiness tasks.
    barrier.wait(timeout=timeout)
    return os.getpid()


def _evaluate_chunk(start, coordinates):
    if _WORKER_POTENTIAL is None:
        raise PotentialError("PES worker was not initialized")
    energy, gradient = _WORKER_POTENTIAL.evaluate(coordinates)
    return start, energy, gradient, os.getpid()


class BeadParallelPotential(_PotentialContext):
    """Evaluate bead chunks in persistent, independently initialized processes.

    Use this from an importable program under ``if __name__ == '__main__':``;
    Python's ``spawn`` multiprocessing method cannot start from a stdin script.
    ``worker_pids`` records all initialized workers. ``last_worker_pids`` records
    which worker evaluated each chunk in the most recent successful call.
    """

    def __init__(self, spec, workers, startup_timeout=60.0):
        if not isinstance(spec, PotentialSpec):
            raise TypeError("BeadParallelPotential requires an importable PotentialSpec")
        if isinstance(workers, bool) or not isinstance(workers, numbers.Integral) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not np.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive and finite")
        self.spec = spec
        self.workers = int(workers)
        self.closed = False
        self.last_worker_pids = ()
        self.worker_pids = ()
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(self.workers)
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(spec, barrier, float(startup_timeout)),
        )
        futures = []
        try:
            futures = [self._executor.submit(_worker_ready) for _ in range(self.workers)]
            self.worker_pids = tuple(sorted(future.result(timeout=startup_timeout + 5) for future in futures))
            if len(set(self.worker_pids)) != self.workers:
                raise PotentialError("Not all PES workers initialized independently")
        except BaseException as exc:
            barrier.abort()
            for future in futures:
                future.cancel()
            self.close()
            if not isinstance(exc, Exception):
                raise
            raise PotentialError("Failed to start PES workers: %s" % exc) from exc

    def evaluate(self, q):
        self._check_open()
        coordinates = _coordinates(q)
        nbeads = coordinates.shape[2]
        nchunks = min(self.workers, nbeads)
        boundaries = np.linspace(0, nbeads, nchunks + 1, dtype=int)
        futures = []
        try:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                futures.append(self._executor.submit(
                    _evaluate_chunk, int(start), np.asfortranarray(coordinates[:, :, start:end])
                ))
            energy = np.empty(nbeads, dtype=np.float64)
            gradient = np.empty(coordinates.shape, dtype=np.float64, order="F")
            pids = []
            # Ordered retrieval and explicit offsets preserve bead indexing even
            # when chunk execution finishes out of order.
            for future in futures:
                start, chunk_energy, chunk_gradient, pid = future.result()
                end = start + chunk_energy.size
                energy[start:end] = chunk_energy
                gradient[:, :, start:end] = chunk_gradient
                pids.append(pid)
            self.last_worker_pids = tuple(pids)
            return _validated_result((energy, gradient), coordinates.shape)
        except BaseException as exc:
            for future in futures:
                future.cancel()
            self.close()
            if not isinstance(exc, Exception):
                raise
            raise PotentialError("Parallel PES evaluation failed: %s" % exc) from exc

    def close(self):
        if not self.closed:
            self.closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


class TorchPotential(_PotentialContext):
    """A batched FP64 differentiable PyTorch PES, normally evaluated on CUDA.

    ``model`` is a ``torch.nn.Module`` or a TorchScript path.  Its forward input
    has shape ``(beads, atoms, 3)`` and its output must contain one independent
    energy per bead, shape ``(beads,)`` or ``(beads, 1)``.  Model parameters and
    buffers are converted to FP64 and evaluation mode; stochastic/training or
    cross-bead models are unsuitable for molecular dynamics.

    ``length_scale`` converts bohr into the model's coordinate units, while
    ``energy_scale`` converts model energies into Hartree.  Autograd applies the
    corresponding chain rule to return dV/dq in Hartree/bohr.  For example, an
    Angstrom/eV model uses ``length_scale=0.529177210903`` and
    ``energy_scale=1/27.211386245988``.

    CUDA is required by default and never silently replaced by CPU.  Explicit
    ``device='cpu'`` is available for reference and gradient verification.
    """

    def __init__(self, model, device="cuda", energy_scale=1.0, length_scale=1.0):
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError(
                "TorchPotential requires PyTorch; install a CUDA-enabled build for GPU execution"
            ) from exc
        self._torch = torch
        self.closed = False
        try:
            self.device = torch.device(device)
        except Exception as exc:
            raise ValueError("Invalid PyTorch device %r" % device) from exc
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("TorchPotential supports CUDA, or explicit CPU reference execution")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailableError("CUDA was requested, but no usable CUDA device is available")
        for name, scale in (("energy_scale", energy_scale), ("length_scale", length_scale)):
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("%s must be positive and finite" % name)
        self.energy_scale = float(energy_scale)
        self.length_scale = float(length_scale)
        try:
            if isinstance(model, (str, bytes, os.PathLike, Path)):
                model = torch.jit.load(os.fspath(model), map_location=self.device)
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model must be a torch.nn.Module or a TorchScript path")
            self.model = model.to(device=self.device, dtype=torch.float64).eval()
        except Exception as exc:
            raise PotentialError("Failed to load PyTorch PES on %s: %s" % (self.device, exc)) from exc

    def evaluate(self, q):
        self._check_open()
        coordinates = _coordinates(q)
        torch = self._torch
        try:
            # enable_grad also makes the interface robust when its caller uses
            # torch.no_grad elsewhere; inference_mode is intentionally invalid.
            with torch.enable_grad():
                positions = torch.tensor(
                    coordinates.transpose(2, 1, 0).copy(),
                    dtype=torch.float64,
                    device=self.device,
                    requires_grad=True,
                )
                energy = self.model(positions * self.length_scale)
                if not isinstance(energy, torch.Tensor):
                    raise PotentialError("PyTorch PES must return an energy Tensor")
                if energy.shape == (coordinates.shape[2], 1):
                    energy = energy[:, 0]
                if tuple(energy.shape) != (coordinates.shape[2],):
                    raise PotentialError("PyTorch PES must return one energy per bead")
                if energy.dtype != torch.float64:
                    raise PotentialError("PyTorch PES energies must remain FP64")
                if energy.device != positions.device:
                    raise PotentialError("PyTorch PES moved its energies off the requested device")
                energy = energy * self.energy_scale
                if not energy.requires_grad:
                    raise PotentialError("PyTorch PES energy is detached from the coordinate autograd graph")
                gradient, = torch.autograd.grad(energy.sum(), positions, create_graph=False)
                output = (
                    energy.detach().cpu().numpy(),
                    gradient.detach().cpu().numpy().transpose(2, 1, 0),
                )
        except PotentialError:
            raise
        except Exception as exc:
            raise PotentialError("PyTorch PES evaluation failed: %s" % exc) from exc
        return _validated_result(output, coordinates.shape)
