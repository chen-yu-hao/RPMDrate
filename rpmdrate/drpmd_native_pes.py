"""Synchronous FP64 C ABI for reentrant, bead-parallel C/Fortran potentials.

See examples/native_pes/drpmd_pes.h. No multiprocessing, ndarray serialization,
or f2py is involved. Already contiguous FP64 coordinates are passed directly;
outputs are allocated by Python and remain valid after subsequent evaluations.
A conforming native library MUST treat q and initialized model data as read-only.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import numbers
from pathlib import Path
import threading

import numpy as np

from .drpmd_backends import PotentialError, BackendUnavailableError, _PotentialContext

# Prevent concurrent first-time initialization and mutation of a frozen model.
# Retain CDLL handles for process lifetime, as a library may own immutable data.
_LIBRARIES = {}
_INIT_LOCK = threading.Lock()
_DBL = ct.POINTER(ct.c_double)
_INT = ct.POINTER(ct.c_int32)
_LIMIT = np.iinfo(np.int32).max


def _digest(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


class NativePotential(_PotentialContext):
    """A v1 native PES with one serial initializer and an OpenMP batch entry.

    This adapter cannot make arbitrary legacy code thread-safe. The library must
    implement the ABI and be audited for read-only model data / local workspaces.
    Non-reentrant potentials should use SharedMemoryPotential instead.
    """

    def __init__(self, library, model=None, threads=1):
        if isinstance(threads, bool) or not isinstance(threads, numbers.Integral) or not 1 <= threads <= _LIMIT:
            raise ValueError("threads must be a positive int32")
        self.closed = False
        self.threads = int(threads)
        self.library = str(Path(library).expanduser().resolve())
        self.model = "" if model is None else str(Path(model).expanduser().resolve())
        encoded = self.model.encode("utf-8")
        if b"\0" in encoded or len(encoded) > 4095:
            raise ValueError("Native model path must be NUL-free and at most 4095 UTF-8 bytes")
        identity = (_digest(self.library), _digest(self.model) if self.model else None)
        self.worker_pids = self.last_worker_pids = ()
        self.last_threads_used = 0
        # Serialize calls on this instance, including close; no lock is required
        # between different instances after immutable model initialization.
        self._lock = threading.RLock()
        with _INIT_LOCK:
            cached = _LIBRARIES.get(self.library)
            if cached is not None:
                if cached[0] != identity:
                    raise PotentialError("Native library/model bytes changed in this process; use a fresh process")
                self._lib = cached[1]
            else:
                try:
                    lib = ct.CDLL(self.library)
                    lib.drpmd_pes_abi_version.argtypes = []
                    lib.drpmd_pes_abi_version.restype = ct.c_int32
                    if lib.drpmd_pes_abi_version() != 1:
                        raise PotentialError("Unsupported native PES ABI version (expected 1)")
                    lib.drpmd_pes_omp_enabled.argtypes = []
                    lib.drpmd_pes_omp_enabled.restype = ct.c_int32
                    lib.drpmd_pes_init.argtypes = [ct.c_char_p]
                    lib.drpmd_pes_init.restype = ct.c_int32
                    lib.drpmd_pes_eval.argtypes = [ct.c_int32, ct.c_int32, _DBL, _DBL, _DBL,
                                                  ct.c_int32, _INT, _INT]
                    lib.drpmd_pes_eval.restype = ct.c_int32
                    status = lib.drpmd_pes_init(encoded)
                    if status:
                        raise PotentialError(f"Native PES initialization status {status}")
                except (AttributeError, OSError) as exc:
                    raise BackendUnavailableError(f"Cannot load native PES ABI: {exc}") from exc
                _LIBRARIES[self.library] = (identity, lib)
                self._lib = lib
        if self.threads > 1 and not self._lib.drpmd_pes_omp_enabled():
            raise BackendUnavailableError("Native PES was not built with OpenMP; use threads=1 or rebuild")

    def evaluate(self, q):
        with self._lock:
            self._check_open()
            q = np.asarray(q)
            if np.iscomplexobj(q) or q.ndim != 3 or q.shape[0] != 3 or min(q.shape[1:]) < 1:
                raise ValueError("Native PES coordinates must have real shape (3, atoms, beads)")
            if max(q.shape[1:]) > _LIMIT:
                raise ValueError("Native PES dimensions exceed the int32 ABI")
            q = np.require(q, dtype=np.float64, requirements=["F", "A"])
            if not np.isfinite(q).all():
                raise ValueError("PES coordinates must all be finite")
            energy = np.full(q.shape[2], np.nan, dtype=np.float64)
            gradient = np.full(q.shape, np.nan, dtype=np.float64, order="F")
            statuses = np.full(q.shape[2], -1, dtype=np.int32)
            used = ct.c_int32(0)
            status = self._lib.drpmd_pes_eval(
                q.shape[1], q.shape[2], q.ctypes.data_as(_DBL), energy.ctypes.data_as(_DBL),
                gradient.ctypes.data_as(_DBL), self.threads, statuses.ctypes.data_as(_INT), ct.byref(used))
            self.last_threads_used = used.value
            bad = np.flatnonzero(statuses)
            if status or bad.size:
                details = [(int(i), int(statuses[i])) for i in bad[:8]]
                raise PotentialError(f"Native PES status={status}; failed bead statuses (zero-based)={details}")
            if not 1 <= used.value <= self.threads:
                raise PotentialError("Native PES reported an invalid OpenMP team size")
            if not np.isfinite(energy).all() or not np.isfinite(gradient).all():
                raise PotentialError("Native PES returned nonfinite or unwritten energies/gradients")
            return energy, gradient

    def close(self):
        with self._lock:
            self.closed = True


# Importable scalar reference for measuring process/shared-memory transport
# around exactly the same compiled kernel as the direct OpenMP path.
_REFERENCE_PES = None


def initialize_potential(library, model=None):
    global _REFERENCE_PES
    if _REFERENCE_PES is not None:
        _REFERENCE_PES.close()
    _REFERENCE_PES = NativePotential(library, model=model, threads=1)


def get_potential(q):
    if _REFERENCE_PES is None:
        raise PotentialError("Native reference PES has not been initialized")
    return _REFERENCE_PES.evaluate(q)


def build_hhoo(output=None, compiler="gfortran"):
    """Build the original HHOO kernel plus the thin OpenMP/C ABI adapter."""
    import os
    import shlex
    import subprocess
    import tempfile
    root = Path(__file__).resolve().parents[1]
    sources = root / "examples/H-H-O-O/pes"
    output = sources / "libhhoo_drpmd.so" if output is None else Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(compiler)
    if not command:
        raise ValueError("compiler must not be empty")
    # Build elsewhere and atomically publish: no half-built DSO or .mod files in
    # the source tree, and failed builds do not corrupt an existing library.
    with tempfile.TemporaryDirectory(prefix="drpmd-hhoo-", dir=output.parent) as directory:
        temporary = Path(directory) / "libhhoo_drpmd.so"
        subprocess.run(command + ["-O3", "-fPIC", "-frecursive", "-fopenmp", "-shared",
                       str(sources / "hhoo_1014_analytic.f90"),
                       str(sources / "hhoo_drpmd_omp.f90"), "-o", str(temporary)],
                       cwd=directory, check=True)
        os.replace(temporary, output)
    return output


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Build the HHOO C ABI/OpenMP PES (requires gfortran)")
    parser.add_argument("--build-hhoo", action="store_true", required=True)
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(build_hhoo(args.output, args.compiler))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
