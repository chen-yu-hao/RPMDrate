"""Cayley free-ring propagator with optional reentrant C/OpenMP acceleration.

Coordinates and momenta have the RPMDrate layout ``(3, Natoms, Nbeads)`` and
atomic units.  The ring frequency is ``Nbeads / beta`` and physical-potential
kicks subtract the potential *gradient*, not the force.

No compilation occurs on import.  Build the optional native kernel explicitly::

    python -m rpmdrate.drpmd_integrator --build

The NumPy implementation is a serial fallback.  Multiple threads require the
OpenMP native kernel; selecting them never silently falls back to serial work.
"""

import argparse
import ctypes
import functools
import math
import numbers
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import threading

import numpy as np


_NATIVE_PATH = Path(__file__).with_name("_drpmd_native.so")
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


def build_native(compiler=None):
    """Build the C/OpenMP kernel explicitly, returning its absolute path.

    ``compiler`` or the ``CC`` environment variable can select a compiler.
    Compilation failure leaves any previously built kernel untouched.
    """
    command = shlex.split(compiler or os.environ.get("CC", "cc"))
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError("A C compiler with OpenMP support is required (set CC).")
    source = Path(__file__).with_name("_drpmd_native.c")
    with tempfile.TemporaryDirectory(prefix="drpmd-build-", dir=str(source.parent)) as directory:
        target = Path(directory) / _NATIVE_PATH.name
        args = command + ["-std=c99", "-O3", "-fPIC", "-shared", "-fopenmp",
                          str(source), str(source.with_name("_drpmd_engine.c")), "-lm", "-o", str(target)]
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError("C/OpenMP build failed:\n" + result.stdout + result.stderr)
        os.replace(target, _NATIVE_PATH)
    _load_native.cache_clear()
    return _NATIVE_PATH.resolve()


@functools.lru_cache(maxsize=1)
def _load_native():
    """Load a prebuilt library; shared library handles contain no ring state."""
    if not _NATIVE_PATH.is_file():
        return None
    try:
        library = ctypes.CDLL(str(_NATIVE_PATH))
        library.drpmd_abi_version.restype = ctypes.c_int
        if library.drpmd_abi_version() != 1:
            raise RuntimeError("Unsupported dRPMD native ABI; rebuild the native kernel.")
        library.drpmd_has_openmp.restype = ctypes.c_int
        library.drpmd_kick.argtypes = [
            _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_size_t, ctypes.c_double, ctypes.c_int]
        library.drpmd_kick.restype = None
        library.drpmd_free_step.argtypes = [
            _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_int, ctypes.c_int,
            _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_double, ctypes.c_double,
            _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR,
            _DOUBLE_PTR, ctypes.c_int]
        library.drpmd_free_step.restype = None
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Cannot load dRPMD native kernel; rebuild it with --build.") from exc
    return library


def _pointer(array):
    return array.ctypes.data_as(_DOUBLE_PTR)


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or value < 1 or value > np.iinfo(np.int32).max):
        raise ValueError(name + " must be a positive 32-bit integer.")
    return int(value)


class CayleyIntegrator:
    """An independent cached Cayley map for one ring-polymer configuration.

    ``masses`` are electron masses, ``beta`` is inverse hartree and ``dt`` is
    atomic time.  The finite time step may be negative or zero for diagnostics.
    Parameters are immutable; create a new object to change them.  ``native``
    is ``None`` (use a prebuilt kernel if present), ``True`` (require it), or
    ``False`` (serial NumPy).  Separate instances may run concurrently without
    sharing numerical state.  Calls on one instance are serialized because
    they reuse that instance's workspace.
    """

    def __init__(self, masses, beta, nbeads, dt, threads=1, native=None):
        self._masses = np.array(masses, dtype=np.float64, copy=True)
        if (self._masses.ndim != 1 or not self._masses.size
                or not np.all(np.isfinite(self._masses)) or np.any(self._masses <= 0)):
            raise ValueError("masses must be a nonempty vector of finite positive values.")
        self._masses.flags.writeable = False
        self._beta = float(beta)
        self._dt = float(dt)
        if not math.isfinite(self._beta) or self._beta <= 0:
            raise ValueError("beta must be finite and positive.")
        if not math.isfinite(self._dt):
            raise ValueError("dt must be finite.")
        self._nbeads = _positive_integer(nbeads, "nbeads")
        self._threads = _positive_integer(threads, "threads")
        self._nd = 3 * len(self._masses)
        if self._nd > np.iinfo(np.int32).max:
            raise ValueError("Too many Cartesian degrees of freedom for the native ABI.")
        if native is not None and native is not True and native is not False:
            raise ValueError("native must be None, True, or False.")
        self._library = None if native is False else _load_native()
        if native is True and self._library is None:
            raise RuntimeError("Native kernel is missing; run python -m rpmdrate.drpmd_integrator --build.")
        if self._threads > 1 and (self._library is None or not self._library.drpmd_has_openmp()):
            raise RuntimeError("threads > 1 requires the OpenMP native kernel; run "
                               "python -m rpmdrate.drpmd_integrator --build.")
        self._shape = (3, len(self._masses), self._nbeads)
        self._lock = threading.Lock()
        self._prepare()

    @property
    def masses(self):
        # A copy prevents clients from re-enabling writes on cached mass data.
        return self._masses.copy()

    @property
    def beta(self):
        return self._beta

    @property
    def nbeads(self):
        return self._nbeads

    @property
    def dt(self):
        return self._dt

    @property
    def threads(self):
        return self._threads

    @property
    def native(self):
        return self._library is not None

    def _prepare(self):
        beads = self._nbeads
        omega = beads / self._beta
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self._alpha = float(np.float64(0.25) * self._dt * self._dt * omega * omega)
            self._half = np.repeat(0.5 * self._dt / self._masses, 3)
            self._spring = np.repeat(self._dt * self._masses * omega * omega, 3)
        if (not math.isfinite(self._alpha) or not np.all(np.isfinite(self._half))
                or not np.all(np.isfinite(self._spring)) or not math.isfinite(4 * self._alpha)):
            raise ValueError("Cayley coefficients overflow; check beta, dt, masses and nbeads.")
        self._pair_scale = 1.0 / (1.0 + 4.0 * self._alpha)
        self._inverse_diagonal = np.ones(beads)
        self._lower = np.zeros(beads)
        self._z1 = np.zeros(beads)
        self._z2 = np.zeros(beads)
        self._sinverse = np.array([1.0, 0.0, 0.0, 1.0])
        self._work = np.empty((self._nd, beads), dtype=np.float64)
        if beads < 3:
            return
        alpha = self._alpha
        diagonal = 1.0 + 2.0 * alpha
        self._inverse_diagonal[0] = 1.0 / diagonal
        for k in range(1, beads):
            self._lower[k] = -alpha * self._inverse_diagonal[k - 1]
            self._inverse_diagonal[k] = 1.0 / (diagonal + self._lower[k] * alpha)

        def solve(rhs):
            y = rhs.copy()
            for k in range(1, beads):
                y[k] -= self._lower[k] * y[k - 1]
            y[-1] *= self._inverse_diagonal[-1]
            for k in range(beads - 2, -1, -1):
                y[k] = (y[k] + alpha * y[k + 1]) * self._inverse_diagonal[k]
            return y

        rhs = np.zeros(beads)
        rhs[0] = -alpha
        self._z1 = solve(rhs)
        rhs[0], rhs[-1] = 0.0, -alpha
        self._z2 = solve(rhs)
        s11, s12 = 1.0 + self._z1[-1], self._z2[-1]
        s21, s22 = self._z1[0], 1.0 + self._z2[0]
        determinant = s11 * s22 - s12 * s21
        if determinant <= 0 or not math.isfinite(determinant):
            raise ValueError("Cyclic Cayley factorization lost numerical precision; reduce dt.")
        self._sinverse[:] = np.array([s22, -s12, -s21, s11]) / determinant

    def _array(self, array, name, writable=True):
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float64):
            raise TypeError(name + " must be a float64 NumPy array.")
        if array.shape != self._shape:
            raise ValueError("{} must have shape {}; got {}.".format(name, self._shape, array.shape))
        if not array.flags.f_contiguous:
            raise ValueError(name + " must be Fortran-contiguous; use numpy.asfortranarray.")
        if writable and not array.flags.writeable:
            raise ValueError(name + " must be writable.")
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " contains NaN or infinity.")

    def kick(self, p, gradient, interval):
        """Update ``p -= interval * gradient`` in place (gradient in hartree/bohr)."""
        self._array(p, "p")
        self._array(gradient, "gradient", writable=False)
        if np.shares_memory(p, gradient):
            raise ValueError("p and gradient must not share memory.")
        interval = float(interval)
        if not math.isfinite(interval):
            raise ValueError("kick interval must be finite.")
        with self._lock:
            if self._library is None:
                p -= interval * gradient
            else:
                self._library.drpmd_kick(_pointer(p), _pointer(gradient), p.size,
                                         interval, self._threads)

    def free_step(self, p, q):
        """Apply one full Cayley free-ring step to independent ``p`` and ``q`` arrays."""
        self._array(p, "p")
        self._array(q, "q")
        if np.shares_memory(p, q):
            raise ValueError("p and q must not share memory.")
        with self._lock:
            if self._library is not None:
                self._library.drpmd_free_step(
                    _pointer(p), _pointer(q), self._nd, self._nbeads,
                    _pointer(self._half), _pointer(self._spring), self._alpha, self._pair_scale,
                    _pointer(self._inverse_diagonal), _pointer(self._lower),
                    _pointer(self._z1), _pointer(self._z2), _pointer(self._sinverse),
                    _pointer(self._work), self._threads)
            else:
                self._numpy_step(p.reshape((self._nd, self._nbeads), order="F"),
                                 q.reshape((self._nd, self._nbeads), order="F"))

    def _numpy_step(self, p, q):
        beads = self._nbeads
        if beads == 1:
            q[:, 0] += 2.0 * self._half * p[:, 0]
            return
        mid = self._work
        mid[:] = q + self._half[:, None] * p
        if beads == 2:
            b1, b2 = mid[:, 0].copy(), mid[:, 1].copy()
            mid[:, 0] = ((1.0 + 2.0 * self._alpha) * b1 + 2.0 * self._alpha * b2) * self._pair_scale
            mid[:, 1] = (2.0 * self._alpha * b1 + (1.0 + 2.0 * self._alpha) * b2) * self._pair_scale
        else:
            for k in range(1, beads):
                mid[:, k] -= self._lower[k] * mid[:, k - 1]
            mid[:, -1] *= self._inverse_diagonal[-1]
            for k in range(beads - 2, -1, -1):
                mid[:, k] = (mid[:, k] + self._alpha * mid[:, k + 1]) * self._inverse_diagonal[k]
            c1 = self._sinverse[0] * mid[:, -1] + self._sinverse[1] * mid[:, 0]
            c2 = self._sinverse[2] * mid[:, -1] + self._sinverse[3] * mid[:, 0]
            mid -= c1[:, None] * self._z1 + c2[:, None] * self._z2
        p -= self._spring[:, None] * (2.0 * mid - np.roll(mid, 1, axis=1) - np.roll(mid, -1, axis=1))
        q[:] = 2.0 * mid - q


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="compile the optional C/OpenMP kernel")
    parser.add_argument("--compiler", help="C compiler command (default: CC or cc)")
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("specify --build to compile the native kernel")
    try:
        print(build_native(args.compiler))
    except RuntimeError as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
