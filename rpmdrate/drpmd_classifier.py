"""One-time translation of the built-in product classifier to a C workspace.

Only chemical symbols and channel formulas are interpreted in Python. All
per-check centroid/graph/fragment/motion operations and the persistence counter
run in C, inside the integration loop. No Python callback is passed to C.
"""
from __future__ import annotations

import ctypes as ct
import threading
import weakref

import numpy as np

from .drpmd_backends import PotentialError
from .drpmd_integrator import _positive_integer, _pointer

_I = ct.POINTER(ct.c_int32)
_D = ct.POINTER(ct.c_double)


def classifier_library(lib):
    if not hasattr(lib, "drpmd_classifier_abi_version"):
        raise RuntimeError("Native product checks require rebuilding: python -m rpmdrate.drpmd_integrator --build")
    lib.drpmd_classifier_abi_version.restype = ct.c_int32
    if lib.drpmd_classifier_abi_version() != 1:
        raise RuntimeError("Unsupported native classifier ABI; rebuild drpmd_integrator")
    lib.drpmd_classifier_create.argtypes = [ct.c_int32]*4 + [_I, _D, _D, ct.c_double, _I, _I, _I]
    lib.drpmd_classifier_create.restype = ct.c_void_p
    lib.drpmd_classifier_destroy.argtypes = [ct.c_void_p]
    lib.drpmd_classifier_destroy.restype = None
    lib.drpmd_classifier_evaluate.argtypes = [ct.c_void_p, _D, _D, _I]
    lib.drpmd_classifier_evaluate.restype = ct.c_int32
    lib.drpmd_classifier_observe.argtypes = [ct.c_void_p, _D, _D, ct.c_int32, _I]
    lib.drpmd_classifier_observe.restype = ct.c_int32
    lib.drpmd_classifier_checks.argtypes = [ct.c_void_p]
    lib.drpmd_classifier_checks.restype = ct.c_int64
    lib.drpmd_engine_advance_classified.argtypes = [
        ct.c_void_p, ct.c_void_p, _D, _D, _D, _D, ct.c_int32, ct.c_double,
        _D, _D, _I, _I, ct.c_void_p, ct.c_int64, ct.c_int32, ct.c_int32, _I]
    lib.drpmd_engine_advance_classified.restype = ct.c_int32
    return lib


class NativeClassifier:
    """Snapshot of one built-in ConnectivityClassifier, with a per-collision streak.

    Instances are NOT reused between collisions. Subclasses/custom callables
    remain Python callbacks unless explicitly rejected by classifier_backend.
    The configuration is copied into C; changing it mid-trajectory is unsupported.
    """
    def __init__(self, classifier, nbeads, lib=None):
        from .drpmd_core import ConnectivityClassifier, _formula_counts
        if type(classifier) is not ConnectivityClassifier:
            raise TypeError("Native checks require the exact built-in ConnectivityClassifier")
        if lib is None:
            from .drpmd_engine import engine_library
            lib = engine_library()
        if lib is None:
            raise RuntimeError("Build the native kernel before creating NativeClassifier")
        self.lib = classifier_library(lib)
        nbeads = _positive_integer(nbeads, "nbeads")
        self.shape = (3, len(classifier.atoms), nbeads)
        self.lock = threading.RLock()
        elements = sorted(set(classifier.atoms))
        types = np.array([elements.index(a) for a in classifier.atoms], dtype=np.int32)
        signatures = [classifier.reactant_signature, *classifier.channels.keys()]
        self.labels = ["reactants", *classifier.channels.values()]
        formulas = np.zeros((len(signatures), self.shape[1], len(elements)), dtype=np.int32)
        sizes = np.array([len(parts) for parts in signatures], dtype=np.int32)
        for k, fragments in enumerate(signatures):
            for f, formula in enumerate(fragments):
                counts = _formula_counts(formula)
                formulas[k, f] = [counts[element] for element in elements]
        cutoffs = np.zeros((self.shape[1], self.shape[1]), dtype=np.float64)
        for i, atom in enumerate(classifier.atoms):
            for j, other in enumerate(classifier.atoms):
                if i != j:
                    cutoffs[i, j] = classifier.cutoffs[tuple(sorted((atom, other)))]
        masses = np.require(classifier.masses, dtype=np.float64, requirements=["C", "A"])
        status = ct.c_int32()
        self.context = self.lib.drpmd_classifier_create(
            self.shape[1], nbeads, len(elements), len(signatures), types.ctypes.data_as(_I),
            _pointer(masses), _pointer(cutoffs), classifier.exit_distance,
            sizes.ctypes.data_as(_I), formulas.ctypes.data_as(_I), ct.byref(status))
        if not self.context:
            raise ValueError(f"Native classifier initialization failed (status={status.value})")
        self._finalizer = weakref.finalize(self, self.lib.drpmd_classifier_destroy, self.context)

    def _check_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("Native classifier is closed")

    def label(self, code):
        if code == -1:
            return None
        if not 0 <= code < len(self.labels):
            raise PotentialError("Invalid channel code from native classifier")
        return self.labels[code]

    @property
    def checks(self):
        with self.lock:
            self._check_open()
            return int(self.lib.drpmd_classifier_checks(self.context))

    def evaluate(self, q, p, *, persistence=None):
        """Standalone reference/testing call; production calls from the C loop."""
        with self.lock:
            self._check_open()
            arrays = []
            for data in (q, p):
                array = np.asarray(data)
                if array.shape != self.shape or np.iscomplexobj(array):
                    raise ValueError(f"Classifier arrays must be real with shape {self.shape}")
                arrays.append(np.require(array, dtype=np.float64, requirements=["F", "A"]))
            label = ct.c_int32(-1)
            if persistence is None:
                code = self.lib.drpmd_classifier_evaluate(self.context, *map(_pointer, arrays), ct.byref(label))
            else:
                persistence = _positive_integer(persistence, "persistence")
                code = self.lib.drpmd_classifier_observe(self.context, *map(_pointer, arrays), persistence, ct.byref(label))
            if code:
                raise PotentialError(f"Native product classification failed (status={code})")
            return self.label(label.value)

    def close(self):
        with self.lock:
            self._finalizer()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()


def select_classifier(dynamics, classifier):
    """Compile the default classifier; never silently replace custom semantics."""
    from .drpmd_core import ConnectivityClassifier
    setting = dynamics.classifier_backend
    if classifier is None:
        dynamics.last_classifier_backend = "none"
        return None
    supported = type(classifier) is ConnectivityClassifier
    if setting == "native" and not supported:
        raise ValueError("Native product checks require the built-in ConnectivityClassifier; custom callbacks need auto/python")
    if setting == "python" or not supported:
        dynamics.last_classifier_backend = "python"
        return None
    if (len(classifier.masses) != len(dynamics.masses) or
            not np.array_equal(classifier.masses, dynamics.masses)):
        raise ValueError("Native classifier masses must match the trajectory dynamics")
    result = NativeClassifier(classifier, dynamics.nbeads, dynamics.engine.lib)
    dynamics.last_classifier_backend = "native"
    return result
