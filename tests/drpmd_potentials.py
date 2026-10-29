"""Small importable PES functions used by spawn-process backend regressions."""

import os
import time
import numpy as np


_initialized = 0
_spring = None


def initialize_potential(spring=1.25):
    global _initialized, _spring
    _initialized += 1
    _spring = float(spring)


def get_potential(q, shift=0.0):
    if _spring is None:
        raise RuntimeError("PES initializer was not called")
    return 0.5 * _spring * np.sum(q * q, axis=(0, 1)) + shift, _spring * q, 0


def get_with_dimensions(q, natoms, nbeads):
    if q.shape != (3, natoms, nbeads):
        raise RuntimeError("Wrong explicit dimensions")
    return get_potential(q)


def initialization_count(q):
    return np.full(q.shape[2], _initialized, dtype=float), np.zeros_like(q), 0


def failing_potential(q):
    return np.zeros(q.shape[2]), np.zeros_like(q), -7


def throwing_potential(q):
    raise RuntimeError("intentional PES failure")


def slow_identity(q):
    # Enough work duration for the scheduler to exercise every persistent
    # worker; the delay belongs only to this test fixture.
    time.sleep(0.03)
    return np.full(q.shape[2], os.getpid(), dtype=float), np.zeros_like(q), 0


def failing_initializer():
    raise RuntimeError("intentional initialization failure")


def status_initializer(status=None):
    """Status-only initializer; independent of the initialization-count fixture."""
    return status


def zero_potential(q):
    return np.zeros(q.shape[2]), np.zeros_like(q), 0
