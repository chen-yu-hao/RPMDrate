"""Importable test PES for fixed shared-memory transport; never production data."""
import os
from pathlib import Path
import time
import numpy as np

_K = 1.0

def initialize_potential(spring=1.0, log=None, fail=False):
    global _K
    if fail:
        raise ValueError("intentional initializer failure")
    _K = float(spring)
    if log:
        with Path(log).open("a") as stream:
            stream.write(f"{os.getpid()}\n")


def get_potential(q, action=None):
    if action == "exit":
        os._exit(17)
    if action == "sleep":
        time.sleep(2)
    if action == "raise":
        raise ValueError("intentional PES failure")
    energy = 0.5*_K*np.sum(q*q, axis=(0, 1))
    grad = _K*q.copy(order="F")
    if action == "mutate":
        q.fill(123456.)
    if action == "nan":
        grad[0, 0, 0] = np.nan
    return energy, grad
