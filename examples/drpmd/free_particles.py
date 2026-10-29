"""Noninteracting two-particle smoke-test PES, not a chemical potential."""
import numpy as np


def get_potential(q):
    return np.zeros(q.shape[2]), np.zeros_like(q), 0
