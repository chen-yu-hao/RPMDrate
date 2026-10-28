"""Run after `python build_python.py`; coordinates/gradients use atomic units."""
from pathlib import Path

import numpy as np

import hhoo_pes


weights = Path(__file__).resolve().with_name("txt_1014")
result = hhoo_pes.hhoo_1014_init(str(weights))
if result != 0:
    raise RuntimeError(f"PES initialization failed: {result}")

x = np.asfortranarray([
    [-0.7, 0.7, 0.0, 0.0],
    [ 0.0, 0.0, 3.0, 3.0],
    [ 0.0, 0.0,-1.4, 1.4],
], dtype=np.float64)
energy, gradient = hhoo_pes.hhoo_1014_grad(x)
force = -gradient
print("Energy / hartree:", energy)
print("Gradient dV/dx / (hartree/bohr):\n", gradient)

# Demonstrate the RPMD-shaped interface with two diagnostic geometries.
q = np.asfortranarray(np.stack([x, x + np.array([[0.2], [-0.1], [0.3]])], axis=2))
V, dVdq, result = hhoo_pes.get_potential(q)
if result != 0:
    raise RuntimeError(f"PES evaluation failed: {result}")
np.testing.assert_allclose(V, energy, rtol=0, atol=1e-11)
np.testing.assert_allclose(dVdq, np.repeat(gradient[:, :, None], 2, axis=2), rtol=1e-10, atol=1e-10)
print("Batch shapes:", V.shape, dVdq.shape, "; result =", result)
