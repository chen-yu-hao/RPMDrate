"""NumPy and optional PyTorch implementations of the repository's HHOO PES.

These implementations use the *unchanged* ``txt_1014`` five-network ensemble
from ``examples/H-H-O-O/pes``.  They reproduce the descriptors, activation,
scaling and averaging in ``hhoo_1014_analytic.f90``; no fitting is performed.

Atom order is H, H, O, O.  Coordinates are bohr, energies are hartree and
gradients are positive dV/dq in hartree/bohr.  The NumPy callable has RPMDrate's
``(3, 4, beads)`` layout.  The Torch energy module uses ``(beads, 4, 3)`` and
lets :class:`rpmdrate.drpmd_backends.TorchPotential` differentiate its energies.
Torch is imported only when :func:`make_torch_model` is called.

This is an implementation of the supplied model, not evidence of its physical
accuracy, electronic spin assignment or validity outside its training domain.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading

import numpy as np


_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_HEADER = (2, 7, 50, 50, 3001)


def default_weights_path():
    """Return the example model path in a source checkout.

    Installed copies without the examples must pass ``weights_path`` explicitly.
    The file is never downloaded or substituted with another model.
    """
    return Path(__file__).resolve().parent.parent / "examples/H-H-O-O/pes/txt_1014"


class _WeightReader:
    """Read the fixed Fortran list-directed records used by ``txt_1014``."""

    def __init__(self, contents):
        self.lines = contents.splitlines()
        self.line = 0

    def record(self):
        if self.line >= len(self.lines):
            raise ValueError("Truncated HHOO weight file")
        record = self.lines[self.line]
        self.line += 1
        return record

    def values(self, count):
        values = []
        while len(values) < count:
            record = self.record()
            try:
                values.extend(float(word.replace("D", "E").replace("d", "e"))
                              for word in record.split())
            except ValueError as exc:
                raise ValueError("Invalid HHOO numeric record at line %d" % self.line) from exc
        if len(values) != count:
            raise ValueError("Unexpected HHOO record width at line %d" % self.line)
        return np.asarray(values, dtype=np.float64)


def _read_weights(weights_path):
    path = Path(weights_path if weights_path is not None else default_weights_path()).expanduser().resolve()
    contents = path.read_bytes()
    try:
        reader = _WeightReader(contents.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise ValueError("HHOO weights must be an ASCII text file") from exc
    arrays = {
        "w1": np.empty((5, 7, 50)), "b1": np.empty((5, 50)),
        "w2": np.empty((5, 50, 50)), "b2": np.empty((5, 50)),
        "w3": np.empty((5, 50)), "b3": np.empty(5),
        "input_lower": np.empty((5, 7)), "input_upper": np.empty((5, 7)),
        "output_lower": np.empty(5), "output_upper": np.empty(5),
    }
    for network in range(5):
        header = tuple(reader.values(1)[0] for _ in range(5))
        if header != _HEADER:
            raise ValueError("HHOO network %d has invalid architecture/header" % (network + 1))
        # Fortran scile(2,7) is stored as seven consecutive lower/upper pairs.
        bounds = reader.values(16).reshape(8, 2)
        arrays["input_lower"][network] = bounds[:7, 0]
        arrays["input_upper"][network] = bounds[:7, 1]
        arrays["output_lower"][network], arrays["output_upper"][network] = bounds[7]
        # These are connection-index records, skipped by the reference reader.
        for _ in range(101):
            reader.record()
        for neuron in range(50):
            values = reader.values(8)
            arrays["w1"][network, :, neuron] = values[:7]
            arrays["b1"][network, neuron] = values[7]
        for neuron in range(50):
            values = reader.values(51)
            arrays["w2"][network, :, neuron] = values[:50]
            arrays["b2"][network, neuron] = values[50]
        values = reader.values(51)
        arrays["w3"][network] = values[:50]
        arrays["b3"][network] = values[50]
    if any(line.strip() for line in reader.lines[reader.line:]):
        raise ValueError("Unexpected trailing data after the five HHOO networks")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("HHOO weights and scales must all be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        width = arrays["input_upper"] - arrays["input_lower"]
        output_width = arrays["output_upper"] - arrays["output_lower"]
    if not np.isfinite(width).all() or np.any(width == 0) or not np.isfinite(output_width).all():
        raise ValueError("Invalid HHOO scaling widths")
    for value in arrays.values():
        value.flags.writeable = False
    return path, hashlib.sha256(contents).hexdigest(), arrays


def _numpy_descriptors(rho):
    cross = rho[:, 1:5]
    return np.column_stack((
        cross.sum(axis=1),
        np.sqrt((cross * cross).sum(axis=1)),
        np.sqrt(cross[:, 0] * cross[:, 1] + cross[:, 2] * cross[:, 3]),
        np.sqrt(cross[:, 0] * cross[:, 2] + cross[:, 1] * cross[:, 3]),
        (cross ** 3).sum(axis=1) ** (1.0 / 3.0),
        rho[:, 0], rho[:, 5],
    ))


class HHOOModel:
    """Read-only, batched NumPy HHOO potential with analytic gradients.

    ``model(q)`` returns ``(energies, positive_gradients, 0)``.  Invalid inputs,
    unreadable weights and nonfinite evaluations raise errors; partial batches
    are never reported as successful.  Each process may create its own model.
    """

    def __init__(self, weights_path=None):
        path, digest, arrays = _read_weights(weights_path)
        self.weights_path = str(path)
        self.weights_sha256 = digest
        self._weights = arrays

    def __call__(self, q):
        if np.iscomplexobj(q):
            raise ValueError("HHOO coordinates must be real")
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 3 or q.shape[0:2] != (3, 4) or q.shape[2] < 1:
            raise ValueError("HHOO coordinates must have shape (3, 4, beads), beads >= 1")
        if not np.isfinite(q).all():
            raise ValueError("HHOO coordinates must be finite")
        xyz = q.transpose(2, 1, 0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            delta = np.stack([xyz[:, i] - xyz[:, j] for i, j in _PAIRS], axis=1)
            distance2 = (delta * delta).sum(axis=2)
        if not np.isfinite(distance2).all() or np.any(distance2 <= 0):
            raise ValueError("HHOO pair distances must be finite and positive")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            rho = 1.0 / np.sqrt(distance2)
            descriptors = _numpy_descriptors(rho)
            w = self._weights
            width = w["input_upper"] - w["input_lower"]
            a0 = (descriptors[None, :, :] - w["input_lower"][:, None, :]) / width[:, None, :] * 2.0 - 1.0
            z1 = np.matmul(a0, w["w1"]) + w["b1"][:, None, :]
            den1 = np.sqrt(1.0 + z1 * z1)
            a1 = z1 / den1
            z2 = np.matmul(a1, w["w2"]) + w["b2"][:, None, :]
            den2 = np.sqrt(1.0 + z2 * z2)
            a2 = z2 / den2
            a3 = (a2 * w["w3"][:, None, :]).sum(axis=2) + w["b3"][:, None]
            output_width = w["output_upper"] - w["output_lower"]
            energies = ((a3 * output_width[:, None] + w["output_lower"][:, None]
                         + w["output_upper"][:, None]) / 10.0).sum(axis=0)
            adj2 = w["w3"][:, None, :] * (output_width / 10.0)[:, None, None] / den2 ** 3
            adj1 = np.matmul(adj2, w["w2"].transpose(0, 2, 1)) / den1 ** 3
            adj0 = np.matmul(adj1, w["w1"].transpose(0, 2, 1))
            dvdp = (adj0 * 2.0 / width[:, None, :]).sum(axis=0)

            dvdrho = np.empty_like(rho)
            dvdrho[:, 0], dvdrho[:, 5] = dvdp[:, 5], dvdp[:, 6]
            dvdrho[:, 1:5] = (
                dvdp[:, 0, None] + dvdp[:, 1, None] * rho[:, 1:5] / descriptors[:, 1, None]
                + dvdp[:, 4, None] * rho[:, 1:5] ** 2 / descriptors[:, 4, None] ** 2
            )
            cross = rho[:, 1:5]
            dvdrho[:, 1:5] += (
                dvdp[:, 2, None] * cross[:, [1, 0, 3, 2]] / (2.0 * descriptors[:, 2, None])
                + dvdp[:, 3, None] * cross[:, [2, 3, 0, 1]] / (2.0 * descriptors[:, 3, None])
            )
            pair_gradient = -dvdrho[:, :, None] * rho[:, :, None] ** 3 * delta
            gradients = np.zeros_like(xyz)
            for pair, (i, j) in enumerate(_PAIRS):
                gradients[:, i] += pair_gradient[:, pair]
                gradients[:, j] -= pair_gradient[:, pair]
        if not np.isfinite(energies).all() or not np.isfinite(gradients).all():
            raise FloatingPointError("Nonfinite HHOO energy or gradient")
        return energies, np.asfortranarray(gradients.transpose(2, 1, 0)), 0

    get_potential = __call__


_potential = None
_initialization_lock = threading.Lock()


def initialize_potential(weights_path=None):
    """Initialize a process-local NumPy model for ``PotentialSpec``.

    Repeating initialization with the same resolved path and identical file
    contents is harmless.  Replacing an initialized global model is rejected;
    use a separate ``HHOOModel`` instead.  Checking the digest prevents repeated
    runs from recording new file provenance while evaluating cached old weights.
    """
    global _potential
    path = Path(weights_path if weights_path is not None else default_weights_path()).expanduser().resolve()
    with _initialization_lock:
        if _potential is not None:
            if str(path) != _potential.weights_path:
                raise ValueError("HHOO global model is already initialized with different weights")
            if hashlib.sha256(path.read_bytes()).hexdigest() != _potential.weights_sha256:
                raise ValueError("HHOO weights changed at the initialized path; start a new process or use a separate HHOOModel")
            return
        _potential = HHOOModel(path)


def get_potential(q):
    """Evaluate the initialized model, using the example weights by default."""
    if _potential is None:
        initialize_potential()
    return _potential(q)


def make_torch_model(weights_path=None):
    """Build an optional FP64 ``torch.nn.Module`` from the exact HHOO weights.

    ``forward(xyz)`` accepts ``(beads, 4, 3)`` and returns ``(beads,)`` energies
    in hartree.  All five networks execute in batched tensor operations on the
    selected device.  The caller may move the returned module with ``.to()``;
    :class:`TorchPotential` handles device placement and Cartesian autograd.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("The HHOO Torch model requires optional PyTorch; install a build matching your device") from exc
    path, digest, arrays = _read_weights(weights_path)

    class HHOOEnergy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights_path = str(path)
            self.weights_sha256 = digest
            for name, value in arrays.items():
                self.register_buffer(name, torch.from_numpy(np.array(value, copy=True)))

        def forward(self, xyz):
            if xyz.ndim != 3 or xyz.shape[1:] != (4, 3) or xyz.shape[0] < 1:
                raise ValueError("HHOO Torch coordinates must have shape (beads, 4, 3), beads >= 1")
            if xyz.dtype != torch.float64:
                raise ValueError("HHOO Torch coordinates must be float64")
            delta = torch.stack([xyz[:, i] - xyz[:, j] for i, j in _PAIRS], dim=1)
            distance2 = (delta * delta).sum(dim=2)
            if not bool(torch.isfinite(distance2).all()) or bool((distance2 <= 0).any()):
                raise ValueError("HHOO pair distances must be finite and positive")
            rho = torch.rsqrt(distance2)
            cross = rho[:, 1:5]
            descriptors = torch.stack((
                cross.sum(dim=1),
                torch.sqrt((cross * cross).sum(dim=1)),
                torch.sqrt(cross[:, 0] * cross[:, 1] + cross[:, 2] * cross[:, 3]),
                torch.sqrt(cross[:, 0] * cross[:, 2] + cross[:, 1] * cross[:, 3]),
                (cross ** 3).sum(dim=1) ** (1.0 / 3.0),
                rho[:, 0], rho[:, 5],
            ), dim=1)
            a0 = ((descriptors[None, :, :] - self.input_lower[:, None, :])
                  / (self.input_upper - self.input_lower)[:, None, :] * 2.0 - 1.0)
            z1 = torch.matmul(a0, self.w1) + self.b1[:, None, :]
            a1 = z1 / torch.sqrt(1.0 + z1 * z1)
            z2 = torch.matmul(a1, self.w2) + self.b2[:, None, :]
            a2 = z2 / torch.sqrt(1.0 + z2 * z2)
            a3 = (a2 * self.w3[:, None, :]).sum(dim=2) + self.b3[:, None]
            return ((a3 * (self.output_upper - self.output_lower)[:, None]
                     + self.output_lower[:, None] + self.output_upper[:, None]) / 10.0).sum(dim=0)

    return HHOOEnergy()
