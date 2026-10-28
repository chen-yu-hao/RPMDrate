#!/usr/bin/env python3
"""Minimal end-to-end H-H-O-O PES/RPMD precheck.

This deliberately runs only two 32-bead umbrella steps.  It checks that the
analytic PES wrapper, the RPMDrate Fortran extension, the reaction-coordinate
surface, and the Cayley free-ring path can be loaded and called together.  It
is not a PMF calculation and makes no statistical or chemical claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pes"))
sys.path.insert(0, str(HERE.parents[1]))

import hhoo_pes
from rpmdrate import _main
from rpmdrate.main import RPMD
from rpmdrate.surface import Reactants, TransitionState
from rpmdrate.thermostat import AndersenThermostat


def potential(q, *args):
    """Adapt the f2py PES wrapper to RPMDrate's callback signature."""
    return hhoo_pes.get_potential(q)


def main():
    weights = HERE / "pes" / "txt_1014"
    status = int(hhoo_pes.hhoo_1014_init(str(weights)))
    if status != 0:
        raise RuntimeError(f"H-H-O-O PES initialization failed: {status}")

    geometry = numpy.asfortranarray(
        numpy.array(
            [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 3.0, -1.4], [0.0, 3.0, 1.4]],
            dtype=numpy.float64,
        ).T
    )
    q = numpy.asfortranarray(numpy.repeat(geometry[:, :, None], 32, axis=2))
    values, gradients, status = hhoo_pes.get_potential(q)
    if int(status) != 0 or not numpy.isfinite(values).all() or not numpy.isfinite(gradients).all():
        raise RuntimeError(f"H-H-O-O batch PES evaluation failed: {status}")

    reactants = Reactants(["H", "H", "O", "O"], [1], [2, 3, 4], (10.0, "bohr"))
    transition_state = TransitionState(
        (geometry.T, "bohr"), formingBonds=[(1, 3)], breakingBonds=[(1, 2)]
    )
    system = RPMD(
        label="H-H-O-O precheck",
        T=300.0,
        Nbeads=32,
        reactants=reactants,
        transitionState=transition_state,
        potential=potential,
        thermostat=AndersenThermostat(),
        outputDirectory=str(HERE),
        randomSeed=20261029,
    )
    system.dt = 0.05
    system.mode = 1
    system.activate()
    p = numpy.asfortranarray(system.sampleMomentum())

    equilibration_result = _main.system.equilibrate(
        0.0, p, q, 1, 1.0, system.potential, 0.1, 0, 0, "1.0", 1, 1
    )
    av, av2, actual_steps, sampling_result = _main.system.umbrella_trajectory(
        0.0, p, q, 2, 1.0, system.potential, 0.1, 0.0, 0, "1.0", 1, 1
    )
    if equilibration_result != 0 or sampling_result != 0 or actual_steps != 2:
        raise RuntimeError(
            f"RPMD precheck failed: equilibration={equilibration_result}, "
            f"sampling={sampling_result}, steps={actual_steps}"
        )
    if not numpy.isfinite(q).all() or not numpy.isfinite(p).all():
        raise RuntimeError("RPMD precheck produced non-finite coordinates or momenta")

    report = {
        "status": "passed",
        "natoms": 4,
        "nbeads": 32,
        "pes_status": int(status),
        "energy_min_hartree": float(values.min()),
        "energy_max_hartree": float(values.max()),
        "max_abs_gradient_hartree_per_bohr": float(numpy.abs(gradients).max()),
        "equilibration_result": int(equilibration_result),
        "sampling_result": int(sampling_result),
        "sampling_steps": int(actual_steps),
        "sampling_xi_sum": float(av),
        "sampling_xi2_sum": float(av2),
    }
    output = HERE / "precheck_result.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
