#!/usr/bin/env python
# Minimal 300 K, 32-bead H-H-O-O PMF chain test.  This is a short numerical
# precheck (three windows and two sampling steps per window), not a converged
# production PMF.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pes"))
from hhoo_pes import get_potential, hhoo_1014_init


def initialize_potential():
    status = hhoo_1014_init(os.path.join(os.path.dirname(__file__), "pes", "txt_1014"))
    if status != 0:
        raise RuntimeError("H-H-O-O PES initialization failed: {0}".format(status))

label = "H-H-O-O Cayley PMF trial"
random_seed = 20261029

reactants(
    atoms=["H", "H", "O", "O"],
    reactant1Atoms=[1],
    reactant2Atoms=[2, 3, 4],
    Rinf=(10.0, "bohr"),
)

transitionState(
    geometry=(
        [
            [-0.7, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.0, 3.0, -1.4],
            [0.0, 3.0, 1.4],
        ],
        "bohr",
    ),
    formingBonds=[(1, 3)],
    breakingBonds=[(1, 2)],
)

thermostat("Andersen")

xi_list = [0.99, 1.0, 1.01]
kforce = 0.01 * T
dt = (0.001, "ps")

generateUmbrellaConfigurations(
    dt=dt,
    evolutionTime=(0.0, "ps"),
    xi_list=xi_list,
    kforce=kforce,
)

windows = [
    Window(
        xi=xi,
        kforce=kforce,
        trajectories=1,
        equilibrationTime=dt,
        evolutionTime=(0.002, "ps"),
    )
    for xi in xi_list
]

conductUmbrellaSampling(dt=dt, windows=windows, saveTrajectories=False)
computePotentialOfMeanForce(
    windows=windows,
    xi_min=xi_list[0],
    xi_max=xi_list[-1],
    bins=4,
)
