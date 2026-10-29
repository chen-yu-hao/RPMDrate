"""SYNTHETIC scheduling workload, not a chemical potential or speed claim.

A zero-energy two-atom PES plus a prescribed sleep ONCE per collision. Both
schedulers receive identical collision costs by global trajectory ID. The sleep
models uneven trajectory runtimes, not floating point work or chemical rates.
Hooks are installed only when this explicit benchmark PES is selected.
"""
import multiprocessing as mp
import time
import numpy as np


def initialize_potential(costs, chains):
    from rpmdrate import drpmd_core as core
    from rpmdrate.drpmd_pool import _Runtime
    from rpmdrate.drpmd_parallel import make_plan
    name = mp.current_process().name
    original = core.propagate
    current = [None]
    counter = [0]
    if name.startswith("drpmd-parent-"):
        cid = int(name.rsplit("-", 1)[1])
        first = make_plan(len(costs), trajectory_workers=chains)["chains"][cid-1].first_trajectory
    else:
        first = None
        collision = _Runtime.collision
        def marked(self, task):
            current[0] = task["trajectory"]
            try:
                return collision(self, task)
            finally:
                current[0] = None
        _Runtime.collision = marked

    def timed(*args, **kwargs):
        if first is not None:
            index = first + counter[0]
            counter[0] += 1
        else:
            index = current[0]
        if index is None:
            raise RuntimeError("Synthetic benchmark lost collision identity")
        time.sleep(costs[index-1])
        return original(*args, **kwargs)
    core.propagate = timed


def get_potential(q):
    return np.zeros(q.shape[2]), np.zeros_like(q, order="F")
