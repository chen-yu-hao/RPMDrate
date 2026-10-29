"""Retain the original static scheduler regressions and CUDA/plan coverage.

The dynamic pool has its own tests in drpmd_pooltest. Legacy RNG replay and
fixed-partition failure expectations explicitly select scheduler='static'.
"""
from .drpmd_parallel_legacy import *
from .drpmd_parallel_legacy import SpawnedEnsembleTest as _LegacyEnsemble


class SpawnedEnsembleTest(_LegacyEnsemble):
    def setUp(self):
        super().setUp()
        public_run = self.run
        def legacy_run(config, **kwargs):
            if config.get("parallel") or any(k in kwargs for k in ("trajectory_workers", "gpu_devices", "trajectories_per_gpu")):
                kwargs.setdefault("scheduler", "static")
            return public_run(config, **kwargs)
        self.run = legacy_run


del _LegacyEnsemble


if __name__ == "__main__":
    unittest.main()
