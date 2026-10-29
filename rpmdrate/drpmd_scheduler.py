"""Public scheduling entrypoint; retain the original static scheduler unchanged."""
from __future__ import annotations
from . import drpmd_parallel as _static


def normalize_input(config):
    """Separate scheduler settings before validating the original physical input."""
    if not isinstance(config, dict):
        raise ValueError("Input must be a JSON object")
    config = dict(config)
    parallel = config.get("parallel", {})
    if not isinstance(parallel, dict):
        raise ValueError("parallel must be an object")
    parallel = dict(parallel)
    options = {}
    for name in ("scheduler", "task_prefetch"):
        if name in parallel:
            value = parallel.pop(name)
            if value is None:
                raise ValueError(f"{name} must not be null; omit unused settings")
            options[name] = value
    config["parallel"] = parallel
    physical, remaining = _static.normalize_input(config)
    return physical, {**remaining, **options}


def run_ensemble(config, *, scheduler="dynamic", task_prefetch=None, **kwargs):
    """Dynamic tasks by default; static retains the legacy fixed-chain contract."""
    if scheduler not in ("dynamic", "static"):
        raise ValueError("scheduler must be dynamic or static")
    if scheduler == "static":
        if task_prefetch is not None:
            raise ValueError("task_prefetch applies only to the dynamic scheduler")
        return _static.run_ensemble(config, **kwargs)
    from .drpmd_pool import run_dynamic
    return run_dynamic(config, task_prefetch=task_prefetch, **kwargs)
