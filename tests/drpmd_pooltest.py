"""Dynamic queue, real spawn workers, migration, receipts and failure regressions."""
from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from rpmdrate.drpmd import run, ThermalState, Dynamics, AMU, KB, TIME_FS, sample_momenta
from rpmdrate.drpmd_parallel import make_plan
from rpmdrate.drpmd_scheduler import normalize_input
from rpmdrate.drpmd_pool import TaskQueue, task_rng
from tests.drpmd_paralleltest import tiny_config

_SLEEP = 0.


def initialize_potential(slow_worker=None, delay=0.01, fail_worker=None,
                         crash_kind=None, crash_trajectory=1, after_receipt=False):
    """Test-only fault injection: no crash/delay knobs exist in production."""
    global _SLEEP
    name = mp.current_process().name
    _SLEEP = delay if name == f"drpmd-worker-{slow_worker}" else 0.
    if name == f"drpmd-worker-{fail_worker}":
        os._exit(19)
    if crash_kind:
        from rpmdrate.drpmd_pool import _Runtime
        method = getattr(_Runtime, crash_kind)
        def crashing(self, task):
            if task["trajectory"] == crash_trajectory:
                if after_receipt:
                    method(self, task)
                os._exit(23)
            return method(self, task)
        setattr(_Runtime, crash_kind, crashing)


def get_potential(q):
    if _SLEEP and np.linalg.norm(q[:, 0].mean(axis=1)) > 0.1:
        time.sleep(_SLEEP)
    return np.zeros(q.shape[2]), np.zeros_like(q, order="F")


class QueueTest(unittest.TestCase):
    def test_exclusive_parent_exact_total_and_bounded_queue(self):
        chains = make_plan(23, trajectory_workers=4)["chains"]
        queue = TaskQueue(chains, [None]*4, 5)
        rng = np.random.default_rng(10)
        collisions = []
        while not queue.exhausted:
            for worker in range(1, 5):
                if worker not in queue.inflight:
                    queue.take(worker)
            self.assertEqual(len(queue.sampling), sum(t["kind"] == "sample" for t in queue.inflight.values()))
            self.assertLessEqual(len(queue.ready)+len(queue.sampling), 5)
            self.assertTrue(queue.inflight)
            worker = int(rng.choice(list(queue.inflight)))
            task = queue.inflight[worker]
            if task["kind"] == "sample":
                queue.finish_sample(worker, f"parent-{task['chain_id']}")
            else:
                collisions.append(task["trajectory"])
                queue.finish_collision(worker)
        self.assertEqual(sorted(collisions), list(range(1, 24)))
        self.assertLessEqual(queue.max_buffered, 5)

    def test_parent_continuation_can_migrate_to_a_different_worker(self):
        queue = TaskQueue(make_plan(4, trajectory_workers=2)["chains"], [None]*2, 4)
        first = queue.take(1)
        queue.finish_sample(1, "parent-one.npz")
        other = queue.take(1)
        self.assertEqual(other["chain_id"], 2)
        second = queue.take(2)
        self.assertEqual((second["chain_id"], second["local_trajectory"], second["source"]), (1, 2, "parent-one.npz"))
        self.assertNotEqual(first["trajectory"], second["trajectory"])

    def test_parent_failure_disables_only_its_future_samples(self):
        queue = TaskQueue(make_plan(4, trajectory_workers=2)["chains"], [None]*2, 2)
        queue.take(1)
        queue.fail(1, "thermal failure")
        task = queue.take(2)
        self.assertEqual(task["chain_id"], 2)
        queue.finish_sample(2, "parent-two")
        self.assertEqual(queue.disabled, {1: "thermal failure"})

    def test_new_options_validate_without_changing_input(self):
        config = tiny_config()
        config["parallel"] = {"scheduler": "dynamic", "trajectory_workers": 2, "task_prefetch": 3}
        physical, options = normalize_input(config)
        self.assertNotIn("parallel", physical)
        self.assertEqual(config["parallel"], options)
        with tempfile.TemporaryDirectory() as d:
            for kwargs in ({"scheduler": "bad"}, {"task_prefetch": 0}, {"task_prefetch": True},
                           {"scheduler": "static", "task_prefetch": 1}):
                with self.assertRaises(ValueError):
                    run(tiny_config(), trajectory_workers=2, output=Path(d)/"invalid", **kwargs)
                self.assertFalse((Path(d)/"invalid").exists())


class DynamicPoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def execute(self, name, total=7, faults=None, **kwargs):
        config = tiny_config(total)
        config["potential"] = {"type": "module", "module": "tests.drpmd_pooltest",
                               "initializer_kwargs": faults or {}}
        directory = self.root/name
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run(config, output=directory, trajectory_workers=kwargs.pop("trajectory_workers", 3), **kwargs)
        return directory, summary, config

    def test_default_dynamic_counts_parent_clocks_and_no_rng_checkpoint(self):
        directory, summary, config = self.execute("normal")
        self.assertEqual(summary["scheduler"], "dynamic")
        self.assertEqual(summary["trajectories"], 7)
        self.assertEqual(summary["counts"]["failed"], 0)
        self.assertEqual(summary["failed_workers"], 0)
        self.assertTrue(all(r["ok"] and r["exitcode"] == 0 for r in summary["worker_reports"]))
        records = json.loads((directory/"trajectories.json").read_text())
        self.assertEqual([r["trajectory"] for r in records], list(range(1, 8)))
        for chain in summary["parent_chains"]:
            state = ThermalState.load(directory/f"chains/chain_{chain['chain_id']:04d}/thermal_latest.npz")
            self.assertEqual(state.metadata["thermal_steps"], config["thermalization_steps"]+chain["trajectories"]*config["sampling_steps"])
            self.assertNotIn("rng_state", state.metadata)
        self.assertLessEqual(summary["max_buffered_tasks"], summary["task_prefetch"])

    def test_parent_qp_and_clock_match_manual_task_local_thermalization(self):
        from rpmdrate.drpmd_backends import SerialPotential
        directory, summary, c = self.execute("continuity", total=4, trajectory_workers=2, task_prefetch=1)
        masses = np.array(c["masses_amu"])*AMU
        fragments = ((0,), (1,))
        for chain in summary["parent_chains"]:
            cid = chain["chain_id"]
            q = np.asfortranarray(np.repeat(np.asarray(c["initial_geometry_bohr"], dtype=float).T[:, :, None], c["nbeads"], axis=2))
            dynamics = Dynamics(masses, c["temperature_K"], c["nbeads"], c["dt_fs"]/TIME_FS, SerialPotential(get_potential))
            rng = task_rng(c["seed"], cid, 1, 0)
            p = sample_momenta(masses, c["nbeads"], 1/(KB*c["temperature_K"]), rng)
            _, clock = dynamics.thermalize(q, p, fragments, c["thermalization_steps"], c["andersen_interval"], rng)
            for local in range(1, chain["trajectories"]+1):
                if local > 1:
                    rng = task_rng(c["seed"], cid, local, 0)
                _, clock = dynamics.thermalize(q, p, fragments, c["sampling_steps"], c["andersen_interval"], rng, clock)
            saved = ThermalState.load(directory/f"chains/chain_{cid:04d}/thermal_latest.npz")
            np.testing.assert_allclose(saved.q, q, atol=1e-13, rtol=0)
            np.testing.assert_allclose(saved.p, p, atol=1e-13, rtol=0)
            self.assertEqual(saved.metadata["thermal_steps"], clock)

    def test_children_cannot_change_parent_even_when_schedule_changes(self):
        first, _, c = self.execute("first", total=6, trajectory_workers=2, faults={"slow_worker": 1})
        c["evolution_steps"] = 21
        with contextlib.redirect_stdout(io.StringIO()):
            run(c, trajectory_workers=2, task_prefetch=1, output=self.root/"second")
        for cid in (1, 2):
            for local in (1, 2, 3):
                path = f"chains/chain_{cid:04d}/trajectory_{local:06d}_initial.npz"
                a, b = ThermalState.load(first/path), ThermalState.load(self.root/"second"/path)
                np.testing.assert_array_equal(a.q, b.q)
                np.testing.assert_array_equal(a.p, b.p)

    def test_fast_worker_takes_other_chains_children(self):
        directory, summary, _ = self.execute("imbalanced", total=12, trajectory_workers=2,
                                            faults={"slow_worker": 1, "delay": 0.025})
        counts = {r["worker_id"]: r["collision_tasks"] for r in summary["worker_reports"]}
        self.assertGreater(counts[2], counts[1])
        rows = json.loads((directory/"trajectories.json").read_text())
        self.assertTrue(any(r["worker_id"] == 2 and r["chain_id"] == 1 for r in rows))
        self.assertEqual(summary["counts"]["failed"], 0)

    def test_startup_worker_death_does_not_discard_its_parent(self):
        _, summary, _ = self.execute("initdeath", total=5, faults={"fail_worker": 1})
        self.assertEqual(summary["trajectories"], 5)
        self.assertEqual(summary["counts"]["failed"], 0)
        self.assertEqual(summary["failed_workers"], 1)
        self.assertEqual(summary["failed_parent_chains"], 0)

    def test_collision_hard_exit_is_unknown_and_others_continue(self):
        _, summary, _ = self.execute("collisiondeath", total=6, faults={"crash_kind": "collision"})
        self.assertEqual(summary["trajectories"], 6)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["counts"]["unresolved"], 5)
        self.assertIsNone(summary["total_reaction"]["rate_cm3_molecule_s"])

    def test_failed_sample_disables_only_its_chain(self):
        _, summary, _ = self.execute("sampledeath", total=6, faults={"crash_kind": "sample"})
        self.assertEqual(summary["failed_parent_chains"], 1)
        self.assertEqual(summary["counts"]["failed"], 2)
        self.assertEqual(summary["counts"]["unresolved"], 4)

    def test_atomic_sample_and_result_receipts_survive_hard_exit(self):
        for kind in ("sample", "collision"):
            with self.subTest(kind=kind):
                _, summary, _ = self.execute("receipt-"+kind, total=6,
                    faults={"crash_kind": kind, "after_receipt": True})
                self.assertEqual(summary["counts"]["failed"], 0)
                self.assertEqual(summary["counts"]["unresolved"], 6)
                self.assertEqual(summary["failed_workers"], 1)

    def test_last_parent_reuse_no_extra_thermalization_for_first_imported_sample(self):
        directory, _, c = self.execute("source", total=6, trajectory_workers=2)
        c["thermalization_steps"] = 0
        c["trajectories"] = 2
        with contextlib.redirect_stdout(io.StringIO()):
            run(c, trajectory_workers=2, thermal_state=directory, output=self.root/"reuse")
        for cid in (1, 2):
            a = ThermalState.load(directory/f"chains/chain_{cid:04d}/thermal_latest.npz")
            b = ThermalState.load(self.root/f"reuse/chains/chain_{cid:04d}/thermal_state.npz")
            np.testing.assert_array_equal(a.q, b.q)
            np.testing.assert_array_equal(a.p, b.p)

    def test_more_workers_than_trajectories_and_no_overwrite(self):
        directory, summary, _ = self.execute("small", total=2, trajectory_workers=10)
        self.assertEqual(summary["trajectory_workers"], 2)
        self.assertEqual(summary["counts"]["failed"], 0)
        with self.assertRaises(FileExistsError):
            run(tiny_config(), trajectory_workers=2, output=directory)

    def test_nested_bead_shared_memory_with_dynamic_pool(self):
        from rpmdrate.drpmd_integrator import _load_native
        if _load_native() is None:
            self.skipTest("Build the native Cayley kernel first")
        directory, summary, _ = self.execute("shared", total=5, trajectory_workers=2,
            mode="bead", workers=2, pes_backend="shared_memory")
        self.assertEqual(summary["counts"]["failed"], 0)
        self.assertTrue(all(len(r["pes_worker_pids"]) == 2 for r in summary["worker_reports"]))
        self.assertEqual(summary["failed_workers"], 0)


if __name__ == "__main__":
    unittest.main()
