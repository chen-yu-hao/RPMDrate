"""Planning, real spawned parents, nested bead pools, and optional CUDA tests."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rpmdrate.drpmd_parallel import collect_chain, make_plan, normalize_input, _states

_FAIL_WHEN_FILE = None


def initialize_potential(fail_chain=None, hard_exit=False, fail_when_file=None):
    """Importable test PES initialization, including controlled process failures."""
    global _FAIL_WHEN_FILE
    _FAIL_WHEN_FILE = fail_when_file
    if fail_chain is not None and mp.current_process().name == f"drpmd-parent-{fail_chain}":
        if hard_exit:
            os._exit(19)
        raise RuntimeError("Intentional parent initialization failure")


def get_potential(q):
    if _FAIL_WHEN_FILE and Path(_FAIL_WHEN_FILE).is_file():
        raise RuntimeError("Intentional parent failure after the first saved child")
    return np.zeros(q.shape[2]), np.zeros_like(q, order="F")


def tiny_config(total=5):
    from rpmdrate.drpmd import AMU, KB, TIME_FS
    return {
        "atoms": ["H", "H"], "masses_amu": [1 / AMU, 2 / AMU],
        "fragments": [[0], [1]], "temperature_K": 1 / (2 * KB),
        "nbeads": 4, "dt_fs": 0.025 * TIME_FS,
        "initial_geometry_bohr": [[0, 0, 0], [0, 0, 5]],
        "thermalization_steps": 28, "sampling_steps": 9, "andersen_interval": 7,
        "trajectories": total, "evolution_steps": 12,
        "initial_separation_bohr": 5., "bmax_bohr": 1., "seed": 812,
        "bond_cutoffs_bohr": {"H-H": 1.}, "channels": {},
        "exit_distance_bohr": 4., "check_interval": 5, "persistence": 2, "save_every": 4,
        "potential": {"type": "module", "module": "tests.drpmd_paralleltest"},
    }


class PlanTest(unittest.TestCase):
    def test_balanced_partition_preserves_exact_total_and_global_ids(self):
        plan = make_plan(10003, trajectory_workers=8)
        chains = plan["chains"]
        self.assertEqual([c.trajectories for c in chains], [1251] * 3 + [1250] * 5)
        ids = [j for c in chains for j in range(c.first_trajectory, c.first_trajectory + c.trajectories)]
        self.assertEqual(ids, list(range(1, 10004)))
        self.assertEqual(len({c.seed for c in chains}), 8)

    def test_no_empty_chains_when_workers_exceed_N(self):
        plan = make_plan(3, trajectory_workers=32)
        self.assertEqual(plan["requested_trajectory_workers"], 32)
        self.assertEqual(plan["active_trajectory_workers"], 3)
        self.assertEqual([c.trajectories for c in plan["chains"]], [1, 1, 1])

    def test_parallelism_is_orthogonal_to_bead_backend(self):
        serial = make_plan(11, trajectory_workers=3, seed=1)
        bead = make_plan(11, mode="bead", workers=4, trajectory_workers=3, seed=1)
        self.assertEqual(serial["chains"], bead["chains"])
        self.assertEqual(bead["cpu_compute_slots"], 12)

    def test_multigpu_one_trajectory_per_card(self):
        plan = make_plan(10, mode="gpu", gpu_devices=[2, 0])
        self.assertEqual([c.gpu_device for c in plan["chains"]], [2, 0])
        self.assertEqual(plan["active_slots_per_gpu"], {"2": 1, "0": 1})

    def test_multigpu_multiple_trajectories_per_card(self):
        plan = make_plan(17, mode="gpu", gpu_devices=[0, 1], trajectories_per_gpu=3)
        self.assertEqual([c.gpu_device for c in plan["chains"]], [0, 1, 0, 1, 0, 1])
        self.assertEqual([c.trajectories for c in plan["chains"]], [3, 3, 3, 3, 3, 2])
        self.assertEqual(plan["active_slots_per_gpu"], {"0": 3, "1": 3})

    def test_explicit_GPU_worker_count_can_be_distributed_unevenly(self):
        plan = make_plan(20, mode="gpu", gpu_devices=[0, 1], trajectory_workers=3)
        self.assertEqual(plan["active_slots_per_gpu"], {"0": 2, "1": 1})

    def test_reject_invalid_or_ambiguous_resources(self):
        invalid = [dict(trajectory_workers=0), dict(trajectory_workers=True),
                   dict(trajectory_workers=1.5), dict(mode="serial", workers=2),
                   dict(mode="bead", workers=1), dict(mode="gpu", workers=2),
                   dict(gpu_devices=[0]), dict(mode="gpu", gpu_devices=[]),
                   dict(mode="gpu", gpu_devices=[0, 0]), dict(mode="gpu", gpu_devices=[-1]),
                   dict(mode="gpu", gpu_devices=[True]), dict(mode="gpu", gpu_devices="0,1"),
                   dict(mode="gpu", trajectories_per_gpu=0),
                   dict(mode="gpu", gpu_devices=[0, 1], trajectories_per_gpu=2, trajectory_workers=3)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(10, **kwargs)

    def test_seed_streams_are_stable_for_chain_ids(self):
        first = make_plan(10, trajectory_workers=3, seed=731)
        repeated = make_plan(10, trajectory_workers=3, seed=731)
        extended = make_plan(100, trajectory_workers=3, seed=731)
        self.assertEqual(first, repeated)
        self.assertEqual([c.seed for c in first["chains"]], [c.seed for c in extended["chains"]])

    def test_sampling_time_is_exact_steps_and_does_not_mutate_input(self):
        original = {"dt_fs": .1, "sampling_time_fs": 10., "parallel": {"trajectory_workers": 4}}
        physical, options = normalize_input(original)
        self.assertEqual(physical, {"dt_fs": .1, "sampling_steps": 100})
        self.assertEqual(options, {"trajectory_workers": 4})
        self.assertIn("sampling_time_fs", original)
        self.assertIn("parallel", original)

    def test_reject_conflicting_or_fractional_sampling_intervals(self):
        for config in ({"dt_fs": .1, "sampling_time_fs": .25},
                       {"dt_fs": .1, "sampling_time_fs": 10., "sampling_steps": 2},
                       {"dt_fs": 0, "sampling_time_fs": 10.},
                       {"dt_fs": .1, "sampling_time_fs": float("inf")},
                       {"parallel": {"trajectory_worker": 4}}, {"parallel": []},
                       {"parallel": {"trajectory_workers": None}}, []):
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_input(config)

    def test_thermal_reuse_requires_distinct_per_chain_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            path.write_bytes(b"path validation only")
            self.assertEqual(_states(path, 1), [str(path)])
            with self.assertRaisesRegex(ValueError, "Parallel reuse"):
                _states(path, 2)
            with self.assertRaisesRegex(ValueError, "distinct"):
                _states([path, path], 2)

    def test_missing_worker_results_do_not_shrink_the_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            chain = make_plan(7, trajectory_workers=3)["chains"][0]
            records = collect_chain(Path(directory), chain, {"error": "worker died"})
            self.assertEqual(len(records), 3)
            self.assertTrue(all(row["status"] == "failed" and not row["launched"] for row in records))
            self.assertEqual([row["trajectory"] for row in records], [1, 2, 3])


class SpawnedEnsembleTest(unittest.TestCase):
    def setUp(self):
        from rpmdrate.drpmd import run
        self.run = run
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def execute(self, name, config=None, **kwargs):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            summary = self.run(config or tiny_config(), output=output, **kwargs)
        return output, summary

    def test_exact_total_real_processes_independent_parents_and_clocks(self):
        config = tiny_config(7)
        root, summary = self.execute("serial", config, trajectory_workers=3)
        self.assertEqual(summary["trajectories"], 7)
        self.assertEqual(summary["requested_trajectories"], 7)
        self.assertEqual(summary["counts"]["failed"], 0)
        self.assertEqual(summary["launched_trajectories"], 7)
        self.assertEqual(len({r["pid"] for r in summary["worker_reports"]}), 3)
        self.assertTrue(all(r["ok"] and r["pid"] != os.getpid() for r in summary["worker_reports"]))
        records = json.loads((root / "trajectories.json").read_text())
        self.assertEqual([row["trajectory"] for row in records], list(range(1, 8)))
        states = []
        for chain in summary["parent_chains"]:
            directory = root / f"chains/chain_{chain['chain_id']:04d}"
            child_summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(child_summary["thermal_steps_at_last_sample"],
                             config["thermalization_steps"] + chain["trajectories"] * config["sampling_steps"])
            with np.load(directory / "thermal_state.npz") as data:
                states.append(data["q"])
        self.assertFalse(np.array_equal(states[0], states[1]))

    def test_chain_matches_standalone_replay_with_recorded_seed(self):
        config = tiny_config(5)
        root, summary = self.execute("ensemble", config, trajectory_workers=2)
        chain = summary["parent_chains"][1]
        replay = dict(config, seed=chain["seed"], trajectories=chain["trajectories"])
        direct, _ = self.execute("replay", replay)
        directory = root / f"chains/chain_{chain['chain_id']:04d}"
        for index in range(1, chain["trajectories"] + 1):
            for phase in ("initial", "final"):
                name = f"trajectory_{index:06d}_{phase}.npz"
                with np.load(directory / name) as left, np.load(direct / name) as right:
                    np.testing.assert_array_equal(left["q"], right["q"])
                    np.testing.assert_array_equal(left["p"], right["p"])

    def test_children_do_not_modify_the_parent_chain(self):
        first, _ = self.execute("short", tiny_config(4), trajectory_workers=2)
        config = tiny_config(4)
        config["evolution_steps"] = 23
        second, _ = self.execute("long", config, trajectory_workers=2)
        for chain in (1, 2):
            for child in (1, 2):
                name = f"chains/chain_{chain:04d}/trajectory_{child:06d}_initial.npz"
                with np.load(first / name) as left, np.load(second / name) as right:
                    np.testing.assert_array_equal(left["q"], right["q"])
                    np.testing.assert_array_equal(left["p"], right["p"])

    def test_nested_bead_pools_work_and_match_serial_ensembles(self):
        from rpmdrate.drpmd_integrator import _load_native
        if _load_native() is None:
            self.skipTest("Native OpenMP kernel is not built")
        serial, summary_s = self.execute("serial", tiny_config(4), trajectory_workers=2)
        bead, summary_b = self.execute("bead", tiny_config(4), mode="bead", workers=2, trajectory_workers=2)
        self.assertEqual(summary_s["counts"], summary_b["counts"])
        pes_pids = []
        for chain in (1, 2):
            directory = Path(f"chains/chain_{chain:04d}")
            manifest = json.loads((bead / directory / "input.json").read_text())
            self.assertEqual(len(manifest["pes_worker_pids"]), 2)
            pes_pids.extend(manifest["pes_worker_pids"])
            for child in (1, 2):
                name = directory / f"trajectory_{child:06d}_final.npz"
                with np.load(serial / name) as left, np.load(bead / name) as right:
                    np.testing.assert_allclose(left["q"], right["q"], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(left["p"], right["p"], atol=1e-12, rtol=0)
        self.assertEqual(len(set(pes_pids)), 4)

    def test_one_failed_parent_does_not_stop_other_parents_or_drop_counts(self):
        config = tiny_config(7)
        config["potential"]["initializer_kwargs"] = {"fail_chain": 1}
        root, summary = self.execute("failure", config, trajectory_workers=3)
        self.assertEqual(summary["trajectories"], 7)
        self.assertEqual(summary["counts"]["failed"], 3)
        self.assertEqual(summary["counts"]["unresolved"], 4)
        self.assertEqual(summary["not_launched_trajectories"], 3)
        self.assertIsNone(summary["total_reaction"]["rate_cm3_molecule_s"])
        self.assertTrue((root / "summary.json").is_file())

    def test_hard_worker_exit_is_recorded_without_hanging(self):
        config = tiny_config(3)
        config["potential"]["initializer_kwargs"] = {"fail_chain": 1, "hard_exit": True}
        _, summary = self.execute("hard-exit", config, trajectory_workers=2)
        self.assertEqual(summary["trajectories"], 3)
        self.assertEqual(summary["counts"]["failed"], 2)
        self.assertEqual(summary["worker_reports"][0]["exitcode"], 19)
        self.assertEqual(summary["counts"]["unresolved"], 1)

    def test_completed_child_is_recovered_when_later_parent_sampling_fails(self):
        config = tiny_config(3)
        root = self.root / "recovery"
        config["potential"]["initializer_kwargs"] = {
            "fail_when_file": str(root / "chains/chain_0001/trajectory_000001_final.npz")}
        _, summary = self.execute("recovery", config, trajectory_workers=1)
        records = json.loads((root / "trajectories.json").read_text())
        self.assertEqual(summary["trajectories"], 3)
        self.assertEqual(summary["counts"]["unresolved"], 1)
        self.assertEqual(summary["counts"]["failed"], 2)
        self.assertTrue(records[0]["recovered_from_checkpoint"])

    def test_parallel_thermal_state_reuse_has_no_hidden_rethermalization(self):
        first, _ = self.execute("initial", tiny_config(2), trajectory_workers=2)
        config = tiny_config(2)
        config["thermalization_steps"] = 0
        second, summary = self.execute("reuse", config, trajectory_workers=2, thermal_state=first)
        self.assertEqual(summary["counts"]["failed"], 0)
        for chain in (1, 2):
            name = f"chains/chain_{chain:04d}/thermal_state.npz"
            with np.load(first / name) as left, np.load(second / name) as right:
                np.testing.assert_array_equal(left["q"], right["q"])
                np.testing.assert_array_equal(left["p"], right["p"])

    def test_invalid_parallel_input_does_not_create_output(self):
        config = tiny_config()
        config["termalization_steps"] = 5
        output = self.root / "invalid"
        with self.assertRaises(ValueError):
            self.run(config, output=output, trajectory_workers=2)
        self.assertFalse(output.exists())

    def test_JSON_options_and_CLI_override_total(self):
        config = tiny_config(100)
        config["parallel"] = {"trajectory_workers": 2}
        config["sampling_time_fs"] = config.pop("sampling_steps") * config["dt_fs"]
        input_file, output = self.root / "input.json", self.root / "cli"
        input_file.write_text(json.dumps(config))
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-m", "rpmdrate.drpmd", str(input_file),
                                 "--trajectories", "3", "--output", str(output)],
                                cwd=repository, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["requested_trajectories"], 3)
        self.assertEqual(summary["trajectory_workers"], 2)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "occupied"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run(tiny_config(), output=output, trajectory_workers=2)
        self.assertEqual(list(output.iterdir()), [])


class CudaEnsembleTest(unittest.TestCase):
    def check_layout(self, cards, per_card):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available() or torch.cuda.device_count() < cards:
            self.skipTest(f"Needs {cards} real CUDA device(s)")
        from rpmdrate.drpmd import run
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            model = torch.jit.trace(lambda q: (q * q * 0.).sum(dim=(1, 2)),
                                    torch.zeros((4, 2, 3), dtype=torch.float64))
            model.save(str(root / "free.pt"))
            total = cards * per_card + 1
            gpu_config = tiny_config(total)
            gpu_config["potential"] = {"type": "torchscript", "model": str(root / "free.pt")}
            gpu = run(gpu_config, mode="gpu", gpu_devices=list(range(cards)),
                      trajectories_per_gpu=per_card, output=root / "gpu")
            cpu = run(tiny_config(total), trajectory_workers=cards * per_card, output=root / "cpu")
            self.assertEqual(gpu["counts"], cpu["counts"])
            self.assertEqual(gpu["counts"]["failed"], 0)
            self.assertEqual(gpu["active_slots_per_gpu"], {str(card): per_card for card in range(cards)})
            for report in gpu["worker_reports"]:
                self.assertEqual(report["cuda_current_device"], report["gpu_device"])
            for chain in gpu["parent_chains"]:
                for child in range(1, chain["trajectories"] + 1):
                    name = f"chains/chain_{chain['chain_id']:04d}/trajectory_{child:06d}_final.npz"
                    with np.load(root / "gpu" / name) as left, np.load(root / "cpu" / name) as right:
                        np.testing.assert_allclose(left["q"], right["q"], atol=1e-11, rtol=0)
                        np.testing.assert_allclose(left["p"], right["p"], atol=1e-11, rtol=0)

    def test_single_gpu_multiple_trajectories(self):
        self.check_layout(cards=1, per_card=2)

    def test_multiple_gpus_one_trajectory_per_card(self):
        self.check_layout(cards=2, per_card=1)

    def test_multiple_gpus_multiple_trajectories_per_card(self):
        self.check_layout(cards=2, per_card=2)


if __name__ == "__main__":
    unittest.main()
