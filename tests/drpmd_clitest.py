"""End-to-end dRPMD CLI tests using real subprocesses and output files.

Short trajectories here verify execution, provenance and failure reporting.
They intentionally remain unresolved and do not establish chemical rates or
thermalization convergence.  Multicore coverage requires a prebuilt OpenMP
kernel; CUDA failure coverage is skipped on hosts with a usable CUDA device.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rpmdrate.drpmd_integrator import _load_native


_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = '''"""Temporary importable free-particle PES for CLI regressions."""
import numpy as np

_production_calls = 0

def get_potential(q, fail_after=None):
    global _production_calls
    # Fragment 0 is fixed at z=0 during thermalization; launch places its
    # centroid at negative z.  This distinguishes phases without knowing how
    # many thermal PES evaluations the driver performs.
    if float(q[2, 0, :].mean()) < -1.0:
        _production_calls += 1
        if fail_after is not None and _production_calls >= fail_after:
            return np.zeros(q.shape[2]), np.zeros_like(q), -7
    return np.zeros(q.shape[2]), np.zeros_like(q), 0
'''


class DirectRPMDCLITest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="drpmd-cli-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "cli_test_pes.py").write_text(_FIXTURE)

    def free_config(self):
        config = json.loads((_ROOT / "examples/drpmd/free_particles.json").read_text())
        config.update(thermalization_steps=30, sampling_steps=10, evolution_steps=100,
                      save_every=25, trajectories=1)
        config["potential"] = {"type": "module", "module": "cli_test_pes", "initializer": None}
        return config

    def hhoo_config(self):
        config = json.loads((_ROOT / "examples/drpmd/hhoo_smoke.json").read_text())
        config["potential"]["weights"] = str(_ROOT / "examples/H-H-O-O/pes/txt_1014")
        return config

    def invoke(self, config, name, *, flags=(), legacy=False, expected=0, output=None):
        input_path = self.directory / (name + ".json")
        input_path.write_text(json.dumps(config))
        output = self.directory / name if output is None else Path(output)
        command = [sys.executable, str(_ROOT / "rpmdrate.py"), "--drpmd"] if legacy else [
            sys.executable, "-m", "rpmdrate.drpmd"]
        command += [str(input_path), "--output", str(output), *map(str, flags)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        # Explicitly keep optional BLAS implementations from consuming cores
        # before the application's own threadpool control has initialized.
        environment.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(command, cwd=self.directory, env=environment,
                                capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, expected,
                         "Command: {}\nstdout:\n{}\nstderr:\n{}".format(command, result.stdout, result.stderr))
        return output, result

    def summary(self, output):
        return json.loads((output / "summary.json").read_text())

    def assert_unresolved(self, output):
        summary = self.summary(output)
        self.assertEqual(summary["counts"], {"product": 0, "nonreactive": 0, "unresolved": 1, "failed": 0})
        self.assertFalse(summary["all_trajectories_resolved"])
        self.assertIsNone(summary["total_reaction"]["rate_cm3_molecule_s"])
        self.assertIsNone(summary["total_reaction"]["iid_wilson95_cm3_molecule_s"])
        self.assertGreater(summary["total_reaction"]["possible_upper_cm3_molecule_s"], 0)
        for estimate in summary["channels"].values():
            self.assertIsNone(estimate["rate_cm3_molecule_s"])
        return summary

    def assert_states_match(self, left, right, filename):
        with np.load(left / filename, allow_pickle=False) as a, np.load(right / filename, allow_pickle=False) as b:
            np.testing.assert_allclose(a["q"], b["q"], rtol=2e-11, atol=2e-11)
            np.testing.assert_allclose(a["p"], b["p"], rtol=2e-11, atol=2e-10)
            np.testing.assert_array_equal(a["masses"], b["masses"])

    def test_module_serial_and_legacy_bead_cli_preserve_same_seed_states(self):
        native = _load_native()
        if native is None or not native.drpmd_has_openmp():
            self.skipTest("Build the C/OpenMP kernel for end-to-end multicore coverage")
        config = self.free_config()
        serial, _ = self.invoke(config, "serial", flags=("--mode", "serial"))
        parallel, _ = self.invoke(config, "bead", legacy=True,
                                  flags=("--mode", "bead", "--workers", "2"))
        serial_summary, parallel_summary = self.assert_unresolved(serial), self.assert_unresolved(parallel)
        self.assertEqual((serial_summary["mode"], serial_summary["workers"]), ("serial", 1))
        self.assertEqual((parallel_summary["mode"], parallel_summary["workers"]), ("bead", 2))
        manifest = json.loads((parallel / "input.json").read_text())
        self.assertTrue(manifest["native_cayley"])
        self.assertEqual(len(set(manifest["pes_worker_pids"])), 2)
        for filename in ("thermal_state.npz", "trajectory_000001_initial.npz", "trajectory_000001_final.npz"):
            self.assert_states_match(serial, parallel, filename)

    def test_hhoo_saved_state_reuse_accepts_identical_relocated_weights_and_rejects_changes(self):
        config = self.hhoo_config()
        original, _ = self.invoke(config, "hhoo-original")
        self.assert_unresolved(original)
        thermal = original / "thermal_state.npz"
        # Content provenance should permit relocating the exact model bytes.
        copied = self.directory / "copied-hhoo-weights.txt"
        copied.write_bytes(Path(config["potential"]["weights"]).read_bytes())
        config["potential"]["weights"] = str(copied)
        reused, result = self.invoke(config, "hhoo-reuse", flags=("--thermal-state", thermal))
        self.assertNotIn("Thermalizing:", result.stdout)
        self.assert_unresolved(reused)
        self.assert_states_match(original, reused, "thermal_state.npz")
        self.assert_states_match(original, reused, "trajectory_000001_initial.npz")
        self.assert_states_match(original, reused, "trajectory_000001_final.npz")
        manifest = json.loads((reused / "input.json").read_text())
        self.assertEqual(manifest["source_thermal_state"], str(thermal))
        self.assertEqual(manifest["potential_identity"], json.loads((original / "input.json").read_text())["potential_identity"])

        # Alter one finite model coefficient, preserving parseable architecture.
        contents = copied.read_text()
        lines = contents.splitlines(keepends=True)
        last_numeric = len(lines) - 1
        while not lines[last_numeric].strip():
            last_numeric -= 1
        fields = lines[last_numeric].split()
        fields[-1] = format(float(fields[-1].replace("D", "E").replace("d", "e")) + 1e-6, ".17g")
        lines[last_numeric] = " ".join(fields) + "\n"
        copied.write_text("".join(lines))
        rejected, result = self.invoke(config, "hhoo-modified", expected=2,
                                       flags=("--thermal-state", thermal))
        self.assertIn("matching potential identity", result.stderr)
        self.assertFalse(rejected.exists())

    def test_existing_output_is_refused_and_not_changed(self):
        config = self.free_config()
        output, _ = self.invoke(config, "first")
        before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
        _, result = self.invoke(config, "second", output=output, expected=2)
        self.assertIn("already exists", result.stderr)
        after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_gpu_request_fails_instead_of_using_cpu_when_cuda_unavailable(self):
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            self.skipTest("This host has CUDA; this regression specifically covers unavailable CUDA")
        output, result = self.invoke(self.hhoo_config(), "gpu-unavailable", expected=2,
                                     flags=("--mode", "gpu"))
        self.assertRegex(result.stderr, "CUDA|PyTorch")
        self.assertFalse((output / "summary.json").exists())
        self.assertNotIn('"mode": "serial"', result.stdout)

    def test_malformed_json_roots_fail_cleanly_without_outputs(self):
        for index, config in enumerate((None, [], "invalid", {"potential": []})):
            with self.subTest(config=config):
                output, result = self.invoke(config, "malformed-" + str(index), expected=2)
                self.assertIn("dRPMD error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_production_pes_failure_preserves_last_state_and_reports_failed_unknown_rate(self):
        config = self.free_config()
        config["trajectories"] = 3
        config["potential"]["kwargs"] = {"fail_after": 4}
        output, _ = self.invoke(config, "failed-pes", expected=1)
        summary = self.summary(output)
        self.assertEqual(summary["counts"], {"product": 0, "nonreactive": 0, "unresolved": 0, "failed": 1})
        self.assertEqual(summary["requested_trajectories"], 3)
        self.assertEqual(summary["trajectories"], 1)
        self.assertFalse(summary["all_trajectories_resolved"])
        self.assertIsNone(summary["total_reaction"]["rate_cm3_molecule_s"])
        with (output / "trajectories.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(int(records[0]["steps"]), 2)
        self.assertIn("info=-7", records[0]["error"])
        with np.load(output / "trajectory_000001_final.npz", allow_pickle=False) as data:
            self.assertTrue(np.isfinite(data["q"]).all())
            self.assertTrue(np.isfinite(data["p"]).all())
            metadata = json.loads(str(data["metadata"].item()))
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["steps"], 2)
        self.assertFalse((output / "trajectory_000002_initial.npz").exists())


if __name__ == "__main__":
    unittest.main()
