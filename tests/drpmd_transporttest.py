"""Transport and C/Fortran ABI tests; timings are benchmarked separately."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
import ctypes as ct
import io
import json
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from rpmdrate.drpmd_backends import PotentialSpec, SerialPotential, PotentialError, BackendUnavailableError
from rpmdrate.drpmd_shared import SharedMemoryPotential
from rpmdrate.drpmd_native_pes import NativePotential, build_hhoo

ROOT = Path(__file__).resolve().parents[1]
MODULE = "tests.drpmd_transport_potential"
WEIGHTS = ROOT / "examples/H-H-O-O/pes/txt_1014"


def geometry(beads):
    xyz = np.array([[0., 0., 20.], [-0.65, 1.72, 0.], [0., 0., 0.], [2.52, 0., 0.]])
    rng = np.random.default_rng(693)
    return np.asfortranarray(xyz.T[:, :, None] + rng.normal(0, 0.015, (3, 4, beads)))


class SharedMemoryTest(unittest.TestCase):
    def test_uneven_and_single_bead_match_serial_and_preserve_old_results(self):
        spec = PotentialSpec(MODULE)
        rng = np.random.default_rng(44)
        with SerialPotential(spec) as serial:
            for beads in (1, 2, 7, 32):
                shape = (3, 4, beads)
                with SharedMemoryPotential(spec, 3, shape) as shared:
                    q = np.asfortranarray(rng.normal(size=shape))
                    before = q.copy()
                    first = shared.evaluate(q)
                    self.assertEqual(len(set(shared.worker_pids)), min(beads, 3))
                    self.assertEqual(shared.worker_pids, shared.last_worker_pids)
                    for left, right in zip(first, serial.evaluate(q)):
                        np.testing.assert_array_equal(left, right)
                    shared.evaluate(q*2)
                    for left, right in zip(first, serial.evaluate(q)):
                        np.testing.assert_array_equal(left, right)
                    np.testing.assert_array_equal(q, before)
                    self.assertTrue(first[1].flags.f_contiguous)
                self.assertTrue(shared.closed)
                with self.assertRaises(FileNotFoundError):
                    SharedMemory(name=shared.shared_memory_name)
                shared.close()

    def test_initializer_once_and_mutating_callback_cannot_change_input(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/"init.log"
            spec = PotentialSpec(MODULE, initializer_kwargs={"log": str(log)}, kwargs={"action": "mutate"})
            q = np.ones((3, 2, 5), order="F")
            with SharedMemoryPotential(spec, 2, q.shape) as shared:
                for _ in range(6):
                    _, g = shared.evaluate(q)
                    np.testing.assert_array_equal(g, q)
                pids = [int(x) for x in log.read_text().splitlines()]
                self.assertEqual(sorted(pids), sorted(shared.worker_pids))
            np.testing.assert_array_equal(q, 1.)

    def test_failures_and_timeout_release_all_processes_and_memory(self):
        q = np.zeros((3, 2, 4), order="F")
        for action in ("raise", "nan", "exit", "sleep"):
            with self.subTest(action=action):
                spec = PotentialSpec(MODULE, kwargs={"action": action})
                shared = SharedMemoryPotential(spec, 2, q.shape, timeout=0.2 if action == "sleep" else 10.)
                with self.assertRaises(PotentialError):
                    shared.evaluate(q)
                self.assertTrue(shared.closed)
                self.assertTrue(all(not process.is_alive() for process in shared._processes))
                with self.assertRaises(FileNotFoundError):
                    SharedMemory(name=shared.shared_memory_name)
                with self.assertRaises(PotentialError):
                    shared.evaluate(q)

    def test_initialization_failure_is_reported(self):
        with self.assertRaisesRegex(PotentialError, "initializer failure"):
            SharedMemoryPotential(PotentialSpec(MODULE, initializer_kwargs={"fail": True}), 2, (3, 2, 4))

    def test_invalid_input_does_not_poison_live_backend(self):
        q = np.ones((3, 2, 4), order="F")
        with SharedMemoryPotential(PotentialSpec(MODULE), 2, q.shape) as shared:
            for bad in (q.astype(complex), q[:, :, :2], np.full(q.shape, np.nan)):
                with self.assertRaises(ValueError):
                    shared.evaluate(bad)
            np.testing.assert_array_equal(shared.evaluate(q)[1], q)
        for shape in ((2, 2, 4), (3, 0, 4), (3, 2, True)):
            with self.assertRaises(ValueError):
                SharedMemoryPotential(PotentialSpec(MODULE), 2, shape)

    def test_instance_workspaces_are_independent_and_same_instance_calls_serialize(self):
        q = np.ones((3, 2, 7), order="F")
        with SharedMemoryPotential(PotentialSpec(MODULE), 2, q.shape) as first, \
             SharedMemoryPotential(PotentialSpec(MODULE, initializer_kwargs={"spring": 2}), 2, q.shape) as second:
            self.assertNotEqual(first.shared_memory_name, second.shared_memory_name)
            with ThreadPoolExecutor(max_workers=3) as pool:
                a = pool.submit(first.evaluate, q)
                b = pool.submit(first.evaluate, q*3)
                c = pool.submit(second.evaluate, q)
                np.testing.assert_array_equal(a.result()[1], q)
                np.testing.assert_array_equal(b.result()[1], q*3)
                np.testing.assert_array_equal(c.result()[1], q*2)


@unittest.skipUnless(shutil.which("gcc"), "native C ABI tests require gcc")
class NativeCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.library = cls.directory/"harmonic.so"
        source = ROOT/"examples/native_pes/harmonic.c"
        subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-fopenmp", str(source), "-o", str(cls.library)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_c_layout_sign_threads_and_result_lifetimes(self):
        rng = np.random.default_rng(845)
        for beads in (1, 2, 7, 64):
            q = np.asfortranarray(rng.normal(size=(3, 4, beads)))
            expected = 0.5*np.sum(q*q, axis=(0, 1))
            for threads in (1, 2):
                with NativePotential(self.library, threads=threads) as native:
                    v, g = native.evaluate(q)
                    np.testing.assert_allclose(v, expected, rtol=2e-15, atol=2e-15)
                    np.testing.assert_array_equal(g, q)
                    self.assertEqual(native.last_threads_used, threads)
                    native.evaluate(q+2)
                    np.testing.assert_array_equal(g, q)
                    native.evaluate(np.array(q, order="C", dtype=np.float32))
                    with self.assertRaises(ValueError):
                        native.evaluate(q.astype(complex))
                    with self.assertRaises(ValueError):
                        native.evaluate(np.full(q.shape, np.nan))
                with self.assertRaises(PotentialError):
                    native.evaluate(q)

    def test_no_hidden_coordinate_copy_for_already_fortran_FP64_input(self):
        q = np.ones((3, 4, 7), order="F")
        with NativePotential(self.library) as native:
            function = native._lib.drpmd_pes_eval
            address = []
            def record(*args):
                address.append(ct.cast(args[2], ct.c_void_p).value)
                return function(*args)
            # Registry shares CDLL handles: always restore the original symbol.
            native._lib.drpmd_pes_eval = record
            try:
                native.evaluate(q)
            finally:
                native._lib.drpmd_pes_eval = function
            self.assertEqual(address, [q.ctypes.data])
            np.testing.assert_array_equal(q, 1.)

    def test_openmp_is_not_silently_faked(self):
        path = self.directory/"serial.so"
        subprocess.run(["gcc", "-shared", "-fPIC", str(ROOT/"examples/native_pes/harmonic.c"), "-o", str(path)], check=True)
        with self.assertRaisesRegex(BackendUnavailableError, "OpenMP"):
            NativePotential(path, threads=2)
        with NativePotential(path) as native:
            native.evaluate(np.ones((3, 2, 5)))
            self.assertEqual(native.last_threads_used, 1)

    def test_changed_model_bytes_are_rejected_in_same_process(self):
        path = self.directory/"model-library.so"
        shutil.copyfile(self.library, path)
        model = self.directory/"model.dat"
        model.write_text("first version")
        NativePotential(path, model=model).close()
        model.write_text("second version")
        with self.assertRaisesRegex(PotentialError, "bytes changed"):
            NativePotential(path, model=model)

    def test_bad_abi_and_unwritten_outputs_are_rejected(self):
        original = (ROOT/"examples/native_pes/harmonic.c").read_text()
        variants = {
            "abi": original.replace("drpmd_pes_abi_version(void) { return 1; }", "drpmd_pes_abi_version(void) { return 9; }"),
            "init": original.replace("(void)path; return 0;", "(void)path; return 19;"),
            "unwritten": original.replace("g[i] = q[i];", "/* leave g unwritten */"),
        }
        for name, source in variants.items():
            cfile, library = self.directory/(name+".c"), self.directory/(name+".so")
            cfile.write_text(source)
            subprocess.run(["gcc", "-shared", "-fPIC", "-fopenmp", "-I", str(ROOT/"examples/native_pes"),
                            str(cfile), "-o", str(library)], check=True)
            with self.assertRaises(PotentialError):
                with NativePotential(library) as native:
                    native.evaluate(np.ones((3, 2, 5)))


@unittest.skipUnless(shutil.which("gfortran") and WEIGHTS.exists(), "requires gfortran and repository HHOO weights")
class NativeHHOOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = build_hhoo(Path(cls.tmp.name)/"hhoo.so")
        cls.spec = PotentialSpec("rpmdrate.drpmd_hhoo", initializer_kwargs={"weights_path": str(WEIGHTS)})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unchanged_fortran_matches_numpy_and_native_thread_counts(self):
        with SerialPotential(self.spec) as reference:
            for beads in (1, 2, 7, 32):
                q = geometry(beads)
                expected = reference.evaluate(q)
                with NativePotential(self.library, WEIGHTS, threads=1) as scalar:
                    exact = scalar.evaluate(q)
                for threads in (1, 2, 4):
                    with NativePotential(self.library, WEIGHTS, threads=threads) as native:
                        actual = native.evaluate(q)
                        for a, b, c in zip(actual, expected, exact):
                            np.testing.assert_allclose(a, b, rtol=0, atol=2e-12)
                            np.testing.assert_array_equal(a, c)
                        self.assertEqual(native.last_threads_used, threads)

    def test_bad_bead_returns_status_and_does_not_reuse_old_outputs(self):
        q = geometry(7)
        with NativePotential(self.library, WEIGHTS, threads=2) as native:
            good = native.evaluate(q)
            bad = q.copy(order="F")
            bad[:, 1, 3] = bad[:, 0, 3]
            with self.assertRaisesRegex(PotentialError, "failed bead"):
                native.evaluate(bad)
            for a, b in zip(native.evaluate(q), good):
                np.testing.assert_array_equal(a, b)

    def test_complete_nve_and_thermal_trajectories_match(self):
        from rpmdrate.drpmd_core import AMU, TIME_FS, Dynamics, propagate, sample_momenta, KB
        masses = np.array([1., 1., 16., 16.])*AMU
        q = geometry(7)
        rng = np.random.default_rng(81)
        p = sample_momenta(masses, 7, 1/(KB*300), rng)
        with ExitStack() as stack:
            backends = [stack.enter_context(SerialPotential(self.spec)),
                        stack.enter_context(SharedMemoryPotential(self.spec, 2, q.shape)),
                        stack.enter_context(NativePotential(self.library, WEIGHTS, threads=2))]
            dynamics = [Dynamics(masses, 300., 7, 0.02/TIME_FS, b, native=False) for b in backends]
            trajectories = [propagate(d, q, p, 80) for d in dynamics]
            for result in trajectories[1:]:
                self.assertEqual(result.status, trajectories[0].status)
                np.testing.assert_allclose(result.q, trajectories[0].q, rtol=0, atol=5e-11)
                np.testing.assert_allclose(result.p, trajectories[0].p, rtol=0, atol=5e-10)
            thermal = []
            for d in dynamics:
                qt, pt = q.copy(order="F"), p.copy(order="F")
                d.thermalize(qt, pt, ((0,), (1, 2, 3)), 47, 7, np.random.default_rng(91))
                thermal.append((qt, pt))
            for result in thermal[1:]:
                for a, b in zip(result, thermal[0]):
                    np.testing.assert_allclose(a, b, rtol=0, atol=5e-10)

    def test_nested_shared_and_native_trajectory_workers(self):
        from rpmdrate.drpmd import run, ThermalState
        config = json.loads((ROOT/"examples/drpmd/hhoo_smoke.json").read_text())
        config.update(trajectories=3, thermalization_steps=20, sampling_steps=7,
                      andersen_interval=5, evolution_steps=15, nbeads=7)
        config["potential"]["weights"] = str(WEIGHTS)
        config["potential"]["native_library"] = str(self.library)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            outputs = []
            for backend in ("process", "shared_memory", "native"):
                output = Path(directory)/backend
                summary = run(config, base=ROOT, mode="bead", workers=2, trajectory_workers=2,
                              pes_backend=backend, output=output)
                self.assertEqual(summary["trajectories"], 3)
                self.assertEqual(summary["counts"]["failed"], 0)
                outputs.append(output)
            files = list(outputs[0].glob("chains/chain_*/trajectory_*_final.npz"))
            self.assertEqual(len(files), 3)
            for reference in files:
                relative = reference.relative_to(outputs[0])
                ref = ThermalState.load(reference)
                for output in outputs[1:]:
                    other = ThermalState.load(output/relative)
                    np.testing.assert_allclose(other.q, ref.q, rtol=0, atol=5e-11)
                    np.testing.assert_allclose(other.p, ref.p, rtol=0, atol=5e-10)

    def test_shared_transport_keeps_physical_identity_but_native_library_is_hashed(self):
        from rpmdrate.drpmd_core import _potential_provenance
        config = {"atoms": ["H", "H", "O", "O"],
                  "potential": {"type": "hhoo", "weights": str(WEIGHTS)}}
        original = _potential_provenance(config, ROOT)[1]
        config["potential"].update(cpu_backend="shared_memory", evaluation_timeout=100.)
        self.assertEqual(original, _potential_provenance(config, ROOT)[1])
        config["potential"].update(cpu_backend="native", native_library=str(self.library))
        provenance, native_identity = _potential_provenance(config, ROOT)
        self.assertIn("native_library_sha256", provenance)
        self.assertNotEqual(original, native_identity)


if __name__ == "__main__":
    unittest.main()
