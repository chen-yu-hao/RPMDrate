"""PES decomposition, status propagation and optional Torch gradient checks."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rpmdrate.drpmd_backends import (
    BackendUnavailableError,
    BeadParallelPotential,
    PotentialError,
    PotentialSpec,
    SerialPotential,
    TorchPotential,
)


class SerialPotentialTest(unittest.TestCase):
    def setUp(self):
        self.q = np.random.default_rng(216).normal(size=(3, 4, 7))

    def test_initialization_and_explicit_dimensions(self):
        spec = PotentialSpec(
            "tests.drpmd_potentials", function="get_with_dimensions",
            initializer_kwargs={"spring": 2.75}, with_dimensions=True,
        )
        with SerialPotential(spec) as backend:
            energy, gradient = backend.evaluate(self.q)
        np.testing.assert_allclose(energy, 1.375 * np.sum(self.q**2, axis=(0, 1)))
        np.testing.assert_allclose(gradient, 2.75 * self.q)
        self.assertTrue(gradient.flags.f_contiguous)
        with self.assertRaisesRegex(PotentialError, "closed"):
            backend.evaluate(self.q)

    def test_callback_cannot_modify_caller_coordinates(self):
        original = self.q.copy()
        def destructive_callback(q):
            q[:] = 0.0
            return np.zeros(q.shape[2]), q
        SerialPotential(destructive_callback).evaluate(self.q)
        np.testing.assert_array_equal(self.q, original)

    def test_initializer_status_none_or_zero_is_accepted(self):
        for status in (None, 0, 0.0, np.int64(0)):
            with self.subTest(status=status):
                spec = PotentialSpec(
                    "tests.drpmd_potentials", function="zero_potential",
                    initializer="status_initializer", initializer_kwargs={"status": status},
                )
                with SerialPotential(spec) as backend:
                    energy, gradient = backend.evaluate(self.q)
                np.testing.assert_array_equal(energy, np.zeros(self.q.shape[2]))
                np.testing.assert_array_equal(gradient, np.zeros_like(self.q))

    def test_initializer_nonzero_or_invalid_status_prevents_use(self):
        for status in (1, -7, 2.5, np.nan, np.inf, "failed", [0], 0j):
            with self.subTest(status=status):
                spec = PotentialSpec(
                    "tests.drpmd_potentials", function="zero_potential",
                    initializer="status_initializer", initializer_kwargs={"status": status},
                )
                with self.assertRaisesRegex(PotentialError, "PES initialization"):
                    SerialPotential(spec)

    def test_invalid_pes_results_fail_without_retries(self):
        cases = [
            (np.zeros(7), np.zeros_like(self.q), -1),
            (np.zeros(7), np.zeros_like(self.q), 1),
            (np.zeros(7), np.zeros_like(self.q), np.nan),
            (np.zeros(8), np.zeros_like(self.q)),
            (np.zeros(7), np.zeros((4, 3, 7))),
            (np.full(7, np.inf), np.zeros_like(self.q)),
            (np.zeros(7), np.full_like(self.q, np.nan)),
            (np.zeros(7, dtype=complex), np.zeros_like(self.q)),
        ]
        for result in cases:
            with self.subTest(shapes=[np.shape(value) for value in result]):
                calls = []
                def callback(q):
                    calls.append(1)
                    return result
                with self.assertRaises(PotentialError):
                    SerialPotential(callback).evaluate(self.q)
                self.assertEqual(len(calls), 1)

    def test_bad_coordinates_are_rejected_before_pes(self):
        backend = SerialPotential(lambda q: self.fail("PES must not be called"))
        for q in (np.zeros((3, 4)), np.zeros((3, 0, 1)), np.zeros((3, 1, 0)),
                  np.zeros((3, 4, 7), dtype=complex), np.full((3, 4, 7), np.inf)):
            with self.subTest(shape=q.shape), self.assertRaises(ValueError):
                backend.evaluate(q)


class ParallelPotentialTest(unittest.TestCase):
    def test_serial_parallel_match_for_uneven_chunks_and_single_bead(self):
        spec = PotentialSpec(
            "tests.drpmd_potentials", initializer_kwargs={"spring": 3.125},
            kwargs={"shift": 2.0},
        )
        with SerialPotential(spec) as serial, BeadParallelPotential(spec, 3) as parallel:
            pids = parallel.worker_pids
            self.assertEqual(len(set(pids)), 3)
            self.assertNotIn(os.getpid(), pids)
            for nbeads in (1, 2, 7, 32):
                q = np.random.default_rng(nbeads).normal(size=(3, 4, nbeads))
                expected = serial.evaluate(q)
                actual = parallel.evaluate(q)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                self.assertEqual(parallel.worker_pids, pids)
        with self.assertRaisesRegex(PotentialError, "closed"):
            parallel.evaluate(q)

    def test_workers_initialize_once_and_reuse_pool(self):
        spec = PotentialSpec("tests.drpmd_potentials", function="initialization_count")
        with BeadParallelPotential(spec, 2) as parallel:
            for _ in range(3):
                energy, gradient = parallel.evaluate(np.ones((3, 2, 7)))
                np.testing.assert_array_equal(energy, np.ones(7))
                np.testing.assert_array_equal(gradient, np.zeros((3, 2, 7)))

    def test_bead_calls_actually_run_in_distinct_processes(self):
        spec = PotentialSpec("tests.drpmd_potentials", function="slow_identity")
        with BeadParallelPotential(spec, 3) as parallel:
            energy, _ = parallel.evaluate(np.ones((3, 2, 8)))
            self.assertEqual(set(energy.astype(int)), set(parallel.worker_pids))
            self.assertEqual(len(set(parallel.last_worker_pids)), 3)

    def test_failure_closes_pool_and_preserves_error(self):
        for function in ("failing_potential", "throwing_potential"):
            with self.subTest(function=function):
                backend = BeadParallelPotential(PotentialSpec("tests.drpmd_potentials", function=function), 2)
                with self.assertRaisesRegex(PotentialError, "info=-7|intentional PES failure"):
                    backend.evaluate(np.zeros((3, 2, 5)))
                self.assertTrue(backend.closed)
                backend.close()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchPotentialTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        class Harmonic(torch.nn.Module):
            def forward(self, positions):
                return (0.75 * positions**2).sum(dim=(1, 2))
        self.model = Harmonic()
        self.q = np.random.default_rng(77).normal(size=(3, 4, 9))

    def test_cpu_reference_gradient_and_unit_chain_rule(self):
        length_scale, energy_scale = 0.529177210903, 1.0 / 27.211386245988
        backend = TorchPotential(self.model, device="cpu", length_scale=length_scale, energy_scale=energy_scale)
        with self.torch.no_grad():
            energy, gradient = backend.evaluate(self.q)
        factor = energy_scale * length_scale**2
        np.testing.assert_allclose(energy, 0.75 * factor * np.sum(self.q**2, axis=(0, 1)), rtol=2e-14)
        np.testing.assert_allclose(gradient, 1.5 * factor * self.q, rtol=2e-14)
        index, h = (1, 2, 3), 1e-6
        plus, minus = self.q.copy(), self.q.copy()
        plus[index] += h
        minus[index] -= h
        numerical = (backend.evaluate(plus)[0][3] - backend.evaluate(minus)[0][3]) / (2 * h)
        self.assertAlmostEqual(gradient[index], numerical, delta=2e-10)

    def test_torchscript_path(self):
        trace = self.torch.jit.trace(self.model, self.torch.zeros((2, 4, 3), dtype=self.torch.float64))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "harmonic.pt"
            trace.save(str(path))
            actual = TorchPotential(path, device="cpu").evaluate(self.q)
        expected = TorchPotential(self.model, device="cpu").evaluate(self.q)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_cuda_is_required_when_requested(self):
        if not self.torch.cuda.is_available():
            with self.assertRaisesRegex(BackendUnavailableError, "CUDA"):
                TorchPotential(self.model)
        else:
            actual = TorchPotential(self.model, device="cuda").evaluate(self.q)
            expected = TorchPotential(self.model, device="cpu").evaluate(self.q)
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-13)

    def test_detached_energy_is_an_error(self):
        torch = self.torch
        class Detached(torch.nn.Module):
            def forward(self, positions):
                return (positions**2).sum(dim=(1, 2)).detach()
        with self.assertRaisesRegex(PotentialError, "detached"):
            TorchPotential(Detached(), device="cpu").evaluate(self.q)


if __name__ == "__main__":
    unittest.main()
