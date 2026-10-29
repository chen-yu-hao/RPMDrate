"""Numerical checks for the exact HHOO NumPy/Torch model port."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from rpmdrate.drpmd_hhoo import HHOOModel, default_weights_path, make_torch_model


def geometries():
    rng = np.random.default_rng(1053)
    # Numerical test configurations, not a claim about the fitted training domain.
    center = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0],
                       [0.0, 3.0, -1.4], [0.0, 3.0, 1.4]])
    xyz = center[None, :, :] + rng.normal(0.0, 0.15, (7, 4, 3))
    return np.asfortranarray(xyz.transpose(2, 1, 0))


class HHOOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = HHOOModel()

    def test_batch_matches_single_geometries(self):
        q = geometries()
        energies, gradients, status = self.model(q)
        self.assertEqual(status, 0)
        self.assertEqual(gradients.shape, q.shape)
        self.assertTrue(gradients.flags.f_contiguous)
        for bead in range(q.shape[2]):
            e, g, status = self.model(q[:, :, bead:bead + 1])
            np.testing.assert_allclose(e, energies[bead:bead + 1], rtol=0, atol=6e-14)
            np.testing.assert_allclose(g[:, :, 0], gradients[:, :, bead], rtol=3e-12, atol=1e-14)

    def test_process_backends_initialize_actual_weights_and_preserve_beads(self):
        from rpmdrate.drpmd_backends import BeadParallelPotential, PotentialSpec, SerialPotential
        spec = PotentialSpec("rpmdrate.drpmd_hhoo", initializer_kwargs={
            "weights_path": str(default_weights_path())})
        q = geometries()
        expected_e, expected_g, _ = self.model(q)
        with SerialPotential(spec) as serial, BeadParallelPotential(spec, workers=2) as parallel:
            self.assertEqual(len(set(parallel.worker_pids)), 2)
            for backend in (serial, parallel):
                energies, gradients = backend.evaluate(q)
                np.testing.assert_allclose(energies, expected_e, rtol=0, atol=6e-14)
                np.testing.assert_allclose(gradients, expected_g, rtol=3e-12, atol=1e-14)

    def test_analytic_gradient_matches_energy_finite_difference(self):
        q = geometries()[:, :, :3]
        _, gradient, _ = self.model(q)
        step = 2.0e-5
        # A fourth-order stencil differentiates only the separately evaluated energy.
        for axis in range(3):
            for atom in range(4):
                values = []
                for multiplier in (-2, -1, 1, 2):
                    displaced = q.copy()
                    displaced[axis, atom, :] += multiplier * step
                    values.append(self.model(displaced)[0])
                numeric = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
                np.testing.assert_allclose(numeric, gradient[axis, atom], rtol=2e-6, atol=3e-9)

    def test_translation_rotation_and_identical_atom_permutations(self):
        q = geometries()
        energies, gradient, _ = self.model(q)
        shifted = q + np.array([2.0, -3.0, 4.0])[:, None, None]
        e, g, _ = self.model(shifted)
        np.testing.assert_allclose(e, energies, rtol=0, atol=6e-14)
        np.testing.assert_allclose(g, gradient, rtol=2e-11, atol=3e-14)
        matrix, _ = np.linalg.qr(np.random.default_rng(32).normal(size=(3, 3)))
        e, g, _ = self.model(np.einsum("ij,jap->iap", matrix, q))
        np.testing.assert_allclose(e, energies, rtol=0, atol=6e-14)
        np.testing.assert_allclose(g, np.einsum("ij,jap->iap", matrix, gradient), rtol=2e-11, atol=3e-14)
        for permutation in ([1, 0, 2, 3], [0, 1, 3, 2], [1, 0, 3, 2]):
            e, g, _ = self.model(q[:, permutation, :])
            np.testing.assert_allclose(e, energies, rtol=0, atol=6e-14)
            np.testing.assert_allclose(g, gradient[:, permutation, :], rtol=2e-11, atol=3e-14)
        np.testing.assert_allclose(gradient.sum(axis=1), 0, rtol=0, atol=3e-17)

    def test_bad_geometry_and_weights_rejected(self):
        q = geometries()
        for bad in (q[:, :3], q[:, :, :0], q.astype(complex), np.full_like(q, np.nan)):
            with self.assertRaises(ValueError):
                self.model(bad)
        q[:, 1, :] = q[:, 0, :]
        with self.assertRaises(ValueError):
            self.model(q)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            contents = default_weights_path().read_text()
            for bad in (contents[:100], contents.replace("3001", "3002", 1)):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    HHOOModel(path)

    def test_reinitialization_rejects_modified_bytes_at_same_weights_path(self):
        from rpmdrate import drpmd_hhoo
        # Restore the previous process-local model even if an assertion fails.
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(drpmd_hhoo, "_potential", None):
            path = Path(directory) / "weights"
            contents = default_weights_path().read_bytes()
            path.write_bytes(contents)
            drpmd_hhoo.initialize_potential(path)
            drpmd_hhoo.initialize_potential(path)
            digest = drpmd_hhoo._potential.weights_sha256
            path.write_bytes(contents + b"\n")
            with self.assertRaisesRegex(ValueError, "weights changed"):
                drpmd_hhoo.initialize_potential(path)
            self.assertEqual(drpmd_hhoo._potential.weights_sha256, digest)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch not installed")
class HHOOTorchTest(unittest.TestCase):
    def test_fp64_energy_and_autograd_match_numpy(self):
        import torch
        from rpmdrate.drpmd_backends import TorchPotential
        model = make_torch_model()
        q = geometries()
        expected_e, expected_g, _ = HHOOModel()(q)
        xyz = torch.tensor(q.transpose(2, 1, 0), dtype=torch.float64, requires_grad=True)
        energies = model(xyz)
        gradients, = torch.autograd.grad(energies.sum(), xyz)
        np.testing.assert_allclose(energies.detach().numpy(), expected_e, rtol=0, atol=8e-14)
        np.testing.assert_allclose(gradients.numpy().transpose(2, 1, 0), expected_g, rtol=3e-10, atol=5e-14)
        self.assertEqual(model.weights_sha256, HHOOModel().weights_sha256)
        with TorchPotential(model, device="cpu") as backend:
            energies, gradients = backend.evaluate(q)
            np.testing.assert_allclose(energies, expected_e, rtol=0, atol=8e-14)
            np.testing.assert_allclose(gradients, expected_g, rtol=3e-10, atol=5e-14)

    def test_bad_torch_input_rejected(self):
        import torch
        model = make_torch_model()
        for xyz in (torch.zeros((1, 4, 3), dtype=torch.float64),
                    torch.ones((1, 3, 3), dtype=torch.float64),
                    torch.ones((1, 4, 3), dtype=torch.float32)):
            with self.assertRaises(ValueError):
                model(xyz)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch not installed")
    def test_cuda_energy_and_gradient_match_numpy_when_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        model = make_torch_model().to("cuda")
        q = geometries()
        expected_e, expected_g, _ = HHOOModel()(q)
        xyz = torch.tensor(q.transpose(2, 1, 0), dtype=torch.float64, device="cuda", requires_grad=True)
        energies = model(xyz)
        gradients, = torch.autograd.grad(energies.sum(), xyz)
        np.testing.assert_allclose(energies.detach().cpu().numpy(), expected_e, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gradients.cpu().numpy().transpose(2, 1, 0), expected_g, rtol=1e-9, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
