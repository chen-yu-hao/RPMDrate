"""Independent comparisons with the repository's original Fortran kernels.

These tests compile the unchanged Fortran sources into a temporary ctypes
library, without f2py, modifying the checkout, or installing anything.  They
skip when a Fortran compiler is unavailable.  Set FC to a compiler command
(including optional flags) to select a nonstandard installation::

    FC=gfortran python -m unittest -v tests.drpmd_fortrantest

The optional PyTorch comparison uses CPU FP64 and does not validate CUDA.
"""

import ctypes
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from rpmdrate.drpmd_hhoo import HHOOModel, default_weights_path, make_torch_model
from rpmdrate.drpmd_integrator import CayleyIntegrator, _NATIVE_PATH


_BRIDGE = r"""
subroutine reference_cayley(natoms,nbeads,m,inv_temperature,step,p,q) bind(c)
    use iso_c_binding
    use system, only: sys_dt=>dt, sys_beta=>beta, sys_mass=>mass, free_ring_polymer_cayley_step
    implicit none
    integer(c_int), value :: natoms,nbeads
    real(c_double), value :: inv_temperature,step
    real(c_double), intent(in) :: m(natoms)
    real(c_double), intent(inout) :: p(3,natoms,nbeads),q(3,natoms,nbeads)
    sys_dt=step
    sys_beta=inv_temperature
    sys_mass(1:natoms)=m
    call free_ring_polymer_cayley_step(p,q,natoms,nbeads)
end subroutine

subroutine reference_hhoo_init(path,path_length,status) bind(c)
    use iso_c_binding
    use hhoo_1014_analytic_model, only: model_initialize
    implicit none
    integer(c_int), value :: path_length
    character(kind=c_char), intent(in) :: path(path_length)
    integer(c_int), intent(out) :: status
    character(len=path_length) :: local_path
    integer :: k
    do k=1,path_length
        local_path(k:k)=path(k)
    end do
    call model_initialize(local_path,status)
end subroutine

subroutine reference_hhoo(nbeads,q,v,g,status) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int), value :: nbeads
    real(c_double), intent(in) :: q(3,4,nbeads)
    real(c_double), intent(out) :: v(nbeads),g(3,4,nbeads)
    integer(c_int), intent(out) :: status
    call get_potential(q,v,g,4,nbeads,status)
end subroutine
"""

_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


def _ptr(array):
    return array.ctypes.data_as(_DOUBLE_PTR)


def _geometries(beads, seed=731):
    # Numerical test configurations, not a statement about the fit's domain.
    center = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0],
                       [0.0, 3.0, -1.4], [0.0, 3.0, 1.4]])
    xyz = center[None] + np.random.default_rng(seed).normal(0, 0.12, (beads, 4, 3))
    return np.asfortranarray(xyz.transpose(2, 1, 0))


class OriginalFortranComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("FC", "gfortran"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("Fortran compiler unavailable; set FC to run original-source comparisons")
        root = Path(__file__).resolve().parent.parent
        cls._temporary = tempfile.TemporaryDirectory(prefix="drpmd-fortran-test-")
        cls.addClassCleanup(cls._temporary.cleanup)
        directory = Path(cls._temporary.name)
        bridge = directory / "bridge.f90"
        bridge.write_text(_BRIDGE)
        target = directory / "reference.so"
        sources = [root / "rpmdrate" / name for name in
                   ("_math.f90", "_surface.f90", "_main.f90", "blas_lapack.f90")]
        sources += [root / "examples/H-H-O-O/pes/hhoo_1014_analytic.f90", bridge]
        command = compiler + ["-O2", "-shared", "-fPIC", "-ffree-line-length-none",
                              "-J" + str(directory)]
        command += [str(source) for source in sources] + ["-o", str(target)]
        result = subprocess.run(command, cwd=directory, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("Original Fortran reference build failed:\n" + result.stdout + result.stderr)
        cls.reference = ctypes.CDLL(str(target))
        cls.reference.reference_cayley.argtypes = [ctypes.c_int, ctypes.c_int, _DOUBLE_PTR,
                                                  ctypes.c_double, ctypes.c_double,
                                                  _DOUBLE_PTR, _DOUBLE_PTR]
        cls.reference.reference_cayley.restype = None
        cls.reference.reference_hhoo_init.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                                     ctypes.POINTER(ctypes.c_int)]
        cls.reference.reference_hhoo_init.restype = None
        cls.reference.reference_hhoo.argtypes = [ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR,
                                                _DOUBLE_PTR, ctypes.POINTER(ctypes.c_int)]
        cls.reference.reference_hhoo.restype = None
        weights = os.fsencode(default_weights_path())
        status = ctypes.c_int(-1)
        cls.reference.reference_hhoo_init(weights, len(weights), ctypes.byref(status))
        if status.value:
            raise RuntimeError("Original HHOO initialization failed: {}".format(status.value))

    def original_step(self, masses, beta, dt, p, q):
        self.reference.reference_cayley(q.shape[1], q.shape[2], _ptr(masses), beta, dt,
                                        _ptr(p), _ptr(q))

    def original_pes(self, q):
        q = np.asfortranarray(q)
        energies, gradient = np.empty(q.shape[2]), np.empty_like(q, order="F")
        status = ctypes.c_int(-1)
        self.reference.reference_hhoo(q.shape[2], _ptr(q), _ptr(energies), _ptr(gradient),
                                      ctypes.byref(status))
        self.assertEqual(status.value, 0)
        return energies, gradient

    def test_cayley_matches_original_after_parameter_changes(self):
        rng = np.random.default_rng(132)
        for native in ([False, True] if _NATIVE_PATH.exists() else [False]):
            for beads, atoms, beta, dt in (
                    (1, 2, 37.0, 0.013), (2, 4, 17.0, 0.13),
                    (3, 1, 29.0, -0.1), (5, 3, 23.0, 0.0),
                    (32, 4, 37.0, 0.13), (32, 4, 37.0, 0.13),
                    (32, 1, 41.0, 0.013), (128, 6, 37.0, -0.013)):
                with self.subTest(native=native, beads=beads, atoms=atoms, beta=beta, dt=dt):
                    masses = rng.uniform(1.0, 32.0, atoms)
                    p = np.asfortranarray(rng.normal(size=(3, atoms, beads)))
                    q = np.asfortranarray(rng.normal(size=(3, atoms, beads)))
                    reference_p, reference_q = p.copy(order="F"), q.copy(order="F")
                    self.original_step(masses, beta, dt, reference_p, reference_q)
                    CayleyIntegrator(masses, beta, beads, dt, native=native).free_step(p, q)
                    np.testing.assert_allclose(p, reference_p, rtol=3e-13, atol=3e-13)
                    np.testing.assert_allclose(q, reference_q, rtol=3e-13, atol=3e-13)

    def test_numpy_hhoo_energy_and_gradient_match_original(self):
        model = HHOOModel()
        for beads in (1, 2, 7, 32):
            q = _geometries(beads)
            expected_e, expected_g = self.original_pes(q)
            energies, gradient, status = model(q)
            self.assertEqual(status, 0)
            np.testing.assert_allclose(energies, expected_e, rtol=0, atol=2e-13)
            np.testing.assert_allclose(gradient, expected_g, rtol=3e-10, atol=8e-14)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch unavailable")
    def test_torch_cpu_hhoo_energy_and_autograd_match_original(self):
        import torch
        q = _geometries(17, seed=299)
        expected_e, expected_g = self.original_pes(q)
        model = make_torch_model()
        xyz = torch.tensor(q.transpose(2, 1, 0), dtype=torch.float64, requires_grad=True)
        energies = model(xyz)
        gradient, = torch.autograd.grad(energies.sum(), xyz)
        np.testing.assert_allclose(energies.detach().numpy(), expected_e, rtol=0, atol=2e-13)
        np.testing.assert_allclose(gradient.numpy().transpose(2, 1, 0), expected_g,
                                   rtol=3e-10, atol=8e-14)

    def test_full_hhoo_nve_trajectory_matches_original(self):
        masses = np.array([1.00784, 1.00784, 15.999, 15.999]) * 1822.888486209
        beads, beta, dt = 32, 1052.6, 0.2
        initial_q = _geometries(beads, seed=197)
        initial_p = np.asfortranarray(np.random.default_rng(672).normal(size=initial_q.shape)
                                     * np.sqrt(masses[None, :, None] * beads / beta))
        reference_p, reference_q = initial_p.copy(order="F"), initial_q.copy(order="F")
        _, reference_g = self.original_pes(reference_q)
        for _ in range(64):
            reference_p -= 0.5 * dt * reference_g
            self.original_step(masses, beta, dt, reference_p, reference_q)
            _, reference_g = self.original_pes(reference_q)
            reference_p -= 0.5 * dt * reference_g
        model = HHOOModel()
        backends = [(False, 1)] + ([(True, 1), (True, 4)] if _NATIVE_PATH.exists() else [])
        for native, threads in backends:
            with self.subTest(native=native, threads=threads):
                p, q = initial_p.copy(order="F"), initial_q.copy(order="F")
                integrator = CayleyIntegrator(masses, beta, beads, dt, threads=threads, native=native)
                _, g, _ = model(q)
                for _ in range(64):
                    integrator.kick(p, g, 0.5 * dt)
                    integrator.free_step(p, q)
                    _, g, _ = model(q)
                    integrator.kick(p, g, 0.5 * dt)
                np.testing.assert_allclose(q, reference_q, rtol=2e-11, atol=2e-12)
                np.testing.assert_allclose(p, reference_p, rtol=2e-11, atol=2e-11)


if __name__ == "__main__":
    unittest.main()
