import unittest

import numpy

from rpmdrate import _main


class CayleyFreeRingTest(unittest.TestCase):

    def setUp(self):
        self.system = _main.system
        self.system.beta = 37.0
        self.system.dt = 0.013
        self.masses = numpy.array([1.0, 16.0])
        self.system.mass[:2] = self.masses

    def _laplacian(self, nbeads):
        laplacian = 2.0 * numpy.eye(nbeads)
        if nbeads == 1:
            return numpy.zeros((1, 1))
        laplacian -= numpy.eye(nbeads, k=1)
        laplacian -= numpy.eye(nbeads, k=-1)
        laplacian[0, -1] -= 1.0
        laplacian[-1, 0] -= 1.0
        return laplacian

    def _explicit_step(self, p, q):
        nbeads = q.shape[2]
        laplacian = self._laplacian(nbeads)
        omega = nbeads / self.system.beta
        alpha = (self.system.dt * omega) ** 2 / 4.0
        matrix = numpy.eye(nbeads) + alpha * laplacian
        rhs_matrix = numpy.eye(nbeads) - alpha * laplacian
        p_expected = numpy.empty_like(p)
        q_expected = numpy.empty_like(q)
        for i in range(3):
            for j, mass in enumerate(self.masses):
                velocity = p[i, j] / mass
                q_expected[i, j] = numpy.linalg.solve(
                    matrix, rhs_matrix.dot(q[i, j]) + self.system.dt * velocity
                )
                velocity -= 0.5 * self.system.dt * omega ** 2 * laplacian.dot(
                    q[i, j] + q_expected[i, j]
                )
                p_expected[i, j] = mass * velocity
        return p_expected, q_expected

    def test_matches_dense_cyclic_cayley(self):
        rng = numpy.random.default_rng(20261029)
        for nbeads in (1, 2, 3, 4, 5, 32):
            q = numpy.asfortranarray(rng.normal(size=(3, 2, nbeads)))
            p = numpy.asfortranarray(rng.normal(size=(3, 2, nbeads)))
            p_expected, q_expected = self._explicit_step(p.copy(), q.copy())
            self.system.free_ring_polymer_cayley_step(p, q)
            numpy.testing.assert_allclose(q, q_expected, rtol=2e-13, atol=2e-13)
            numpy.testing.assert_allclose(p, p_expected, rtol=2e-13, atol=2e-13)

    def test_free_ring_energy_and_centroid(self):
        rng = numpy.random.default_rng(11)
        nbeads = 32
        q = numpy.asfortranarray(rng.normal(size=(3, 2, nbeads)))
        p = numpy.asfortranarray(rng.normal(size=(3, 2, nbeads)))
        laplacian = self._laplacian(nbeads)
        omega = nbeads / self.system.beta

        def energy(momentum, position):
            value = 0.0
            for i in range(3):
                for j, mass in enumerate(self.masses):
                    value += 0.5 * numpy.dot(momentum[i, j], momentum[i, j]) / mass
                    value += 0.5 * mass * omega ** 2 * numpy.dot(
                        position[i, j], laplacian.dot(position[i, j])
                    )
            return value

        q0 = q.copy()
        p0 = p.copy()
        e0 = energy(p0, q0)
        centroid0 = q0.mean(axis=2)
        velocity_centroid = (p0 / self.masses[None, :, None]).mean(axis=2)
        self.system.free_ring_polymer_cayley_step(p, q)
        numpy.testing.assert_allclose(energy(p, q), e0, rtol=2e-13, atol=2e-13)
        numpy.testing.assert_allclose(q.mean(axis=2), centroid0 + self.system.dt * velocity_centroid,
                                      rtol=2e-13, atol=2e-13)

    def test_reuses_work_after_atom_mass_and_parameter_changes(self):
        rng = numpy.random.default_rng(231)
        for nbeads, natoms, beta, step in (
            (32, 4, 37.0, 0.013),
            (32, 4, 37.0, 0.013),  # Same shape/parameters, different masses.
            (32, 1, 37.0, 0.013),
            (32, 6, 37.0, 0.013),
            (32, 1, 73.0, -0.2),
            (5, 3, 41.0, 0.0),
            (2, 4, 37.0, 0.013),
            (1, 2, 37.0, 0.013),
            (32, 4, 37.0, 0.013),
        ):
            with self.subTest(nbeads=nbeads, natoms=natoms, beta=beta, step=step):
                self.masses = rng.uniform(1.0, 32.0, natoms)
                self.system.mass[:natoms] = self.masses
                self.system.beta = beta
                self.system.dt = step
                p = numpy.asfortranarray(rng.normal(size=(3, natoms, nbeads)))
                q = numpy.asfortranarray(rng.normal(size=(3, natoms, nbeads)))
                p_expected, q_expected = self._explicit_step(p.copy(), q.copy())
                self.system.free_ring_polymer_cayley_step(p, q)
                numpy.testing.assert_allclose(p, p_expected, rtol=2e-13, atol=2e-13)
                numpy.testing.assert_allclose(q, q_expected, rtol=2e-13, atol=2e-13)

    def test_time_reversal(self):
        rng = numpy.random.default_rng(771)
        for nbeads in (2, 3, 32):
            with self.subTest(nbeads=nbeads):
                p = numpy.asfortranarray(rng.normal(size=(3, 2, nbeads)))
                q = numpy.asfortranarray(rng.normal(size=(3, 2, nbeads)))
                p0, q0 = p.copy(), q.copy()
                for _ in range(200):
                    self.system.free_ring_polymer_cayley_step(p, q)
                p *= -1
                for _ in range(200):
                    self.system.free_ring_polymer_cayley_step(p, q)
                numpy.testing.assert_allclose(q, q0, rtol=2e-10, atol=2e-10)
                numpy.testing.assert_allclose(p, -p0, rtol=2e-10, atol=2e-10)


if __name__ == '__main__':
    unittest.main()
