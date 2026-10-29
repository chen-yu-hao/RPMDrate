"""Scientific regressions for the independent direct-RPMD Cayley propagator."""
import concurrent.futures
import unittest

import numpy as np

from rpmdrate.drpmd_integrator import CayleyIntegrator, _NATIVE_PATH


class CayleyIntegratorTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8841)

    def state(self, atoms, beads):
        return tuple(np.asfortranarray(self.rng.normal(size=(3, atoms, beads))) for _ in range(2))

    def dense_reference(self, p, q, masses, beta, dt):
        beads = p.shape[-1]
        # Count both ring edges explicitly: two beads must have doubled edges.
        laplacian = 2.0 * np.eye(beads)
        for k in range(beads):
            laplacian[k, (k - 1) % beads] -= 1.0
            laplacian[k, (k + 1) % beads] -= 1.0
        omega2 = (beads / beta) ** 2
        matrix = np.eye(beads) + 0.25 * dt ** 2 * omega2 * laplacian
        out_p, out_q = np.empty_like(p), np.empty_like(q)
        for atom, mass in enumerate(masses):
            for coordinate in range(3):
                mid = np.linalg.solve(matrix, q[coordinate, atom] + 0.5 * dt * p[coordinate, atom] / mass)
                out_q[coordinate, atom] = 2 * mid - q[coordinate, atom]
                out_p[coordinate, atom] = p[coordinate, atom] - dt * mass * omega2 * laplacian.dot(mid)
        return out_p, out_q

    def test_dense_reference_all_ring_sizes_and_step_signs(self):
        backends = [False, True] if _NATIVE_PATH.exists() else [False]
        for native in backends:
            for beads in (1, 2, 3, 5, 8, 32):
                for dt in (0.0, -0.12, 0.12):
                    with self.subTest(native=native, beads=beads, dt=dt):
                        masses = np.array([1.0, 16.0])
                        p, q = self.state(2, beads)
                        expected_p, expected_q = self.dense_reference(p, q, masses, 37.0, dt)
                        integrator = CayleyIntegrator(masses, 37.0, beads, dt, native=native)
                        integrator.free_step(p, q)
                        np.testing.assert_allclose(p, expected_p, rtol=3e-13, atol=3e-13)
                        np.testing.assert_allclose(q, expected_q, rtol=3e-13, atol=3e-13)

    def test_free_ring_energy_centroid_and_reversal(self):
        for native in ([False, True] if _NATIVE_PATH.exists() else [False]):
            for beads in (1, 2, 5, 32):
                masses = np.array([1.0, 16.0])
                beta, dt = 37.0, 0.013
                integrator = CayleyIntegrator(masses, beta, beads, dt, native=native)
                p, q = self.state(2, beads)
                p0, q0 = p.copy(), q.copy()

                def energy():
                    kinetic = np.sum(p ** 2 / (2 * masses[None, :, None]))
                    spring = 0.5 * (beads / beta) ** 2 * np.sum(
                        masses[None, :, None] * (q - np.roll(q, 1, axis=2)) ** 2)
                    return kinetic + spring

                start_energy = energy()
                for _ in range(200):
                    integrator.free_step(p, q)
                np.testing.assert_allclose(energy(), start_energy, rtol=3e-11, atol=3e-11)
                np.testing.assert_allclose(q.mean(axis=2), q0.mean(axis=2) +
                                          200 * dt * (p0 / masses[None, :, None]).mean(axis=2),
                                          rtol=3e-11, atol=3e-11)
                p *= -1
                for _ in range(200):
                    integrator.free_step(p, q)
                np.testing.assert_allclose(q, q0, rtol=3e-10, atol=3e-10)
                np.testing.assert_allclose(p, -p0, rtol=3e-10, atol=3e-10)

    @unittest.skipUnless(_NATIVE_PATH.exists(), "build the optional C/OpenMP kernel for parallel checks")
    def test_parallel_native_matches_serial_and_numpy(self):
        for beads in (1, 2, 3, 7, 32):
            masses = [1.0, 2.0, 16.0, 17.0]
            initial_p, initial_q = self.state(4, beads)
            trajectories = []
            for native, threads in ((False, 1), (True, 1), (True, 4)):
                integrator = CayleyIntegrator(masses, 20.0, beads, 0.015, threads=threads, native=native)
                p, q = initial_p.copy(order="F"), initial_q.copy(order="F")
                for _ in range(50):
                    # Complete kick-Cayley-kick trajectory on a harmonic PES.
                    integrator.kick(p, q, 0.0075)
                    integrator.free_step(p, q)
                    integrator.kick(p, q, 0.0075)
                trajectories.append((p, q))
            for p, q in trajectories[1:]:
                np.testing.assert_allclose(p, trajectories[0][0], rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(q, trajectories[0][1], rtol=2e-12, atol=2e-12)

    def test_distinct_object_cache_isolation(self):
        specs = [([1.0], 17.0, 3, 0.03), ([2.0, 16.0], 41.0, 32, 0.01),
                 ([3.0, 8.0, 17.0], 25.0, 2, -0.02)]
        originals = [self.state(len(spec[0]), spec[2]) for spec in specs]

        def run(index, native):
            integrator = CayleyIntegrator(*specs[index], native=native)
            p, q = (a.copy(order="F") for a in originals[index])
            for _ in range(80):
                integrator.free_step(p, q)
            return p, q

        expected = [run(i, False) for i in range(len(specs))]
        for native in ([False, True] if _NATIVE_PATH.exists() else [False]):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                actual = list(pool.map(lambda index: run(index, native), range(len(specs))))
            for result, reference in zip(actual, expected):
                np.testing.assert_allclose(result[0], reference[0], rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(result[1], reference[1], rtol=2e-12, atol=2e-12)

    def test_gradient_kick_sign_and_input_guards(self):
        integrator = CayleyIntegrator([1.0], 10.0, 3, 0.01, native=False)
        p, q = self.state(1, 3)
        expected = p - 0.1 * q
        integrator.kick(p, q, 0.1)
        np.testing.assert_array_equal(p, expected)
        with self.assertRaises(ValueError):
            integrator.kick(p, p, 0.1)
        with self.assertRaises(TypeError):
            integrator.free_step(p.astype(np.float32), q)
        with self.assertRaises(ValueError):
            integrator.free_step(p, p)
        with self.assertRaises(ValueError):
            integrator.free_step(np.zeros((3, 1, 3)), q)
        q[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            integrator.free_step(p, q)
        with self.assertRaises(ValueError):
            CayleyIntegrator([0.0], 10.0, 3, 0.01)
        with self.assertRaises(ValueError):
            CayleyIntegrator([1.0], 0.0, 3, 0.01)
        with self.assertRaises(ValueError):
            CayleyIntegrator([1.0], 10.0, 3.5, 0.01)
        with self.assertRaises(RuntimeError):
            CayleyIntegrator([1.0], 10.0, 3, 0.01, threads=2, native=False)


if __name__ == "__main__":
    unittest.main()
