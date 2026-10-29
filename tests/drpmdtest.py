"""Physical sampling, scattering and end-to-end direct RPMD regressions.

The stochastic checks use solvable ring-polymer distributions and fixed seeds.
Their tolerances include Monte Carlo uncertainty; they do not certify a
production thermalization length or timestep for a chemical potential.
"""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from rpmdrate.drpmd import (
    AMU, KB, RATE_TO_CM3_S, TIME_FS,
    CentroidCOMConstraints, ConnectivityClassifier, Dynamics, ThermalState,
    fragment_com, fragment_velocity, launch_collision, main, propagate,
    run, sample_collision, sample_momenta, summarize,
)
from rpmdrate.drpmd_integrator import _load_native


class HarmonicPotential:
    def __init__(self, force_constants):
        self.constants = np.asarray(force_constants, dtype=float)[None, :, None]

    def evaluate(self, q):
        return ((0.5 * self.constants * q * q).sum(axis=(0, 1)),
                np.asfortranarray(self.constants * q))


def toy_config():
    """Dimensionless two-free-atom case, with all units explicit at the boundary."""
    return {
        "atoms": ["H", "H"], "masses_amu": [1.0 / AMU, 2.0 / AMU],
        "fragments": [[0], [1]], "temperature_K": 1.0 / (2.0 * KB),
        "nbeads": 4, "dt_fs": 0.025 * TIME_FS,
        "initial_geometry_bohr": [[0, 0, 0], [0, 0, 5]],
        "thermalization_steps": 50, "sampling_steps": 13, "andersen_interval": 7,
        "trajectories": 2, "evolution_steps": 350,
        "initial_separation_bohr": 5.0, "bmax_bohr": 1.0, "seed": 812,
        "bond_cutoffs_bohr": {"H-H": 1.0}, "channels": {},
        "exit_distance_bohr": 4.0, "check_interval": 5, "persistence": 2,
        "save_every": 37,
        "potential": {"type": "module", "module": "tests.drpmd_potentials",
                      "initializer_kwargs": {"spring": 0.0}},
    }


class ThermalSamplingTest(unittest.TestCase):
    def test_andersen_momenta_have_bead_temperature_and_centroid_physical_temperature(self):
        masses, beta, beads = np.array([1.0, 3.0]), 2.5, 8
        rng = np.random.default_rng(582)
        samples = np.array([sample_momenta(masses, beads, beta, rng) for _ in range(5000)])
        scaled = samples / np.sqrt(masses[None, None, :, None] * beads / beta)
        np.testing.assert_allclose(np.mean(scaled * scaled, axis=(0, 1, 3)), 1.0, atol=0.025, rtol=0)
        centroid = samples.mean(axis=3) / np.sqrt(masses[None, None, :] / beta)
        np.testing.assert_allclose(np.mean(centroid * centroid, axis=(0, 1)), 1.0, atol=0.04, rtol=0)

    def test_constraints_leave_all_noncentroid_modes_and_internal_relative_motion(self):
        masses = np.array([1., 2., 3., 4.])
        fragments = ((0, 1), (2, 3))
        rng = np.random.default_rng(340)
        q = np.asfortranarray(rng.normal(size=(3, 4, 7)))
        p = np.asfortranarray(rng.normal(size=q.shape))
        constraints = CentroidCOMConstraints(q, masses, fragments)
        q += rng.normal(size=(3, 4, 1))
        q_fluctuations = q - q.mean(axis=2, keepdims=True)
        p_fluctuations = p - p.mean(axis=2, keepdims=True)
        relative_q = [q[:, a] - q[:, b] for a, b in fragments]
        relative_velocity = [p[:, a] / masses[a] - p[:, b] / masses[b] for a, b in fragments]
        constraints.positions(q)
        constraints.momenta(p)
        for f, target, dq, dv in zip(fragments, constraints.targets, relative_q, relative_velocity):
            np.testing.assert_allclose(fragment_com(q, masses, f), target, rtol=0, atol=2e-15)
            np.testing.assert_allclose(fragment_velocity(p, masses, f), 0, rtol=0, atol=2e-15)
            np.testing.assert_allclose(q[:, f[0]] - q[:, f[1]], dq, atol=1e-15, rtol=0)
            np.testing.assert_allclose(p[:, f[0]] / masses[f[0]] - p[:, f[1]] / masses[f[1]], dv, atol=1e-15, rtol=0)
        np.testing.assert_allclose(q - q.mean(axis=2, keepdims=True), q_fluctuations, atol=1e-15, rtol=0)
        np.testing.assert_allclose(p - p.mean(axis=2, keepdims=True), p_fluctuations, atol=1e-15, rtol=0)

    def test_free_ring_thermalization_samples_configuration_and_momentum_distributions(self):
        masses, beta, beads = np.array([1., 3.]), 2.0, 4
        dynamics = Dynamics(masses, 1 / (KB * beta), beads, 0.08,
                            HarmonicPotential([0., 0.]), native=False)
        rng = np.random.default_rng(9126)
        q = np.zeros((3, 2, beads), dtype=float, order="F")
        q[:, 1, :] = np.array([0., 0., 5.])[:, None]
        p = sample_momenta(masses, beads, beta, rng)
        fragments = ((0,), (1,))
        _, clock = dynamics.thermalize(q, p, fragments, 500, 7, rng)
        q_samples, p_samples = [], []
        for _ in range(1000):
            _, clock = dynamics.thermalize(q, p, fragments, 19, 7, rng, clock)
            q_samples.append(np.fft.rfft(q, axis=2, norm="ortho")[:, :, 1:])
            p_samples.append(np.fft.rfft(p, axis=2, norm="ortho")[:, :, 1:])
        omega = 2 * beads / beta * np.sin(np.pi * np.arange(1, beads // 2 + 1) / beads)
        expected_q = beads / (beta * masses[:, None] * omega[None, :] ** 2)
        expected_p = beads * masses[:, None] / beta
        q_ratio = np.mean(np.abs(q_samples) ** 2, axis=(0, 1)) / expected_q
        p_ratio = np.mean(np.abs(p_samples) ** 2, axis=(0, 1)) / expected_p
        np.testing.assert_allclose(q_ratio, 1, atol=0.10, rtol=0)
        np.testing.assert_allclose(p_ratio, 1, atol=0.08, rtol=0)
        np.testing.assert_allclose(q.mean(axis=2), [[0, 0], [0, 0], [0, 5]], atol=1e-13, rtol=0)
        np.testing.assert_allclose(p.mean(axis=2), 0, atol=3e-13, rtol=0)

    def test_harmonic_thermalization_retains_and_samples_vibrational_centroid_modes(self):
        # Disjoint atom-pair differences cancel the fragment COM constraint,
        # providing 48 independent Cartesian oscillators, including mode k=0.
        beta, beads, physical_omega = 2., 4, 1.0
        masses = np.repeat([1., 3.], 16)
        fragments = (tuple(range(16)), tuple(range(16, 32)))
        dynamics = Dynamics(masses, 1 / (KB * beta), beads, 0.08,
                            HarmonicPotential(masses * physical_omega ** 2), native=False)
        rng = np.random.default_rng(5514)
        q = np.zeros((3, 32, beads), dtype=float, order="F")
        p = sample_momenta(masses, beads, beta, rng)
        _, clock = dynamics.thermalize(q, p, fragments, 600, 7, rng)
        q_samples, p_samples = [], []
        for _ in range(450):
            _, clock = dynamics.thermalize(q, p, fragments, 23, 7, rng, clock)
            q_modes = np.fft.rfft((q[:, 0::2] - q[:, 1::2]) / math.sqrt(2), axis=2, norm="ortho")
            p_modes = np.fft.rfft((p[:, 0::2] - p[:, 1::2]) / math.sqrt(2), axis=2, norm="ortho")
            q_samples.append(q_modes)
            p_samples.append(p_modes)
        omega2 = physical_omega ** 2 + (2 * beads / beta * np.sin(np.pi * np.arange(3) / beads)) ** 2
        expected_q = beads / (beta * masses[0::2, None] * omega2[None, :])
        expected_p = beads * masses[0::2, None] / beta
        q_ratio = np.mean(np.abs(q_samples) ** 2 / expected_q[None, None, :, :], axis=(0, 1, 2))
        p_ratio = np.mean(np.abs(p_samples) ** 2 / expected_p[None, None, :, :], axis=(0, 1, 2))
        np.testing.assert_allclose(q_ratio, 1, atol=0.08, rtol=0)
        np.testing.assert_allclose(p_ratio, 1, atol=0.06, rtol=0)

    def test_thermostat_clock_is_preserved_across_sampling_blocks(self):
        rng1, rng2 = np.random.default_rng(221), np.random.default_rng(221)
        masses = np.array([1., 2.])
        q1 = np.zeros((3, 2, 4), dtype=float, order="F")
        p1 = sample_momenta(masses, 4, 2., rng1)
        p2 = sample_momenta(masses, 4, 2., rng2)
        q2 = q1.copy(order="F")
        dynamics = Dynamics(masses, 1 / (KB * 2), 4, 0.03, HarmonicPotential([0., 0.]))
        _, clock = dynamics.thermalize(q1, p1, ((0,), (1,)), 23, 7, rng1)
        _, clock2 = dynamics.thermalize(q2, p2, ((0,), (1,)), 9, 7, rng2)
        _, clock2 = dynamics.thermalize(q2, p2, ((0,), (1,)), 14, 7, rng2, clock2)
        self.assertEqual(clock, clock2)
        np.testing.assert_allclose(q1, q2, atol=1e-14, rtol=0)
        np.testing.assert_allclose(p1, p2, atol=1e-14, rtol=0)


class CollisionSamplingTest(unittest.TestCase):
    def test_flux_energy_and_disk_impact_distributions(self):
        rng = np.random.default_rng(718)
        beta, mu, bmax = 2.5, 3.2, 4.0
        samples = [sample_collision(beta, mu, bmax, 8, rng) for _ in range(20000)]
        energy = np.array([s["collision_energy_hartree"] for s in samples])
        impact = np.array([s["impact_parameter_bohr"] for s in samples])
        phi = np.array([s["azimuth_rad"] for s in samples])
        speed = np.array([s["speed_au"] for s in samples])
        self.assertAlmostEqual(energy.mean(), 2 / beta, delta=0.018)
        self.assertAlmostEqual(energy.var(), 2 / beta ** 2, delta=0.025)
        self.assertAlmostEqual(np.mean(impact ** 2), bmax ** 2 / 2, delta=0.12)
        self.assertAlmostEqual(np.cos(phi).mean(), 0, delta=0.02)
        self.assertAlmostEqual(np.sin(phi).mean(), 0, delta=0.02)
        np.testing.assert_allclose(0.5 * mu * speed ** 2, energy, rtol=3e-16, atol=3e-16)

    def test_launch_retains_internal_modes_and_sets_correct_centroid_collision(self):
        rng = np.random.default_rng(33)
        masses, fragments = np.array([1., 2., 5., 7.]), ((0, 1), (2, 3))
        q = np.asfortranarray(rng.normal(size=(3, 4, 5)))
        p = sample_momenta(masses, 5, 2., rng)
        state = ThermalState(q, p, 1 / (KB * 2.), masses, fragments)
        qnew, pnew, sample = launch_collision(state, 2., 8., rng, randomize_orientation=False)
        np.testing.assert_array_equal(state.q, q)
        np.testing.assert_array_equal(state.p, p)
        coms = [fragment_com(qnew, masses, f) for f in fragments]
        velocities = [fragment_velocity(pnew, masses, f) for f in fragments]
        fragment_masses = np.array([3., 12.])
        np.testing.assert_allclose(sum(m * c for m, c in zip(fragment_masses, coms)), 0, atol=2e-14)
        np.testing.assert_allclose(pnew.mean(axis=2).sum(axis=1), 0, atol=2e-14)
        displacement, velocity = coms[1] - coms[0], velocities[1] - velocities[0]
        self.assertAlmostEqual(np.linalg.norm(displacement), 8., places=13)
        self.assertAlmostEqual(np.linalg.norm(displacement[:2]), sample["impact_parameter_bohr"], places=13)
        np.testing.assert_allclose(velocity, [0, 0, -sample["speed_au"]], atol=3e-15)
        mu = np.prod(fragment_masses) / fragment_masses.sum()
        self.assertAlmostEqual(0.5 * mu * np.dot(velocity, velocity), sample["collision_energy_hartree"], places=13)
        for fragment in fragments:
            before = q[:, list(fragment)] - fragment_com(q, masses, fragment)[:, None, None]
            after = qnew[:, list(fragment)] - fragment_com(qnew, masses, fragment)[:, None, None]
            np.testing.assert_allclose(after, before, atol=2e-15)
            before = p[:, list(fragment)] / masses[None, list(fragment), None] - fragment_velocity(p, masses, fragment)[:, None, None]
            after = pnew[:, list(fragment)] / masses[None, list(fragment), None] - fragment_velocity(pnew, masses, fragment)[:, None, None]
            np.testing.assert_allclose(after, before, atol=2e-15)

    def test_rotations_are_isotropic_and_preserve_internal_geometry_and_energy(self):
        masses = np.ones(4)
        fragments = ((0, 1), (2, 3))
        q = np.zeros((3, 4, 1), dtype=float)
        q[0, :, 0] = [-0.5, 0.5, -1., 1.]
        p = np.zeros_like(q)
        p[1, :, 0] = [-0.7, 0.7, -0.9, 0.9]
        state = ThermalState(q, p, 1 / (KB * 2.), masses, fragments)
        rng = np.random.default_rng(461)
        orientations = []
        for _ in range(2000):
            qnew, pnew, _ = launch_collision(state, 1., 6., rng)
            orientation = qnew[:, 1, 0] - qnew[:, 0, 0]
            orientations.append(orientation)
            self.assertAlmostEqual(np.linalg.norm(orientation), 1., places=13)
            internal = pnew[:, [0, 1], :] - fragment_velocity(pnew, masses, fragments[0])[:, None, None]
            self.assertAlmostEqual(0.5 * np.sum(internal ** 2), 0.49, places=13)
        orientations = np.asarray(orientations)
        np.testing.assert_allclose(orientations.mean(axis=0), 0, atol=0.04, rtol=0)
        np.testing.assert_allclose(orientations.T @ orientations / len(orientations), np.eye(3) / 3, atol=0.025, rtol=0)


class ProductionDynamicsTest(unittest.TestCase):
    def test_unthermostatted_production_is_reversible_with_bounded_nve_energy(self):
        rng = np.random.default_rng(961)
        masses, beads = np.array([1., 3.]), 7
        potential = HarmonicPotential(0.36 * masses)
        dynamics = Dynamics(masses, 1 / (KB * 2), beads, 0.025, potential)
        q = np.asfortranarray(rng.normal(0, 0.2, (3, 2, beads)))
        p = np.asfortranarray(rng.normal(0, 0.4, q.shape))
        q0, p0 = q.copy(), p.copy()
        with mock.patch.object(dynamics, "thermalize", side_effect=AssertionError("Production must not thermostat")):
            forward = propagate(dynamics, q, p, 500)
            backward = propagate(dynamics, forward.q, -forward.p, 500)
        self.assertEqual(forward.status, "unresolved")
        self.assertEqual(backward.status, "unresolved")
        np.testing.assert_array_equal(q, q0)
        np.testing.assert_array_equal(p, p0)
        self.assertLess(forward.max_energy_drift_per_bead / forward.initial_energy_per_bead, 3e-4)
        np.testing.assert_allclose(backward.q, q0, atol=3e-12, rtol=0)
        np.testing.assert_allclose(backward.p, -p0, atol=1e-11, rtol=0)

    def test_classification_requires_persistent_checks_and_timeout_remains_unresolved(self):
        dynamics = Dynamics([1., 1.], 1 / KB, 1, 0.01, HarmonicPotential([0., 0.]))
        q = np.zeros((3, 2, 1))
        p = np.zeros_like(q)
        labels = iter(["products", None, "products", "products"])
        result = propagate(dynamics, q, p, 20, lambda q, p: next(labels), check_interval=2, persistence=2)
        self.assertEqual((result.status, result.channel, result.steps), ("product", "products", 8))
        result = propagate(dynamics, q, p, 5, lambda q, p: None, check_interval=1, persistence=2)
        self.assertEqual((result.status, result.channel, result.steps), ("unresolved", None, 5))
        result = propagate(dynamics, q, p, 5, lambda q, p: "reactants", check_interval=1, persistence=2)
        self.assertEqual((result.status, result.channel, result.steps), ("nonreactive", None, 2))

    def test_pes_failure_retains_last_complete_state_and_is_not_reclassified(self):
        class FailAfterOneStep(HarmonicPotential):
            def __init__(self):
                super().__init__([0., 0.])
                self.calls = 0
            def evaluate(self, q):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("PES failed deliberately")
                return super().evaluate(q)
        q = np.ones((3, 2, 1))
        p = np.full_like(q, 0.2)
        dynamics = Dynamics([1., 2.], 1 / KB, 1, 0.1, FailAfterOneStep())
        result = propagate(dynamics, q, p, 5)
        self.assertEqual((result.status, result.steps, result.channel), ("failed", 1, None))
        np.testing.assert_allclose(result.q, q + 0.1 * p / np.array([1., 2.])[None, :, None])
        np.testing.assert_array_equal(result.p, p)
        self.assertIn("PES failed deliberately", result.error)


class ClassificationAndRatesTest(unittest.TestCase):
    def setUp(self):
        self.masses = np.array([1., 1., 16., 16.])
        self.classifier = ConnectivityClassifier(
            ["H", "H", "O", "O"], ((0,), (1, 2, 3)), self.masses,
            {"H-H": 1.6, "H-O": 2.4, "O-O": 3.0},
            {"2OH": ["HO", "HO"], "H2+O2": ["H2", "O2"], "H2O+O": ["H2O", "O"]},
            exit_distance=8.,
        )

    def test_centroid_connectivity_requires_separated_outgoing_products(self):
        q = np.array([[0., 0., 0.], [10., 0., 0.], [0., 0., 1.8], [10., 0., 1.8]]).T[:, :, None]
        velocity = np.array([-0.2, 0.2, -0.2, 0.2])
        p = np.zeros_like(q)
        p[0, :, 0] = self.masses * velocity
        self.assertEqual(self.classifier(q, p), "2OH")
        self.assertIsNone(self.classifier(q, -p))
        self.assertIsNone(self.classifier(q, np.zeros_like(p)))
        q[:, [1, 3], :] -= np.array([6., 0, 0])[:, None, None]
        self.assertIsNone(self.classifier(q, p))

    def test_other_hhoo_product_stoichiometries(self):
        q = np.array([[0., 0., 0.], [0., 0., 1.4], [12., 0., 0.], [12., 0., 2.4]]).T[:, :, None]
        p = np.zeros_like(q)
        p[0, :, 0] = self.masses * np.array([-0.2, -0.2, 0.2, 0.2])
        self.assertEqual(self.classifier(q, p), "H2+O2")
        q = np.array([[0., 0., 1.8], [1.8, 0., 0.], [0., 0., 0.], [12., 0., 0.]]).T[:, :, None]
        p[0, :, 0] = self.masses * np.array([-0.2, -0.2, -0.2, 0.2])
        self.assertEqual(self.classifier(q, p), "H2O+O")

    def test_thermal_connectivity_requires_configured_atom_partition(self):
        q = np.array([[10., 0., 0.], [0., 0., 1.8], [0., 0., 0.], [0., 2.4, 0.]]).T[:, :, None]
        self.classifier.validate_reactants(q)
        # Swapping which hydrogen is free preserves stoichiometry but not the
        # declared reactant atom partition whose centroid COMs were constrained.
        with self.assertRaisesRegex(ValueError, "connectivity"):
            self.classifier.validate_reactants(q[:, [1, 0, 2, 3], :])
        q[:, 3, :] += 20.
        with self.assertRaisesRegex(ValueError, "connectivity"):
            self.classifier.validate_reactants(q)

    def test_rates_keep_unresolved_and_failed_outcomes_in_denominator_and_bounds(self):
        records = [{"status": "product", "channel": "A"}, {"status": "nonreactive"},
                   {"status": "unresolved"}, {"status": "failed"}]
        summary = summarize(records, ["A", "B"], 300., [1., 3.], ((0,), (1,)), 2., 0.25)
        mu = 0.75
        prefactor = 0.25 * math.pi * 4 * math.sqrt(8 * KB * 300 / (math.pi * mu)) * RATE_TO_CM3_S
        self.assertAlmostEqual(summary["collision_prefactor_cm3_molecule_s"], prefactor)
        channel = summary["channels"]["A"]
        self.assertIsNone(channel["rate_cm3_molecule_s"])
        self.assertIsNone(channel["iid_wilson95_cm3_molecule_s"])
        self.assertEqual(channel["observed_lower_cm3_molecule_s"], prefactor / 4)
        self.assertEqual(channel["possible_upper_cm3_molecule_s"], 3 * prefactor / 4)
        self.assertEqual(summary["channels"]["B"]["possible_upper_cm3_molecule_s"], prefactor / 2)
        self.assertFalse(summary["all_trajectories_resolved"])
        self.assertEqual(summary["total_reaction"]["count"], 1)

    def test_resolved_limits_and_unknown_labels(self):
        kwargs = dict(channels=["A"], temperature=300., masses=[1., 2.], fragments=((0,), (1,)), bmax=2.)
        for status, probability in (("product", 1.), ("nonreactive", 0.)):
            summary = summarize([{"status": status, "channel": "A"}] * 8, **kwargs)
            estimate = summary["total_reaction"]
            self.assertEqual(estimate["rate_cm3_molecule_s"], probability * summary["collision_prefactor_cm3_molecule_s"])
            self.assertTrue(summary["all_trajectories_resolved"])
            self.assertIsNotNone(estimate["iid_wilson95_cm3_molecule_s"])
        with self.assertRaises(ValueError):
            summarize([{"status": "product", "channel": "unconfigured"}], **kwargs)
        with self.assertRaises(ValueError):
            summarize([{"status": "timeout"}], **kwargs)


class StateAndDriverTest(unittest.TestCase):
    def test_full_bead_state_roundtrip_and_no_overwrite(self):
        rng = np.random.default_rng(761)
        q, p = rng.normal(size=(2, 3, 4, 7))
        state = ThermalState(q, p, 300., [1., 2., 3., 4.], ((0,), (1, 2, 3)),
                             {"phase": "thermal", "potential_identity": "test-only", "thermal_steps": 91})
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "state.npz"
            state.save(filename)
            restored = ThermalState.load(filename)
            np.testing.assert_array_equal(restored.q, q)
            np.testing.assert_array_equal(restored.p, p)
            self.assertTrue(restored.q.flags.f_contiguous and restored.p.flags.f_contiguous)
            self.assertEqual(restored.fragments, state.fragments)
            self.assertEqual(restored.metadata["thermal_steps"], 91)
            with self.assertRaises(FileExistsError):
                state.save(filename)

    def test_invalid_complex_states_are_rejected_instead_of_discarding_imaginary_part(self):
        with self.assertRaises(ValueError):
            ThermalState(np.ones((3, 2, 4), complex) * (1 + 1j), np.ones((3, 2, 4)),
                         300., [1., 2.], ((0,), (1,)))

    def test_cli_errors_for_unknown_keys_and_invalid_root_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            output = Path(directory) / "output"
            for invalid in (dict(toy_config(), termalization_steps=50), [], None):
                path.write_text(json.dumps(invalid))
                with contextlib.redirect_stderr(io.StringIO()) as captured:
                    code = main([str(path), "--output", str(output)])
                self.assertEqual(code, 2)
                self.assertIn("dRPMD error", captured.getvalue())
                self.assertFalse(output.exists())

    def test_configuration_rejects_invalid_modes_and_unthermalized_new_states(self):
        config = toy_config()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for mode, workers in (("unknown", 1), ("serial", 2), ("bead", 1), ("gpu", 2)):
                with self.assertRaises(ValueError):
                    run(config, mode=mode, workers=workers, output=output)
            config["thermalization_steps"] = 0
            with self.assertRaises(ValueError):
                run(config, output=output)
            self.assertFalse(output.exists())

    def test_driver_reuses_first_complete_thermal_state_without_thermalizing_again(self):
        config = toy_config()
        config["trajectories"] = 1
        config["evolution_steps"] = 3
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first, second = Path(directory) / "first", Path(directory) / "second"
            run(config, output=first)
            thermal_path = first / "thermal_state.npz"
            with mock.patch.object(Dynamics, "thermalize", side_effect=AssertionError("Loaded first state must be used directly")):
                run(config, output=second, thermal_state=thermal_path)
            left, right = ThermalState.load(thermal_path), ThermalState.load(second / "thermal_state.npz")
            np.testing.assert_array_equal(left.q, right.q)
            np.testing.assert_array_equal(left.p, right.p)
            for key, value in (("phase", "collision_final"), ("thermal_separation_bohr", 99.),
                               ("potential_identity", "wrong-model")):
                invalid = ThermalState.load(thermal_path)
                invalid.metadata[key] = value
                invalid_file = Path(directory) / (key + ".npz")
                invalid.save(invalid_file)
                invalid_output = Path(directory) / (key + "-output")
                with self.assertRaisesRegex(ValueError, "matching"):
                    run(config, output=invalid_output, thermal_state=invalid_file)
                self.assertFalse(invalid_output.exists())
            config["temperature_K"] *= 2
            with self.assertRaisesRegex(ValueError, "matching"):
                run(config, output=Path(directory) / "third", thermal_state=thermal_path)

    def test_model_provenance_tracks_content_and_survives_model_path_changes(self):
        from rpmdrate.drpmd import _potential_provenance
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first, second = directory / "first.dat", directory / "second.dat"
            first.write_bytes(b"stand-in model bytes: provenance hashing only")
            second.write_bytes(first.read_bytes())
            config = {"atoms": ["H", "H", "O", "O"],
                      "potential": {"type": "hhoo", "weights": "first.dat"}}
            _, initial = _potential_provenance(config, directory)
            config["potential"]["weights"] = "second.dat"
            _, renamed = _potential_provenance(config, directory)
            self.assertEqual(initial, renamed)
            second.write_bytes(b"modified stand-in model bytes")
            _, modified = _potential_provenance(config, directory)
            self.assertNotEqual(initial, modified)

    @unittest.skipUnless(_load_native() is not None, "bead dynamics require built native OpenMP kernel")
    def test_complete_serial_and_bead_drivers_agree_with_same_seed(self):
        config = toy_config()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first, second = Path(directory) / "serial", Path(directory) / "bead"
            serial = run(config, mode="serial", output=first)
            bead = run(config, mode="bead", workers=2, output=second)
            self.assertEqual(serial["counts"], bead["counts"])
            self.assertEqual(serial["total_reaction"], bead["total_reaction"])
            for trajectory in (1, 2):
                for phase in ("initial", "final"):
                    filename = "trajectory_%06d_%s.npz" % (trajectory, phase)
                    left, right = ThermalState.load(first / filename), ThermalState.load(second / filename)
                    np.testing.assert_allclose(left.q, right.q, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(left.p, right.p, rtol=0, atol=1e-12)


class InputIntegrityTest(unittest.TestCase):
    def test_product_formula_order_is_normalized(self):
        classifier = ConnectivityClassifier(
            ["H", "H", "O", "O"], [[0], [1, 2, 3]], [1, 1, 16, 16],
            {"H-H": 1.9, "H-O": 2.6, "O-O": 3.2},
            {"2OH": ["OH", "OH"]}, 10.0,
        )
        self.assertEqual(classifier.channels, {("HO", "HO"): "2OH"})

    def test_product_formula_must_conserve_atoms_and_be_unique(self):
        for channels in ({"bad": ["H2", "O"]},
                         {"a": ["OH", "OH"], "b": ["HO", "HO"]},
                         {"bad": ["H0O", "H2O"]},
                         {"bad": ["(OH)", "OH"]}):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                ConnectivityClassifier(
                    ["H", "H", "O", "O"], [[0], [1, 2, 3]], [1, 1, 16, 16],
                    {"H-H": 1.9, "H-O": 2.6, "O-O": 3.2}, channels, 10.0,
                )

    def test_saved_metadata_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(path, q=np.zeros((3, 2, 1)), p=np.zeros((3, 2, 1)),
                     masses=np.ones(2), metadata=np.array("[]"))
            with self.assertRaisesRegex(ValueError, "metadata"):
                ThermalState.load(path)


if __name__ == "__main__":
    unittest.main()
