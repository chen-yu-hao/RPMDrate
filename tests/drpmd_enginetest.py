"""Native block propagation vs the retained Python-loop reference.

Harmonic cases are numerical controls. Chemical comparisons use the unchanged
HHOO Fortran model and original weights; no new chemical-rate claim is made.
"""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from rpmdrate.drpmd_core import (AMU, KB, TIME_FS, Dynamics, propagate,
                                sample_momenta, ThermalState, run)
from rpmdrate.drpmd_native_pes import NativePotential, build_hhoo
from rpmdrate.drpmd_integrator import build_native
from rpmdrate.drpmd_engine import engine_library
from rpmdrate.drpmd_backends import PotentialError

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("gcc"), "requires gcc/OpenMP")
class NativeBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if engine_library() is None:
            build_native("gcc")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.tmp.name)/"harmonic.so"
        subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-fopenmp",
                        str(ROOT/"examples/native_pes/harmonic.c"), "-o", str(cls.library)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def states(self, beads):
        rng = np.random.default_rng(70)
        q = np.asfortranarray(rng.normal(size=(3, 4, beads)))
        p = sample_momenta(np.array([1., 2., 3., 4.]), beads, 2., rng)
        return q, p

    def dynamics(self, potential, backend, beads=7, block=64, threads=1):
        return Dynamics([1., 2., 3., 4.], 1/(2*KB), beads, 0.01, potential,
                        threads=threads, backend=backend, block_steps=block)

    def test_ring_sizes_threads_and_long_nve_agree(self):
        for beads in (1, 2, 7, 32):
            q, p = self.states(beads)
            with NativePotential(self.library) as pes:
                ref = propagate(self.dynamics(pes, "python", beads), q, p, 513)
            for threads in (1, 2, 4):
                with self.subTest(beads=beads, threads=threads), NativePotential(self.library, threads=threads) as pes:
                    dyn = self.dynamics(pes, "native", beads, threads=threads)
                    got = propagate(dyn, q, p, 513)
                    self.assertEqual(got.status, "unresolved")
                    np.testing.assert_allclose(got.q, ref.q, rtol=0, atol=3e-12)
                    np.testing.assert_allclose(got.p, ref.p, rtol=0, atol=5e-12)
                    self.assertAlmostEqual(got.max_energy_drift_per_bead, ref.max_energy_drift_per_bead, delta=32*np.finfo(float).eps*max(1.,abs(ref.initial_energy_per_bead)))
                    self.assertEqual(pes.last_threads_used, threads)
                    self.assertEqual(dyn.engine.calls, 9)

    def test_full_native_loop_never_calls_python_PES_evaluate(self):
        q, p = self.states(7)
        with NativePotential(self.library) as pes:
            dyn = self.dynamics(pes, "native")
            with mock.patch.object(pes, "evaluate", side_effect=AssertionError("Python PES re-entry")):
                result = propagate(dyn, q, p, 127)
                self.assertEqual(result.steps, 127)
                dyn.thermalize(q, p, ((0, 1), (2, 3)), 127, 7, np.random.default_rng(91))

    def test_thermostat_draws_clock_and_constraints_agree(self):
        for beads in (1, 2, 7, 32):
            q, p = self.states(beads)
            with NativePotential(self.library) as pes:
                left = self.dynamics(pes, "python", beads)
                right = self.dynamics(pes, "native", beads, block=11)
                a, b = q.copy(order="F"), q.copy(order="F")
                c, d = p.copy(order="F"), p.copy(order="F")
                rng1, rng2 = np.random.default_rng(33), np.random.default_rng(33)
                va, ca = left.thermalize(a, c, ((0,), (1, 2, 3)), 117, 7, rng1, 3)
                vb, cb = right.thermalize(b, d, ((0,), (1, 2, 3)), 117, 7, rng2, 3)
                self.assertEqual(ca, cb)
                np.testing.assert_allclose(a, b, rtol=0, atol=3e-12)
                np.testing.assert_allclose(c, d, rtol=0, atol=5e-12)
                np.testing.assert_allclose(va, vb, rtol=0, atol=5e-12)
                np.testing.assert_array_equal(rng1.normal(size=50), rng2.normal(size=50))

    def test_block_length_does_not_change_NVE_or_RNG_draw_order(self):
        q, p = self.states(7)
        with NativePotential(self.library) as pes:
            results, thermal, tails = [], [], []
            for block in (1, 7, 64, 256):
                dyn = self.dynamics(pes, "native", block=block)
                results.append(propagate(dyn, q, p, 134))
                a, b, rng = q.copy(order="F"), p.copy(order="F"), np.random.default_rng(19)
                dyn.thermalize(a, b, ((0, 1), (2, 3)), 134, 11, rng, 7)
                thermal.append((a, b)); tails.append(rng.normal(size=37))
            for result, (a, b), tail in zip(results, thermal, tails):
                np.testing.assert_array_equal(result.q, results[0].q)
                np.testing.assert_array_equal(result.p, results[0].p)
                np.testing.assert_array_equal(a, thermal[0][0])
                np.testing.assert_array_equal(b, thermal[0][1])
                np.testing.assert_array_equal(tail, tails[0])

    def test_observation_classification_and_persistence_events_agree(self):
        q, p = self.states(7)
        outputs = []
        with NativePotential(self.library) as pes:
            for backend in ("python", "native"):
                seen, checks = [], []
                def classify(q, p):
                    checks.append(1)
                    return "reactants" if len(checks)>=2 else None
                result = propagate(self.dynamics(pes, backend), q, p, 101, classify,
                                   check_interval=7, persistence=3, save_every=5,
                                   observer=lambda step, q, p, energy: seen.append((step, energy)))
                outputs.append((result, seen, checks))
            a, b = outputs
            self.assertEqual(a[0].status, "nonreactive")
            self.assertEqual(a[0].steps, b[0].steps)
            self.assertEqual(a[0].steps, 28)
            self.assertEqual(a[2], b[2])
            np.testing.assert_allclose(a[1], b[1], rtol=0, atol=3e-12)

    def test_reversibility_and_concurrent_independent_workspaces(self):
        q, p = self.states(7)
        with NativePotential(self.library) as pes:
            dyn = self.dynamics(pes, "native")
            forward = propagate(dyn, q, p, 400)
            backward = propagate(dyn, forward.q, -forward.p, 400)
            np.testing.assert_allclose(backward.q, q, rtol=0, atol=3e-12)
            np.testing.assert_allclose(backward.p, -p, rtol=0, atol=8e-12)
        def job(scale):
            with NativePotential(self.library) as local:
                return propagate(self.dynamics(local, "native"), q*scale, p, 40)
        with ThreadPoolExecutor(max_workers=3) as pool:
            actual = list(pool.map(job, [1., 2., 3.]))
        for scale, result in zip([1., 2., 3.], actual):
            np.testing.assert_array_equal(result.q, job(scale).q)

    def test_invalid_buffers_closed_PES_and_explicit_backend(self):
        q, p = self.states(7)
        with NativePotential(self.library) as pes:
            dyn = self.dynamics(pes, "native")
            for a, b in ((q, q), (q.astype(np.float32), p), (q.copy(order="C"), p)):
                with self.assertRaises((ValueError, TypeError)):
                    dyn.engine.start(a, b)
            with self.assertRaises(PotentialError):
                dyn.engine.start(np.full_like(q, np.nan), p)
            dyn.engine.close()
            with self.assertRaises(RuntimeError):
                dyn.engine.start(q, p)
            dyn.engine.close()
        with self.assertRaises(RuntimeError):
            Dynamics([1., 2.], 300, 2, .01, lambda q: q, backend="native")
        with self.assertRaises(ValueError):
            self.dynamics(pes, "invalid")

    def test_failed_step_rolls_back_full_state_and_counts(self):
        source = '''#include <stdint.h>
int32_t drpmd_pes_abi_version(void){return 1;}
int32_t drpmd_pes_omp_enabled(void){return 0;}
int32_t drpmd_pes_init(const char *p){return 0;}
int32_t drpmd_pes_eval(int32_t a,int32_t b,const double*q,double*v,double*g,int32_t t,int32_t*s,int32_t*u){
 *u=1; for(int j=0;j<b;j++){v[j]=0;s[j]=0;} for(int j=0;j<3*a*b;j++)g[j]=0;
 if(q[0]>0.35){s[0]=17;return 17;} return 0;}
'''
        path = Path(self.tmp.name)/"failure.c"; path.write_text(source)
        lib = path.with_suffix(".so")
        subprocess.run(["gcc", "-shared", "-fPIC", str(path), "-o", str(lib)], check=True)
        q = np.zeros((3, 2, 1), order="F"); p = q.copy(order="F"); p[0,0,0]=1
        with NativePotential(lib) as pes:
            results=[]
            for backend in ("python", "native"):
                dyn = Dynamics([1., 2.], 300, 1, 0.1, pes, backend=backend)
                results.append(propagate(dyn, q, p, 20))
            for got in results:
                self.assertEqual(got.status, "failed")
                self.assertEqual(got.steps, 3)
                self.assertAlmostEqual(got.q[0,0,0], .3)
            np.testing.assert_array_equal(results[0].q, results[1].q)
            np.testing.assert_array_equal(results[0].p, results[1].p)
            self.assertAlmostEqual(results[0].final_energy_per_bead, results[1].final_energy_per_bead)


@unittest.skipUnless(shutil.which("gfortran"), "requires original Fortran HHOO")
class HHOOBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if engine_library() is None:
            build_native("gcc")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = build_hhoo(Path(cls.tmp.name)/"hhoo.so")
        cls.weights = ROOT/"examples/H-H-O-O/pes/txt_1014"
        cls.config = json.loads((ROOT/"examples/drpmd/hhoo_smoke.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_HHOO_1000_steps_and_thermalization_native_vs_python(self):
        from tests.drpmd_transporttest import geometry
        masses = np.array(self.config["masses_amu"])*AMU
        for beads in (32, 128):
            q = geometry(beads)
            p = sample_momenta(masses, beads, 1/(KB*300), np.random.default_rng(71))
            with NativePotential(self.library, self.weights, threads=2) as pes:
                outputs=[]
                for backend in ("python", "native"):
                    dyn = Dynamics(masses, 300, beads, .02/TIME_FS, pes, threads=2, backend=backend)
                    result=propagate(dyn,q,p,1000)
                    a,b=q.copy(order="F"),p.copy(order="F")
                    values, clock=dyn.thermalize(a,b,((0,),(1,2,3)),273,10,np.random.default_rng(72),7)
                    outputs.append((result,a,b))
                a,b=outputs
                self.assertEqual(a[0].status,"unresolved")
                self.assertEqual(b[0].status,"unresolved")
                np.testing.assert_allclose(a[0].q,b[0].q,rtol=0,atol=1e-10)
                np.testing.assert_allclose(a[0].p,b[0].p,rtol=0,atol=1e-8)
                np.testing.assert_allclose(a[1],b[1],rtol=0,atol=1e-10)
                np.testing.assert_allclose(a[2],b[2],rtol=0,atol=1e-8)

    def test_complete_driver_states_and_output_metadata(self):
        config = dict(self.config, nbeads=7, trajectories=2, evolution_steps=35,
                      thermalization_steps=50, native_block_steps=17)
        config["potential"] = {"type":"hhoo", "weights":str(self.weights),
                               "native_library":str(self.library), "cpu_backend":"native"}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out=[]
            for backend in ("python", "native"):
                directory=Path(tmp)/backend
                summary=run(dict(config, dynamics_backend=backend), output=directory)
                self.assertEqual(summary["dynamics_backend"], backend)
                if backend=="native": self.assertGreater(summary["native_block_calls"],0)
                out.append(directory)
            for name in ("thermal_state.npz", "trajectory_000001_final.npz", "trajectory_000002_final.npz"):
                a,b=[ThermalState.load(d/name) for d in out]
                np.testing.assert_allclose(a.q,b.q,rtol=0,atol=1e-10)
                np.testing.assert_allclose(a.p,b.p,rtol=0,atol=1e-8)


if __name__ == "__main__":
    unittest.main()
