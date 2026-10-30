"""C product-check semantics and no Python re-entry at check-only boundaries."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import ctypes as ct
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from rpmdrate.drpmd_core import ConnectivityClassifier, Dynamics, propagate, AMU, TIME_FS
from rpmdrate.drpmd_classifier import NativeClassifier
from rpmdrate.drpmd_integrator import build_native
from rpmdrate.drpmd_engine import engine_library
from rpmdrate.drpmd_native_pes import NativePotential, build_hhoo
from rpmdrate.drpmd_backends import PotentialError

ROOT = Path(__file__).resolve().parents[1]
MASS = np.array([1., 2., 16., 18.])
ATOMS = ["H", "H", "O", "O"]
GROUPS = {"reactants": [[0], [1, 2, 3]], "2OH": [[0, 2], [1, 3]],
          "H2+O2": [[0, 1], [2, 3]], "H2O+O": [[0, 1, 2], [3]],
          "H+OH+O": [[0], [1, 2], [3]], "unknown": [[0], [1], [2], [3]],
          "complex": [[0, 1, 2, 3]]}


def classifier(masses=MASS, extra=False):
    channels = {"2OH": ["OH", "HO"], "H2+O2": ["H2", "O2"], "H2O+O": ["H2O", "O"]}
    if extra:
        channels["H+OH+O"] = ["H", "OH", "O"]
    return ConnectivityClassifier(ATOMS, [[0], [1, 2, 3]], masses,
                                  {"H-H": 1.9, "H-O": 2.6, "O-O": 3.2}, channels, 20.)


def state(label, beads=1, masses=MASS, gap=30., direction=1.):
    q = np.zeros((3, 4, beads), dtype=np.float64, order="F")
    p = np.zeros_like(q)
    for k, group in enumerate(GROUPS[label]):
        for local, atom in enumerate(group):
            q[:, atom] = np.array([local*.7, 0., k*gap])[:, None]
            p[2, atom] = direction*k*masses[atom]
    return q, p


@unittest.skipUnless(shutil.which("gcc"), "requires C compiler")
class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if engine_library() is None or not hasattr(engine_library(), "drpmd_classifier_create"):
            build_native("gcc")

    def test_all_channels_unknown_complex_incoming_and_stationary(self):
        for beads in (1, 2, 7, 32, 128):
            reference = classifier(extra=True)
            with NativeClassifier(reference, beads) as native:
                for name in GROUPS:
                    for direction in (-1., 0., 1.):
                        q, p = state(name, beads, direction=direction)
                        expected = name if direction>0 and name not in ("unknown", "complex") else None
                        with self.subTest(beads=beads, name=name, direction=direction):
                            self.assertEqual(reference(q, p), expected)
                            self.assertEqual(native.evaluate(q, p), expected)

    def test_random_rotations_permutations_masses_and_internal_modes(self):
        rng = np.random.default_rng(41)
        names = ["reactants", "2OH", "H2+O2", "H2O+O", "H+OH+O", "unknown"]
        for beads in (2, 7, 32):
            masses = rng.uniform(.7, 22., 4)
            reference = classifier(masses, extra=True)
            with NativeClassifier(reference, beads) as native:
                for j in range(150):
                    q, p = state(names[j%len(names)], beads, masses, gap=rng.uniform(15, 40), direction=rng.choice([-1., 1.]))
                    # Centroid-only semantics: huge internal bead fluctuations
                    # must not turn this into a per-bead/voting classifier.
                    dq, dp = rng.normal(size=(2, *q.shape))*4
                    q += dq-dq.mean(axis=2, keepdims=True)
                    p += dp-dp.mean(axis=2, keepdims=True)
                    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                    q = np.einsum("ij,jab->iab", rotation, q)+rng.normal(size=(3,1,1))
                    p = np.einsum("ij,jab->iab", rotation, p)
                    self.assertEqual(native.evaluate(q, p), reference(q, p))

    def test_cutoff_exit_and_velocity_boundary_semantics(self):
        reference = ConnectivityClassifier(["H","H"], [[0],[1]], [1.,2.], {"H-H":2.}, {}, 4.)
        with NativeClassifier(reference, 1) as native:
            for separation in (2., np.nextafter(4.,0.), 4., np.nextafter(4.,np.inf)):
                for velocity in (-1., 0., np.nextafter(0.,1.), 1.):
                    q=np.zeros((3,2,1),order="F"); p=q.copy(order="F")
                    q[0,1,0]=separation; p[0,1,0]=2*velocity
                    self.assertEqual(native.evaluate(q,p),reference(q,p))
        # A bond EXACTLY at cutoff must be absent (strict '<').
        reference=ConnectivityClassifier(["H"]*3,[[0,1],[2]],[1.]*3,{"H-H":2.},
                                          {"three":["H","H","H"]},3.)
        q=np.zeros((3,3,1),order="F"); p=q.copy(order="F"); q[0,:,0]=[0,2,20]; p[0,2,0]=1
        with NativeClassifier(reference,1) as native:
            self.assertIsNone(native.evaluate(q,p))
            q[0,1,0]=np.nextafter(2.,0.)
            self.assertEqual(native.evaluate(q,p),"reactants")

    def test_every_pair_must_be_outgoing_for_three_fragments(self):
        reference=classifier(extra=True)
        q,p=state("H+OH+O")
        p[2,3]=.5*MASS[3]  # Last two fragments now approach each other.
        with NativeClassifier(reference,1) as native:
            self.assertIsNone(reference(q,p)); self.assertIsNone(native.evaluate(q,p))

    def test_formula_matching_is_unordered_including_duplicate_fragments(self):
        reference=classifier()
        q,p=state("2OH",7)
        with NativeClassifier(reference,7) as native:
            for perm in ([0,1,2,3],[1,0,2,3],[0,1,3,2],[1,0,3,2]):
                # Permute positions and velocities, preserve masses by reweighting p.
                velocity=p/MASS[None,:,None]
                newp=velocity[:,perm,:]*MASS[None,:,None]
                self.assertEqual(native.evaluate(q[:,perm,:],newp),"2OH")

    def test_persistence_resets_on_unknown_and_channel_change(self):
        sequence=["2OH","2OH","unknown","2OH","H2+O2","H2+O2","H2+O2"]
        with NativeClassifier(classifier(),1) as native:
            actual=[native.evaluate(*state(name),persistence=3) for name in sequence]
            self.assertEqual(actual,[None]*6+["H2+O2"])
            self.assertEqual(native.checks,7)

    def test_invalid_closed_and_nonfinite_states_are_rejected(self):
        native=NativeClassifier(classifier(),1)
        q,p=state("2OH")
        with self.assertRaises(ValueError): native.evaluate(q.astype(complex),p)
        with self.assertRaises(ValueError): native.evaluate(q[:,:,0],p)
        with self.assertRaises(PotentialError): native.evaluate(q*np.nan,p)
        native.close(); native.close()
        with self.assertRaises(RuntimeError): native.evaluate(q,p)

    def test_independent_native_workspaces_and_streaks(self):
        def job(label):
            with NativeClassifier(classifier(),32) as native:
                q,p=state(label,32)
                return [native.evaluate(q,p,persistence=3) for _ in range(3)]
        with ThreadPoolExecutor(3) as pool:
            got=list(pool.map(job,["2OH","H2+O2","H2O+O"]))
        self.assertEqual(got,[[None,None,c] for c in ["2OH","H2+O2","H2O+O"]])


@unittest.skipUnless(shutil.which("gcc"), "requires C compiler")
class IntegratedCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ClassifierTest.setUpClass()
        cls.tmp=tempfile.TemporaryDirectory()
        cls.library=Path(cls.tmp.name)/"zero.so"
        src=cls.library.with_suffix(".c")
        src.write_text('''#include <stdint.h>
int32_t drpmd_pes_abi_version(void){return 1;}
int32_t drpmd_pes_omp_enabled(void){return 1;}
int32_t drpmd_pes_init(const char*p){return 0;}
int32_t drpmd_pes_eval(int32_t a,int32_t b,const double*q,double*v,double*g,int32_t t,int32_t*s,int32_t*u){
*u=1; for(int j=0;j<b;j++){v[j]=0;s[j]=0;} for(int j=0;j<3*a*b;j++)g[j]=0;
return 0;}
''')
        subprocess.run(["gcc","-shared","-fPIC",str(src),"-o",str(cls.library)],check=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def dynamics(self,pes,beads=1,block=256,check="auto"):
        return Dynamics(MASS,300,beads,.05,pes,backend="native",block_steps=block,classifier_backend=check)

    def test_no_python_calls_at_product_checks_and_one_block_stops_early(self):
        q,p=state("2OH",7)
        with NativePotential(self.library) as pes:
            dyn=self.dynamics(pes,7)
            with mock.patch.object(ConnectivityClassifier,"__call__",side_effect=AssertionError("Python check")), \
                 mock.patch.object(pes,"evaluate",side_effect=AssertionError("Python PES")):
                result=propagate(dyn,q,p,100,classifier(),check_interval=10,persistence=3)
            self.assertEqual((result.status,result.channel,result.steps),("product","2OH",30))
            self.assertEqual(dyn.engine.calls,1)
            self.assertEqual(dyn.engine.classifier_checks,3)

    def test_chunk_boundaries_output_times_and_streak_reset_between_collisions(self):
        q,p=state("reactants",7)
        for block in (1,6,7,13,256):
            results=[]
            with NativePotential(self.library) as pes:
                for backend in ("python","native"):
                    dyn=self.dynamics(pes,7,block,backend)
                    for repeat in range(2):
                        observed=[]
                        out=propagate(dyn,q,p,100,classifier(),check_interval=7,persistence=3,save_every=5,
                            observer=lambda step,q,p,e: observed.append(step))
                        self.assertEqual((out.status,out.steps),("nonreactive",21))
                        self.assertEqual(observed,[0,5,10,15,20,21])
                        results.append(out)
            for result in results:
                np.testing.assert_array_equal(result.q,results[0].q)
                np.testing.assert_array_equal(result.p,results[0].p)

    def test_no_extra_check_at_end_or_step_zero_and_no_check_boundary_returns(self):
        q,p=state("unknown",7)
        with NativePotential(self.library) as pes:
            dyn=self.dynamics(pes,7)
            result=propagate(dyn,q,p,53,classifier(),check_interval=10,persistence=1)
            self.assertEqual((result.status,result.steps),("unresolved",53))
            self.assertEqual(dyn.engine.calls,1)
            self.assertEqual(dyn.engine.classifier_checks,5)
        q,p=state("2OH")
        with NativePotential(self.library) as pes:
            dyn=self.dynamics(pes)
            self.assertEqual(propagate(dyn,q,p,9,classifier(),10,1).status,"unresolved")
            self.assertEqual(dyn.engine.classifier_checks,0)

    def test_custom_callbacks_and_subclasses_fall_back_without_semantic_changes(self):
        class Custom(ConnectivityClassifier):
            def __call__(self,q,p): return "custom"
        custom=Custom(ATOMS,[[0],[1,2,3]],MASS,{"H-H":1.9,"H-O":2.6,"O-O":3.2},{},20.)
        q,p=state("2OH")
        with NativePotential(self.library) as pes:
            for cb in (custom,lambda q,p:"custom"):
                dyn=self.dynamics(pes)
                out=propagate(dyn,q,p,100,cb,7,3)
                self.assertEqual((out.channel,out.steps),("custom",21))
                self.assertEqual(dyn.last_classifier_backend,"python")
                self.assertEqual(dyn.engine.calls,3)
            with self.assertRaises(ValueError):
                propagate(self.dynamics(pes,check="native"),q,p,100,custom)

    def test_mismatched_masses_and_backend_rejected(self):
        q,p=state("2OH")
        with NativePotential(self.library) as pes:
            with self.assertRaises(ValueError): propagate(self.dynamics(pes),q,p,10,classifier(MASS*2))
            with self.assertRaises(ValueError): self.dynamics(pes,check="bad")
            with self.assertRaises(ValueError): Dynamics(MASS,300,1,.1,pes,backend="python",classifier_backend="native")

    def test_failed_pes_precedes_classification_and_rolls_back(self):
        source=self.library.with_name("failure.c")
        text=self.library.with_suffix(".c").read_text()
        # Fail once the outgoing H crosses x=.35 (after three dt=.1 steps).
        text=text.rsplit("return 0;}",1)[0]+"if(q[0]>.35){s[0]=17;return 17;}return 0;}\n"
        source.write_text(text)
        lib=source.with_suffix(".so")
        subprocess.run(["gcc","-shared","-fPIC",str(source),"-o",str(lib)],check=True)
        q,p=state("reactants");p[0,0,0]=MASS[0]
        out=[]
        with NativePotential(lib) as pes:
            for check in ("python","native"):
                dyn=Dynamics(MASS,300,1,.1,pes,backend="native",classifier_backend=check)
                out.append(propagate(dyn,q,p,100,classifier(),check_interval=1,persistence=10))
        self.assertEqual((out[1].status,out[1].steps),("failed",3))
        np.testing.assert_array_equal(out[0].q,out[1].q)
        np.testing.assert_array_equal(out[0].p,out[1].p)

    def test_dynamic_pool_wires_explicit_native_checks_and_metadata(self):
        from rpmdrate.drpmd import run
        from tests.drpmdtest import toy_config
        config=toy_config()
        config.update(trajectories=4,evolution_steps=100,thermalization_steps=10,check_interval=3)
        config["potential"]={"type":"module","module":"tests.drpmd_potentials","cpu_backend":"native",
                             "native_library":str(self.library)}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            summary=run(config,trajectory_workers=2,classifier_backend="native",output=Path(tmp)/"out")
            self.assertEqual(summary["trajectories"],4)
            self.assertEqual(summary["counts"]["failed"],0)
            self.assertEqual({r["classifier_backend"] for r in summary["worker_reports"] if r["collision_tasks"]},{"native"})
            self.assertGreater(sum(r["native_product_checks"] for r in summary["worker_reports"]),0)


if __name__=="__main__": unittest.main()
