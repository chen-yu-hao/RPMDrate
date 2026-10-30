#!/usr/bin/env python3
"""Same native HHOO dynamics, Python vs C product checks; no synthetic delays.

Includes classifier preparation, initial force and early termination checks.
Model loading and preparation of fixed initial states are outside timed regions.
Output comparison uses in-memory CSV formatting, NOT filesystem I/O throughput.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from rpmdrate.drpmd_core import (AMU, TIME_FS, KB, ConnectivityClassifier, Dynamics,
                                ThermalState, sample_momenta, launch_collision, propagate)
from rpmdrate.drpmd_integrator import build_native
from rpmdrate.drpmd_native_pes import NativePotential, build_hhoo


def positive(value):
    value=int(value)
    if value<1: raise argparse.ArgumentTypeError("must be positive")
    return value


def environment():
    def read(path):
        try: return Path(path).read_text().strip()
        except OSError: return None
    cpu=read('/proc/cpuinfo')
    return {"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),
            "cpu_model":next((line for line in (cpu or '').splitlines() if line.startswith('model name')),None),
            "cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            "cgroup_cpu_max":read('/sys/fs/cgroup/cpu.max'),
            "omp_env":{key:os.environ.get(key) for key in ('OMP_DYNAMIC','OMP_WAIT_POLICY','OMP_PROC_BIND','OMP_PLACES')},
            "compiler":subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
            "fortran":subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                'rpmdrate/_drpmd_classifier.c','rpmdrate/_drpmd_engine.c','rpmdrate/drpmd_classifier.py',
                'rpmdrate/drpmd_engine.py','examples/H-H-O-O/pes/hhoo_1014_analytic.f90',
                'examples/H-H-O-O/pes/txt_1014')}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beads',type=positive,nargs='+',default=[32,128])
    parser.add_argument('--threads',type=positive,nargs='+',default=[1,2])
    parser.add_argument('--check-intervals',type=positive,nargs='+',default=[1,10])
    parser.add_argument('--steps',type=positive,default=1000)
    parser.add_argument('--rounds',type=positive,default=3)
    parser.add_argument('--output',type=Path,default=Path('classifier-benchmark.json'))
    args=parser.parse_args(argv)
    report={"format":"RPMDrate.classifier.benchmark.v1", "environment":environment(),
            "steps":args.steps,"rounds":args.rounds,"dt_fs":.05,"block_steps":256,
            "scope":"Fixed HHOO collision; same C dynamics and Fortran PES; only product-check backend differs",
            "notes":["Preparation/DSO compilation and loading excluded; classifier preparation included.",
                     "CSV mode formats centroid/energy rows in StringIO every 100 steps; no filesystem I/O.",
                     "No assertion on speed. Cloud noise, affinity and quota affect results.",
                     "Short numerical/performance regression, not convergence or a rate calculation."],"results":[]}
    config=json.loads((ROOT/'examples/drpmd/hhoo_native.json').read_text())
    masses=np.asarray(config['masses_amu'])*AMU
    reference=ConnectivityClassifier(config['atoms'],config['fragments'],masses,
                                    config['bond_cutoffs_bohr'],config['channels'],config['exit_distance_bohr'])
    rng=np.random.default_rng(9381)
    build_native('gcc')
    with tempfile.TemporaryDirectory() as temporary,threadpool_limits(limits=1,user_api='blas'):
        library=build_hhoo(Path(temporary)/'hhoo.so')
        weights=ROOT/'examples/H-H-O-O/pes/txt_1014'
        for beads in args.beads:
            with NativePotential(library,weights) as preparation_pes:
                preparation=Dynamics(masses,300,beads,.05/TIME_FS,preparation_pes,backend='native')
                q=np.asfortranarray(np.repeat(np.asarray(config['initial_geometry_bohr']).T[:,:,None],beads,axis=2))
                p=sample_momenta(masses,beads,1/(KB*300),np.random.default_rng(81))
                preparation.thermalize(q,p,config['fragments'],200,10,np.random.default_rng(82))
                q,p,_=launch_collision(ThermalState(q,p,300,masses,config['fragments']),6.,20.,np.random.default_rng(83))
                preparation.engine.close()
            for threads in args.threads:
                with NativePotential(library,weights,threads=threads) as pes:
                    for interval in args.check_intervals:
                        for output in ('none','csv_memory_100'):
                            timings={key:[] for key in ('python','native')}
                            counters={}; first={}
                            dynamics={key:Dynamics(masses,300,beads,.05/TIME_FS,pes,threads=threads,
                                                   backend='native',classifier_backend=key) for key in timings}
                            def execute(key):
                                dyn=dynamics[key]
                                trace=[]
                                stream=io.StringIO()
                                writer=csv.writer(stream)
                                def observe(step,coords,momenta,energy):
                                    trace.append(step)
                                    writer.writerow([step,step*.05,energy,*coords.mean(axis=2).T.ravel()])
                                old_calls=dyn.engine.calls; old_checks=dyn.engine.classifier_checks
                                started=time.perf_counter()
                                result=propagate(dyn,q,p,args.steps,reference,interval,3,100,
                                                 observe if output!='none' else None)
                                elapsed=time.perf_counter()-started
                                return elapsed,result,trace,dyn.engine.calls-old_calls,dyn.engine.classifier_checks-old_checks
                            # Warm both paths, and require exactly the same workload/output events.
                            warm={key:execute(key) for key in timings}
                            for repeat in range(args.rounds):
                                for key in rng.permutation(list(timings)):
                                    elapsed,result,trace,calls,checks=execute(key)
                                    timings[key].append(elapsed)
                                    counters[key]={"native_block_calls":calls,"native_checks":checks,
                                                   "result_steps":result.steps,"status":result.status,
                                                   "channel":result.channel,"output_steps":trace}
                                    first.setdefault(key,result)
                            left,right=first['python'],first['native']
                            np.testing.assert_array_equal(left.q,right.q)
                            np.testing.assert_array_equal(left.p,right.p)
                            for key in ('status','channel','result_steps','output_steps'):
                                assert counters['python'][key]==counters['native'][key],key
                            assert counters['native']['native_checks']==right.steps//interval
                            row={"beads":beads,"threads":threads,"check_interval":interval,"output":output,
                                 "raw_seconds":timings,"median_seconds":{k:float(np.median(v)) for k,v in timings.items()},
                                 "counters":counters,"actual_pes_threads":pes.last_threads_used,
                                 "max_q_difference":float(np.max(np.abs(left.q-right.q))),
                                 "max_p_difference":float(np.max(np.abs(left.p-right.p)))}
                            row['speedup']=row['median_seconds']['python']/row['median_seconds']['native']
                            report['results'].append(row)
                            print(f"P={beads} threads={threads} check={interval} output={output}: "
                                  f"Python={row['median_seconds']['python']:.6f}s C={row['median_seconds']['native']:.6f}s "
                                  f"speedup={row['speedup']:.3f} calls={counters['python']['native_block_calls']}->{counters['native']['native_block_calls']}",flush=True)
                            for dyn in dynamics.values(): dyn.engine.close()
        report['environment']['threadpools']=threadpool_info()
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BENCHMARK_JSON='+json.dumps(report,allow_nan=False))
    return 0


if __name__=='__main__': raise SystemExit(main())
