#!/usr/bin/env python3
"""HHOO native-loop and fixed-work strong scaling; not a chemical-rate test.
Same compiled PES, initial states, timesteps and total trajectory count.
Warm measurements exclude startup; raw rounds and startup are also saved.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from rpmdrate.drpmd_core import (AMU, KB, TIME_FS, Dynamics, ThermalState,
    sample_momenta, propagate, launch_collision)
from rpmdrate.drpmd_native_pes import NativePotential, build_hhoo
from rpmdrate.drpmd_engine import engine_library

MASS = np.array([1.00782503223,1.00782503223,15.99491461957,15.99491461957])*AMU
TEMP, DT_FS, FRAGMENTS = 300., 0.05, ((0,), (1,2,3))
_DYN = _STATES = _BARRIER = _LIMITER = None


def dynamics(pes, beads, threads, backend):
    return Dynamics(MASS,TEMP,beads,DT_FS/TIME_FS,pes,threads=threads,
                    backend=backend,block_steps=256)


def _initializer(library, model, beads, threads, states, barrier):
    global _DYN, _STATES, _BARRIER, _LIMITER
    _LIMITER = threadpool_limits(limits=1,user_api="blas")
    _DYN = dynamics(NativePotential(library,model,threads=threads),beads,threads,"native")
    _STATES, _BARRIER = states, barrier


def _ready():
    propagate(_DYN,*_STATES[0],10)
    _BARRIER.wait(timeout=90)
    return os.getpid()


def _task(args):
    index,steps=args
    result=propagate(_DYN,*_STATES[index],steps)
    if result.status=="failed": raise RuntimeError(result.error)
    return result.q,result.p,_DYN.potential.last_threads_used


def output(command):
    try:
        return subprocess.check_output(command,cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): return None


def discrepancy(actual, reference):
    dq,dp=[float(np.max(np.abs(a-b))) for a,b in zip(actual,reference)]
    np.testing.assert_allclose(actual[0],reference[0],rtol=0,atol=2e-9)
    np.testing.assert_allclose(actual[1],reference[1],rtol=0,atol=2e-7)
    return dq,dp


def run_case(dyn,state,steps,phase):
    q,p=[a.copy(order="F") for a in state]
    if phase=="nve":
        result=propagate(dyn,q,p,steps)
        if result.status=="failed": raise RuntimeError(result.error)
        return result.q,result.p
    dyn.thermalize(q,p,FRAGMENTS,steps,10,np.random.default_rng(602),start_step=0)
    return q,p


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads",nargs="+",type=int,default=[1,2,4])
    parser.add_argument("--beads",nargs="+",type=int,default=[32,128])
    parser.add_argument("--steps",type=int,default=1000)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--trajectories",type=int,default=8)
    parser.add_argument("--output",type=Path,default=Path("native-efficiency.json"))
    args=parser.parse_args(argv)
    if min(*args.threads,*args.beads,args.steps,args.rounds,args.trajectories)<1 or 1 not in args.threads:
        parser.error("positive counts required; threads must include 1")
    if args.output.exists(): parser.error("output already exists")
    if engine_library() is None: parser.error("build the new native kernel first")
    library=ROOT/"examples/H-H-O-O/pes/libhhoo_drpmd.so"
    if not library.exists(): build_hhoo(library)
    model=ROOT/"examples/H-H-O-O/pes/txt_1014"
    report={"format":"RPMDrate.native-efficiency.v1","python":sys.version,"numpy":np.__version__,
        "platform":platform.platform(),"git_commit":output(["git","rev-parse","HEAD"]),
        "dirty":bool(output(["git","status","--porcelain"])),"compiler":output(["gfortran","--version"]),
        "steps":args.steps,"rounds":args.rounds,"trajectories":args.trajectories,"dt_fs":DT_FS,
        "initial_thermal_steps":200,"cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "environment":{k:os.environ.get(k) for k in ("OMP_DYNAMIC","OMP_WAIT_POLICY","OMP_PROC_BIND","OMP_PLACES")},
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/"rpmdrate/_drpmd_engine.c",ROOT/"rpmdrate/_drpmd_native.c",ROOT/"rpmdrate/drpmd_engine.py",model)},
        "single":[],"ensemble":[],"notes":[
            "All runs use identical native Fortran HHOO PES; python means old Python timestep loop.",
            "Initial state preparation untimed. Thermal RNG reset identically for every run.",
            "Fixed length NVE, no classifier or file IO. Normal driver retains these features.",
            "Ensemble fixed total N, identical states; warm timing includes task IPC, excludes pool startup.",
            "Speedup=T(1)/T(c); efficiency=speedup/(workers*threads). CPU slots are logical processors.",
            "Short performance benchmark, not rate/thermal convergence or GPU validation."]}
    for name,path in (("cpu_quota","/sys/fs/cgroup/cpu.max"),("cpuinfo","/proc/cpuinfo")):
        if Path(path).exists():
            text=Path(path).read_text()
            report[name]=next((l for l in text.splitlines() if "model name" in l),text.strip()) if name=="cpuinfo" else text.strip()
    chooser=np.random.default_rng(483)
    with threadpool_limits(limits=1,user_api="blas"):
        for beads in args.beads:
            xyz=np.array([[0.,0.,20.],[-.65,1.72,0.],[0.,0.,0.],[2.52,0.,0.]])
            q=np.asfortranarray(np.repeat(xyz.T[:,:,None],beads,axis=2))
            p=sample_momenta(MASS,beads,1/(KB*TEMP),np.random.default_rng(21))
            with NativePotential(library,model) as pes:
                prep=dynamics(pes,beads,1,"native")
                prep.thermalize(q,p,FRAGMENTS,200,10,np.random.default_rng(22))
                prep.engine.close()
            thermal=ThermalState(q,p,TEMP,MASS,FRAGMENTS)
            states=[launch_collision(thermal,6.,20.,np.random.default_rng(200+i))[:2] for i in range(args.trajectories)]
            references,entries={},{}
            with ExitStack() as stack:
                engines={}
                for backend in ("python","native"):
                    for threads in sorted(set(args.threads)):
                        pes=stack.enter_context(NativePotential(library,model,threads=threads))
                        dyn=dynamics(pes,beads,threads,backend)
                        if dyn.engine: stack.callback(dyn.engine.close)
                        engines[(backend,threads)]=dyn
                        for phase in ("nve","thermal"):
                            run_case(dyn,states[0] if phase=="nve" else (q,p),10,phase)
                            entries[(backend,threads,phase)]={"beads":beads,"backend":backend,"threads":threads,
                                "phase":phase,"raw_seconds":[],"max_q_error_bohr":0.,"max_p_error_au":0.}
                for _ in range(args.rounds):
                    keys=list(entries)
                    for index in chooser.permutation(len(keys)):
                        backend,threads,phase=key=keys[index];dyn=engines[(backend,threads)]
                        started=time.perf_counter()
                        result=run_case(dyn,states[0] if phase=="nve" else (q,p),args.steps,phase)
                        elapsed=time.perf_counter()-started
                        row=entries[key];row["raw_seconds"].append(elapsed)
                        if phase not in references:
                            references[phase]=run_case(engines[("python",1)],states[0] if phase=="nve" else (q,p),args.steps,phase)
                        dq,dp=discrepancy(result,references[phase])
                        row["max_q_error_bohr"]=max(row["max_q_error_bohr"],dq)
                        row["max_p_error_au"]=max(row["max_p_error_au"],dp)
                        row["actual_pes_threads"]=dyn.potential.last_threads_used
                for key,row in entries.items():
                    median=statistics.median(row["raw_seconds"])
                    row.update(median_seconds=median,us_per_step=median/args.steps*1e6)
                    baseline=statistics.median(entries[(row["backend"],1,row["phase"])]["raw_seconds"])
                    row["speedup"]=baseline/median;row["parallel_efficiency"]=row["speedup"]/row["threads"]
                    py=statistics.median(entries[("python",row["threads"],row["phase"])]["raw_seconds"])
                    row["speedup_vs_python_same_threads"]=py/median;report["single"].append(row)
                    print(f"P={beads} {key}: {row['us_per_step']:.2f} us/step S={row['speedup']:.3f} E={row['parallel_efficiency']:.3f}",flush=True)
            plan=[(n,1) for n in sorted(set(args.threads))]
            if 4 in args.threads: plan += [(2,2),(1,4)]
            if 2 in args.threads: plan += [(1,2)]
            baseline_states=None;rows=[]
            for index in chooser.permutation(len(plan)):
                workers,threads=plan[index];ctx=mp.get_context("spawn");barrier=ctx.Barrier(workers)
                started=time.perf_counter()
                with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=_initializer,
                    initargs=(str(library),str(model),beads,threads,states,barrier)) as pool:
                    ready=[pool.submit(_ready) for _ in range(workers)]
                    pids=[f.result(timeout=120) for f in ready];startup=time.perf_counter()-started
                    times=[];maxq=maxp=0.
                    for _ in range(args.rounds):
                        started=time.perf_counter()
                        results=list(pool.map(_task,[(i,args.steps) for i in range(args.trajectories)]))
                        times.append(time.perf_counter()-started)
                        if baseline_states is None: baseline_states=results
                        for a,b in zip(results,baseline_states):
                            dq,dp=discrepancy(a[:2],b[:2]);maxq=max(maxq,dq);maxp=max(maxp,dp)
                            if a[2]!=threads: raise RuntimeError("Unexpected OpenMP team size")
                median=statistics.median(times)
                rows.append({"beads":beads,"workers":workers,"threads_per_trajectory":threads,
                    "compute_slots":workers*threads,"worker_pids":pids,"startup_seconds":startup,
                    "raw_seconds":times,"median_seconds":median,"cold_first_round_seconds":startup+times[0],
                    "trajectories_per_second":args.trajectories/median,"max_q_error_bohr":maxq,"max_p_error_au":maxp})
            baseline=next(row["median_seconds"] for row in rows if row["compute_slots"]==1)
            for row in sorted(rows,key=lambda r:(r["compute_slots"],r["workers"])):
                row["speedup"]=baseline/row["median_seconds"];row["parallel_efficiency"]=row["speedup"]/row["compute_slots"]
                report["ensemble"].append(row)
                print(f"P={beads} ensemble {row['workers']}x{row['threads_per_trajectory']}: {row['median_seconds']:.4f}s S={row['speedup']:.3f} E={row['parallel_efficiency']:.3f}",flush=True)
            report["threadpools"]=threadpool_info()
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("Saved",args.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
