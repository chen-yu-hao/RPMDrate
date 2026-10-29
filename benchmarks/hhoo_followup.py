"""Resume 122 HHOO unknown outcomes and replay two 2OH candidates.

No rethermalization, momentum resampling, PES modification or channel override.
Uses the unchanged original core from the parent commit.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from rpmdrate import drpmd_core as core

SOURCE_RUN = 40909399592
SOURCE_SHA = "60ea1c30672f67b36dc7c070bf0a787c4be4d117"
MODEL_SHA = "08da0b1646a98191f41e5bd4c2330f2c57a0fb16e19b6592df77408e6dd61e2b"
IDENTITY = "e4993d882b91b54572a1d46df864e3e0a3cd376b82ee4a0ae379edeef6399796"
EXPECTED = {"hhoo-b6-s0":15,"hhoo-b6-s1":18,"hhoo-b6-s2":23,"hhoo-b6-s3":26,
            "hhoo-b10-s0":13,"hhoo-b10-s1":6,"hhoo-b10-s2":13,"hhoo-b10-s3":8}
PAIRS = ((0,1),(2,3),(0,2),(0,3),(1,2),(1,3))
ROOT = Path(__file__).resolve().parents[1]


def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_file(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Invalid/missing source path: {relative}")
    return path


def configured(manifest):
    cfg=dict(manifest["input"])
    weights=ROOT/"examples/H-H-O-O/pes/txt_1014"
    if digest(weights)!=MODEL_SHA:
        raise ValueError("Model differs from original experiment")
    cfg["potential"]={"type":"hhoo","weights":str(weights)}
    _,identity=core._potential_provenance(cfg,ROOT)
    if identity!=IDENTITY or manifest["potential_identity"]!=identity:
        raise ValueError("Potential identity mismatch")
    return cfg


def load_source(source):
    """Validate all original denominators, IDs, parameters and checkpoint bytes."""
    source=Path(source).resolve()
    records,tasks,products,configs=[],[],[],{}
    for artifact,expected in EXPECTED.items():
        folder=source/artifact
        manifest=json.loads((folder/"input.json").read_text())
        cfg=configured(manifest);configs[artifact]=cfg
        if cfg["nbeads"]!=32 or cfg["dt_fs"]!=0.1 or cfg["evolution_steps"]!=200000:
            raise ValueError("Not original 32-bead / 20ps experiment")
        if cfg["check_interval"]!=10 or cfg["persistence"]!=3:
            raise ValueError("Unexpected original exit criteria")
        with (folder/"trajectories.csv").open(newline="") as f:
            rows=list(csv.DictReader(f))
        if sorted(int(r["trajectory"]) for r in rows)!=list(range(1,65)):
            raise ValueError(f"Incomplete/duplicated originals: {artifact}")
        if sum(r["status"]=="unresolved" for r in rows)!=expected:
            raise ValueError(f"Changed unresolved ledger: {artifact}")
        for row in rows:
            if row["status"] not in ("product","nonreactive","unresolved"):
                raise ValueError("Unexpected original outcome")
            record={**row,"artifact":artifact,"key":f"{artifact}-t{int(row['trajectory']):03d}"}
            records.append(record)
            if row["status"] in ("product","unresolved"):
                initial=safe_file(folder,row["initial_state"])
                final=safe_file(folder,row["final_state"])
                begin,end=core.ThermalState.load(initial),core.ThermalState.load(final)
                for state,phase in ((begin,"collision_initial"),(end,"collision_final")):
                    if (state.metadata.get("phase")!=phase or state.nbeads!=32 or
                        state.temperature!=cfg["temperature_K"] or
                        state.metadata.get("potential_identity")!=IDENTITY or
                        state.fragments!=tuple(tuple(x) for x in cfg["fragments"]) or
                        not np.array_equal(state.masses,np.asarray(cfg["masses_amu"])*core.AMU)):
                        raise ValueError(f"Saved state mismatch: {record['key']}")
                if end.metadata.get("steps")!=int(row["steps"]) or end.metadata.get("status")!=row["status"]:
                    raise ValueError("Saved state/table disagreement")
                item={"key":record["key"],"artifact":artifact,"record":row,
                      "initial_sha256":digest(initial),"final_sha256":digest(final)}
                if row["status"]=="unresolved":
                    if int(row["steps"])!=200000:raise ValueError("Unknown source did not reach 20ps")
                    tasks.append(item)
                else:products.append(item)
    if len(records)!=512 or len(tasks)!=122:raise ValueError("Incorrect source denominators")
    if {(p["artifact"],p["record"]["trajectory"],p["record"]["channel"]) for p in products}!={
            ("hhoo-b6-s3","8","2OH"),("hhoo-b6-s3","51","2OH")}:
        raise ValueError("Original 2OH identities changed")
    return records,tasks,products,configs


def plan(source,out):
    records,tasks,products,_=load_source(source)
    dump(Path(out)/"plan.json",{"source_run":SOURCE_RUN,"source_sha":SOURCE_SHA,
         "unresolved_count":len(tasks),"original_records":records,"continuations":tasks,
         "audit_candidates":products,"parts":16,"target_total_ps":200,"additional_ps":180,
         "boundary_note":"Original checkpoints lack persistence history. Reset only exit counter at 20ps; q/p unchanged. New checkpoints preserve counter, gradient, energy and step."})
    print("SOURCE_VALIDATED",json.dumps({"records":len(records),"resume":len(tasks),
          "2OH":len(products),"per_artifact":EXPECTED}),flush=True)


def checkpoint(path,q,p,gradient,values,state):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,q=q,p=p,gradient=gradient,values=values,
                            metadata=np.array(json.dumps(state,allow_nan=False)))
    os.replace(tmp,path)


def trace_row(q,p,values,masses,beta,step,dt_fs,energy,e0):
    centroid=q.mean(axis=2)
    distances=[float(np.linalg.norm(centroid[:,a]-centroid[:,b])) for a,b in PAIRS]
    c0=core.fragment_com(q,masses,(0,));c1=core.fragment_com(q,masses,(1,2,3))
    kinetic=float(np.sum(p*p/(2*masses[None,:,None]))/q.shape[2])
    spring=float(0.5*np.sum(masses[None,:,None]*(q.shape[2]/beta)**2 *
                 (q-np.roll(q,1,axis=2))**2)/q.shape[2])
    return [step,step*dt_fs,energy,energy-e0,float(values.mean()),kinetic,spring,
            *distances,float(np.linalg.norm(c0-c1)),*centroid.T.ravel()]


def integrate(dynamics,q,p,*,start,target,e0,prior_drift,classifier,
              check_every,persistence,save_every,out,metadata,restart=None):
    """Original kick-Cayley-kick loop, with absolute clock and energy history."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dt_fs=dynamics.dt*core.TIME_FS
    if not 0<=start<target:raise ValueError("Invalid continuation range")
    q,p=np.array(q,dtype=float,order="F",copy=True),np.array(p,dtype=float,order="F",copy=True)
    values,gradient=dynamics.potential.evaluate(q)
    energy=core.ring_energy(q,p,dynamics.masses,dynamics.beta,values)/dynamics.nbeads
    segment_e0=energy;candidate,streak=None,0
    drift,local_drift=max(prior_drift,abs(energy-e0)),0.
    step,status,channel,error=start,"unresolved",None,None
    last_checked=start-start%check_every
    if restart is not None:
        with np.load(restart,allow_pickle=False) as data:
            saved=json.loads(data["metadata"].item())
            for key in ("key","dt_fs","target_step","initial_state_sha256","potential_identity"):
                if saved[key]!=metadata[key]:raise ValueError(f"Restart mismatch: {key}")
            if saved["status"]!="running":raise ValueError("Only running checkpoints can resume")
            q,p=data["q"].copy(order="F"),data["p"].copy(order="F")
            gradient,values=data["gradient"].copy(order="F"),data["values"].copy()
        step=int(saved["absolute_step"])
        if not start<=step<target:raise ValueError("Checkpoint outside requested interval")
        energy,e0,segment_e0=saved["energy"],saved["e0"],saved["segment_e0"]
        drift,local_drift=saved["max_drift"],saved["segment_max_drift"]
        candidate,streak,last_checked=saved["candidate"],saved["streak"],saved["last_checked"]
    epoch_start=step
    cp_interval=max(1,round((1000. if metadata.get("kind")=="audit" else 5000.)/dt_fs))
    started=time.perf_counter();last_cp=started
    trace=out/f"trace_from_{epoch_start:09d}.csv"
    headers=["step","time_fs","ring_energy_per_bead_hartree","delta_E_hartree",
             "potential_per_bead_hartree","kinetic_per_bead_hartree","spring_per_bead_hartree",
             "H0_H1_bohr","O2_O3_bohr","H0_O2_bohr","H0_O3_bohr","H1_O2_bohr","H1_O3_bohr","H0_HO2_COM_bohr"]
    headers += [f"{a}{i}_{axis}_bohr" for i,a in enumerate(("H","H","O","O")) for axis in "xyz"]
    def snapshot(state_status):
        return {**metadata,"absolute_step":step,"time_fs":step*dt_fs,"e0":e0,"energy":energy,
                "segment_e0":segment_e0,"max_drift":drift,"segment_max_drift":local_drift,
                "candidate":candidate,"streak":streak,"last_checked":last_checked,
                "status":state_status,"channel":channel,"error":error}
    checkpoint(out/"checkpoint.npz",q,p,gradient,values,snapshot("running"))
    with trace.open("x",newline="") as f:
        writer=csv.writer(f);writer.writerow(headers)
        writer.writerow(trace_row(q,p,values,dynamics.masses,dynamics.beta,step,dt_fs,energy,e0))
        f.flush();last_written=step
        for next_step in range(step+1,target+1):
            oldq,oldp=q.copy(order="F"),p.copy(order="F")
            try:
                new_values,new_gradient=dynamics.step(p,q,gradient)
                new_energy=core.ring_energy(q,p,dynamics.masses,dynamics.beta,new_values)/dynamics.nbeads
                if not math.isfinite(new_energy):raise FloatingPointError("Nonfinite ring Hamiltonian")
            except Exception as exc:
                q,p=oldq,oldp;status,error="failed",f"{type(exc).__name__}: {exc}"
                break
            values,gradient,energy=new_values,new_gradient,new_energy;step=next_step
            drift=max(drift,abs(energy-e0));local_drift=max(local_drift,abs(energy-segment_e0))
            label=None
            if step%check_every==0:
                label=classifier(q,p) if classifier else None
                streak=streak+1 if label is not None and label==candidate else (1 if label else 0)
                candidate,last_checked=label,step
            resolved=label is not None and streak>=persistence
            if step%save_every==0 or step==target or resolved:
                writer.writerow(trace_row(q,p,values,dynamics.masses,dynamics.beta,step,dt_fs,energy,e0))
                last_written=step
            now=time.perf_counter()
            if step%cp_interval==0 or now-last_cp>=120:
                f.flush();checkpoint(out/"checkpoint.npz",q,p,gradient,values,snapshot("running"))
                dump(out/"progress.json",{"key":metadata["key"],"time_ps":step*dt_fs/1000,
                     "max_energy_drift_hartree":drift,"elapsed_seconds":now-started})
                print("PROGRESS",metadata["key"],f"{step*dt_fs/1000:.4f} ps",flush=True)
                last_cp=now
            if resolved:
                status="nonreactive" if label=="reactants" else "product"
                channel=None if label=="reactants" else label
                break
        if last_written!=step:
            writer.writerow(trace_row(q,p,values,dynamics.masses,dynamics.beta,step,dt_fs,energy,e0))
    result={**snapshot(status),"start_step":start,"time_ps":step*dt_fs/1000,
            "max_energy_drift_per_bead_hartree":drift,"segment_max_energy_drift_per_bead_hartree":local_drift,
            "final_signed_drift_hartree":energy-e0,"compute_seconds_this_invocation":time.perf_counter()-started}
    checkpoint(out/"final.npz",q,p,gradient,values,result);dump(out/"result.json",result)
    return result


def execute(task):
    """One-CPU worker: no thermostat or collision-launch call."""
    out=Path(task["out"]);out.mkdir(parents=True,exist_ok=True)
    try:
        row=task["record"];folder=Path(task["source"])/task["artifact"]
        initial_path=safe_file(folder,row["initial_state"]);final_path=safe_file(folder,row["final_state"])
        if digest(initial_path)!=task["initial_sha256"] or digest(final_path)!=task["final_sha256"]:
            raise ValueError("Source checkpoint bytes changed")
        manifest=json.loads((folder/"input.json").read_text());cfg=configured(manifest)
        initial=core.ThermalState.load(initial_path)
        state=initial if task["kind"]=="audit" else core.ThermalState.load(final_path)
        dt=task["dt_fs"]
        start=0 if task["kind"]=="audit" else round(int(row["steps"])*cfg["dt_fs"]/dt)
        target=round(task["target_ps"]*1000/dt)
        check=round(cfg["check_interval"]*cfg["dt_fs"]/dt)
        save=round((1. if task["kind"]=="audit" else 10.)/dt)
        meta={"key":task["key"],"kind":task["kind"],"artifact":task["artifact"],
              "trajectory":int(row["trajectory"]),"chain_id":int(row["chain_id"]),"dt_fs":dt,
              "nbeads":32,"temperature_K":state.temperature,"target_step":target,
              "target_total_ps":task["target_ps"],"potential_identity":IDENTITY,"model_sha256":MODEL_SHA,
              "source_run":SOURCE_RUN,"source_sha":SOURCE_SHA,
              "source_checkpoint_sha256":digest(initial_path if task["kind"]=="audit" else final_path),
              "initial_state_sha256":digest(initial_path),"original_record":row,
              "check_interval_fs":check*dt,"persistence":cfg["persistence"],
              "boundary_persistence_reset":task["kind"]=="continuation"}
        with threadpool_limits(limits=1),core._make_potential(cfg,ROOT,"serial",1) as potential:
            dynamics=core.Dynamics(state.masses,state.temperature,32,dt/core.TIME_FS,potential,threads=1,native=True)
            original_values,_=potential.evaluate(initial.q)
            e0=core.ring_energy(initial.q,initial.p,state.masses,dynamics.beta,original_values)/32
            if abs(e0-float(row["initial_energy_per_bead_hartree"]))>1e-9:
                raise ValueError("Original launch energy does not reproduce")
            if task["kind"]=="continuation":
                values,_=potential.evaluate(state.q)
                e=core.ring_energy(state.q,state.p,state.masses,dynamics.beta,values)/32
                if abs(e-float(row["final_energy_per_bead_hartree"]))>1e-9:
                    raise ValueError("20ps checkpoint energy does not reproduce")
            classifier=core.ConnectivityClassifier(cfg["atoms"],state.fragments,state.masses,
                          cfg["bond_cutoffs_bohr"],cfg["channels"],cfg["exit_distance_bohr"])
            result=integrate(dynamics,state.q,state.p,start=start,target=target,e0=e0,
                   prior_drift=float(row["max_energy_drift_per_bead_hartree"]) if task["kind"]=="continuation" else 0.,
                   classifier=classifier,check_every=check,persistence=cfg["persistence"],
                   save_every=save,out=out,metadata=meta)
        if task["kind"]=="audit":
            result["drift_fraction_of_collision_energy"]=result["max_drift"]/float(row["collision_energy_hartree"])
            if dt==0.1:
                result["baseline_reproduced"]=(result["status"]==row["status"] and
                    result["channel"]==row["channel"] and result["absolute_step"]==int(row["steps"]) and
                    abs(result["max_drift"]-float(row["max_energy_drift_per_bead_hartree"]))<1e-7)
            dump(out/"result.json",result)
        print("RESULT",json.dumps({k:result.get(k) for k in
               ("key","status","channel","dt_fs","time_ps","max_drift","baseline_reproduced")}),flush=True)
        return result
    except Exception as exc:
        result={"key":task["key"],"kind":task["kind"],"status":"failed","artifact":task["artifact"],
                "trajectory":int(task["record"]["trajectory"]),"error":f"{type(exc).__name__}: {exc}",
                "traceback":traceback.format_exc()}
        dump(out/"result.json",result);print("FAILED",json.dumps(result),flush=True)
        return result


def group(args):
    _,tasks,products,_=load_source(args.source)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    if args.kind=="continuation":
        selected=[dict(t,kind="continuation",dt_fs=0.1,target_ps=200.) for t in tasks[args.part::16]]
    else:
        selected=[dict(t,key=t["key"]+f"-dt{dt:g}",kind="audit",dt_fs=dt,target_ps=20.)
                  for t in products for dt in (0.1,0.05,0.025)]
    for t in selected:t.update(source=str(Path(args.source).resolve()),out=str(out/t["key"]))
    dump(out/"assignments.json",selected);results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context("spawn")) as pool:
        futures={pool.submit(execute,t):t for t in selected}
        for future in as_completed(futures):
            result=future.result();results.append(result);dump(out/"completed.json",results)
    dump(out/"group.json",{"kind":args.kind,"part":args.part,"assigned":len(selected),"results":results})
    if args.kind=="audit":audit_report(out)
    if any(r["status"]=="failed" for r in results):
        raise RuntimeError("Some assignments failed; failure records/checkpoints retained")
    if args.kind=="audit" and any(r.get("baseline_reproduced") is False for r in results):
        raise RuntimeError("0.1fs baseline replay disagrees; comparison not validated")


def audit_report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[]
    for trajectory in (8,51):
        fig,ax=plt.subplots(figsize=(10,5))
        for dt in (0.1,0.05,0.025):
            directory=out/f"hhoo-b6-s3-t{trajectory:03d}-dt{dt:g}"
            result=json.loads((directory/"result.json").read_text())
            rows.append({k:result.get(k) for k in ("trajectory","dt_fs","status","channel","time_ps",
                         "max_drift","final_signed_drift_hartree","drift_fraction_of_collision_energy","baseline_reproduced")})
            paths=list(directory.glob("trace_from_*.csv"))
            if not paths:continue
            data=np.genfromtxt(paths[0],delimiter=",",names=True)
            if data.size:ax.plot(data["time_fs"]/1000,data["delta_E_hartree"],label=f"dt={dt:g} fs")
        ax.set(xlabel="Time (ps)",ylabel="Delta (H_P/P) from launch (hartree)",
               title=f"Same initial q,p: b6-s3 trajectory {trajectory}")
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(out/f"energy_t{trajectory:03d}.png",dpi=160);plt.close(fig)
    with (out/"timestep_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("TIMESTEP_COMPARISON",json.dumps(rows,allow_nan=False),flush=True)


def aggregate(args):
    originals,tasks,_,configs=load_source(args.source)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);found={}
    for path in Path(args.results).rglob("result.json"):
        r=json.loads(path.read_text())
        if r.get("kind")!="continuation":continue
        if r["key"] in found:raise ValueError("Duplicate continuation result")
        found[r["key"]]=r
    wanted={t["key"] for t in tasks}
    if set(found)-wanted:raise ValueError("Unknown continuation ID")
    merged=[]
    for r in originals:
        row=dict(r)
        if r["status"]=="unresolved":
            result=found.get(r["key"])
            if result is None:row.update(status="failed",channel=None,error="Continuation missing; retained unknown")
            else:row.update(status=result["status"],channel=result.get("channel"),
                 steps=result.get("absolute_step",r["steps"]),error=result.get("error"),continued_to_ps=result.get("time_ps"))
        merged.append(row)
    summaries={}
    for b in (6,10):
        subset=[r for r in merged if r["artifact"].startswith(f"hhoo-b{b}-")]
        cfg=configs[f"hhoo-b{b}-s0"]
        if len(subset)!=256:raise ValueError("Changed original denominator")
        summaries[str(b)]=core.summarize(subset,cfg["channels"],cfg["temperature_K"],
             np.asarray(cfg["masses_amu"])*core.AMU,cfg["fragments"],b,cfg["electronic_weight"])
    dump(out/"updated_512_records.json",merged)
    dump(out/"updated_rate_bounds.json",{"source_run":SOURCE_RUN,"found":len(found),"requested":122,
         "per_bmax":summaries,"note":"Never divide by 122. Denominators remain 256 per bmax. Separate small-dt audits do not replace events in this 0.1fs ensemble."})
    print("UPDATED_512_SUMMARY",json.dumps(summaries),flush=True)
    if set(found)!=wanted or any(r["status"]=="failed" for r in found.values()):
        raise RuntimeError("Incomplete continuation ensemble; conservative partial report retained")


def selftest():
    """Real-core split/restart equivalence and original-loop regression."""
    import tempfile
    class Harmonic:
        def evaluate(self,q):return 0.025*(q*q).sum(axis=(0,1)),np.asfortranarray(0.05*q)
    rng=np.random.default_rng(721)
    q=np.asfortranarray(rng.normal(0,.1,(3,4,4)));p=np.asfortranarray(rng.normal(0,.01,q.shape))
    dyn=core.Dynamics([1.,2.,3.,4.],300.,4,.01,Harmonic(),native=True)
    e0=core.ring_energy(q,p,dyn.masses,dyn.beta,dyn.potential.evaluate(q)[0])/4
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        common=dict(e0=e0,prior_drift=0.,classifier=None,check_every=10,persistence=3,save_every=10,
                    metadata={"key":"test","kind":"test","dt_fs":dyn.dt*core.TIME_FS,
                              "target_step":100,"initial_state_sha256":"test","potential_identity":"test"})
        a=integrate(dyn,q,p,start=0,target=100,out=d/"full",**common)
        ref=core.propagate(dyn,q,p,100,check_interval=10,save_every=10)
        with np.load(d/"full/final.npz") as data:
            np.testing.assert_allclose(data["q"],ref.q,atol=1e-13,rtol=0)
            np.testing.assert_allclose(data["p"],ref.p,atol=1e-13,rtol=0)
        integrate(dyn,q,p,start=0,target=40,out=d/"short",**common)
        with np.load(d/"short/final.npz") as data:
            midq,midp=data["q"],data["p"];midg,midv=data["gradient"],data["values"]
            meta=json.loads(data["metadata"].item());meta.update(status="running",target_step=100)
        checkpoint(d/"restart.npz",midq,midp,midg,midv,meta)
        b=integrate(dyn,q,p,start=0,target=100,out=d/"restarted",restart=d/"restart.npz",**common)
        with np.load(d/"full/final.npz") as x,np.load(d/"restarted/final.npz") as y:
            np.testing.assert_array_equal(x["q"],y["q"]);np.testing.assert_array_equal(x["p"],y["p"])
        assert a["max_drift"]==b["max_drift"]
        c=integrate(dyn,q,p,start=40,target=100,out=d/"exit",**{**common,"classifier":lambda q,p:"2OH"})
        assert c["status"]=="product" and c["absolute_step"]==70 and c["channel"]=="2OH"
    print("SELFTEST_PASSED: original loop agreement, exact checkpoint restart, energy history, conservative exit reset",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","group","aggregate","selftest"))
    p.add_argument("--source",default="source-data");p.add_argument("--out",default="followup-results")
    p.add_argument("--kind",choices=("audit","continuation"),default="audit")
    p.add_argument("--part",type=int,default=0);p.add_argument("--results",default="all-results")
    a=p.parse_args()
    if not 0<=a.part<16:p.error("part must be 0..15")
    if a.command=="prepare":plan(a.source,a.out)
    elif a.command=="group":group(a)
    elif a.command=="aggregate":aggregate(a)
    else:selftest()


if __name__=="__main__":main()
