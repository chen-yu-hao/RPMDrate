"""Aggregate independent HHOO bmax=6 vs 10 bohr collision shards.

Counts all launched and unresolved trajectories in requested denominators.
Strongly correlated children from the same thermal parent are grouped for the
exploratory parent-chain bootstrap. Neither Wilson nor this bootstrap includes
potential, thermalization, bead/timestep, or trajectory-time convergence errors.
"""
from __future__ import annotations
from collections import Counter
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from rpmdrate import drpmd_core as core


CHANNELS = ("2OH","H2+O2","H2O+O")


def read_shard(directory):
    inp=json.loads((directory/"input.json").read_text())
    summary=json.loads((directory/"summary.json").read_text())
    with (directory/"trajectories.csv").open(newline="") as f:
        records=list(csv.DictReader(f))
    b=float(inp["input"]["bmax_bohr"])
    if len(records)!=64 or int(summary["requested_trajectories"])!=64:
        raise ValueError(f"Incomplete shard: {directory}")
    if summary["counts"]["failed"] or summary.get("controller_error"):
        raise RuntimeError(f"Failed trajectories in {directory}")
    if int(inp["nbeads"] if "nbeads" in inp else inp["input"]["nbeads"])!=32:
        raise ValueError("Wrong bead count")
    if int(inp["input"]["thermalization_steps"])!=200000:
        raise ValueError("Thermalization was not 20ps as configured")
    if int(inp["input"]["evolution_steps"])!=200000:
        raise ValueError("Collision cutoff was not 20ps as configured")
    if int(summary["trajectory_workers"])!=4:
        raise ValueError("Expected 4 independent thermal parent chains")
    out=[]
    for row in records:
        row=dict(row)
        row["bmax_bohr"]=b
        row["shard_name"]=directory.name
        row["parent_key"]=directory.name+"-chain"+row["chain_id"]
        row["impact_parameter_bohr"]=float(row["impact_parameter_bohr"])
        row["steps"]=int(row["steps"])
        if row["status"] not in ("product","nonreactive","unresolved"):
            raise RuntimeError(f"Unknown or failed trajectory status: {row}")
        if row["status"]=="product" and row["channel"] not in CHANNELS:
            raise ValueError(f"Unknown reaction channel: {row['channel']}")
        out.append(row)
    return b,out,summary,inp


def bootstrap_parent_estimates(records,b,channel,temperature,masses,fragments,weight,reps=2000):
    grouped={}
    for row in records:
        grouped.setdefault(row["parent_key"],[]).append(row)
    if len(grouped)<8:
        return None
    successes=sum(row["status"]=="product" and
                  (channel=="total" or row["channel"]==channel) for row in records)
    # Do not manufacture zero-width uncertainty intervals when no events occur.
    if successes<5:
        return None
    groups=list(grouped.values())
    generator=np.random.default_rng(20261029+int(b)+sum(map(ord,channel)))
    prefactor=core.summarize([], [*CHANNELS],temperature,masses,fragments,b,weight)["collision_prefactor_cm3_molecule_s"]
    values=[]
    for _ in range(reps):
        ids=generator.integers(0,len(groups),len(groups))
        rows=[r for j in ids for r in groups[j]]
        s=sum(r["status"]=="product" and
              (channel=="total" or r["channel"]==channel) for r in rows)
        values.append(prefactor*s/len(rows))
    return [float(x) for x in np.quantile(values,[.025,.975])]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("all-batches"))
    parser.add_argument("--out",type=Path,default=Path("aggregated-hhoo-512"))
    args=parser.parse_args()
    root=args.input.resolve()
    paths=sorted(p for p in root.iterdir()
                 if p.is_dir() and p.name.startswith("hhoo-b") and (p/"summary.json").is_file())
    if len(paths)!=8:
        raise RuntimeError(f"Expected 8 complete bmax/shard artifacts, got {len(paths)}")
    groups={6.0:[],10.0:[]}
    summaries={}
    all_seeds=set()
    for path in paths:
        b,records,summary,manifest=read_shard(path)
        if b not in groups:
            raise ValueError(f"Unexpected bmax {b}")
        seed=int(manifest["input"]["seed"])
        if seed in all_seeds:
            raise ValueError("Duplicated master seed across independent collision shards")
        all_seeds.add(seed)
        groups[b].extend(records)
        summaries[path.name]={
            "count":len(records),
            "counts":summary["counts"],
            "wall_seconds":summary["timing_seconds"]["wall"],
            "root_seed":seed,
        }
    if len(groups[6.0])!=256 or len(groups[10.0])!=256:
        raise RuntimeError("Each bmax must have 256 trajectories")
    seed_ledger=[]
    for b,records in groups.items():
        unique=set(r["parent_key"] for r in records)
        if len(unique)!=16:
            raise RuntimeError(f"Expected 16 independent thermal parents for b={b}, got {len(unique)}")
        seed_ledger.append({"bmax_bohr":b,"total_trajectories":len(records),"thermal_parents":len(unique)})
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    input0=json.loads((paths[0]/"input.json").read_text())["input"]
    masses=np.asarray(input0["masses_amu"],dtype=float)*core.AMU
    temperature=float(input0["temperature_K"])
    fragments=input0["fragments"]
    weight=float(input0["electronic_weight"])
    results={}
    rate_rows=[]
    for b,records in groups.items():
        summary=core.summarize(records,CHANNELS,temperature,masses,fragments,b,weight)
        total=int(summary["trajectories"])
        unknown=sum(r["status"]=="unresolved" for r in records)
        impact=np.array([r["impact_parameter_bohr"] for r in records])
        statistic={
            "rate_summary":summary,
            "sample_size":total,
            "thermal_parents":len(set(r["parent_key"] for r in records)),
            "fraction_unresolved":unknown/total,
            "impact_mean_normalized_b2":float(np.mean((impact/b)**2)),
            "outer_6_to_10_bohr": None,
            "channel_parent_bootstrap_ci":{}
        }
        if b==10.:
            outer=[r for r in records if r["impact_parameter_bohr"]>6.]
            statistic["outer_6_to_10_bohr"]={
                "count":len(outer),
                "status_counts":dict(Counter(r["status"] for r in outer)),
                "channel_counts":dict(Counter(
                    r["channel"] for r in outer if r["status"]=="product"
                ))
            }
        for channel in ("total",*CHANNELS):
            item=summary["total_reaction"] if channel=="total" else summary["channels"][channel]
            statistic["channel_parent_bootstrap_ci"][channel]=bootstrap_parent_estimates(
                records,b,channel,temperature,masses,fragments,weight)
            rate_rows.append({
                "bmax_bohr":b,"channel":channel,
                "sample_size":total,"parents":statistic["thermal_parents"],
                "product_count":item["count"],"unknown_count":unknown,
                "rate_cm3_molecule_s":item["rate_cm3_molecule_s"],
                "observed_lower_cm3_molecule_s":item["observed_lower_cm3_molecule_s"],
                "possible_upper_cm3_molecule_s":item["possible_upper_cm3_molecule_s"],
                "iid_wilson95_cm3_molecule_s":json.dumps(item["iid_wilson95_cm3_molecule_s"]),
                "parent_bootstrap95_cm3_molecule_s":json.dumps(
                    statistic["channel_parent_bootstrap_ci"][channel]),
            })
        results[str(int(b))]=statistic
    with (out/"rates.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rate_rows[0]))
        w.writeheader();w.writerows(rate_rows)
    combined={
        "format":"RPMDrate.HHOO.impact.pilot.v1",
        "temperature_K":temperature,
        "nbeads":32,
        "initial_thermalization_ps":20,
        "sampling_interval_fs":100,
        "max_collision_propagation_ps":20,
        "per_bmax":results,
        "shards":summaries,
        "parent_ledger":seed_ledger,
        "notes":[
            "512 total collisions over two bmax choices; 256 per bmax.",
            "16 independent thermal parent chains per bmax, 16 child collisions per chain.",
            "Unresolved results remain unknown; channel rate is null if any unknown.",
            "Wilson iid intervals do not account for parent-chain autocorrelation.",
            "Cluster bootstrap is exploratory; fewer than 5 product events suppresses it.",
            "At b=10 outer shell (6,10] must contain no significant reactive contribution before b=6 can be deemed adequate.",
            "The HHOO seed geometry, PES spin assignment, 32-bead sampling and 0.1fs timestep require independent validation.",
        ]
    }
    (out/"impact_summary.json").write_text(json.dumps(combined,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for i,b in enumerate((6.,10.)):
        r=groups[b]
        x=np.array([v["impact_parameter_bohr"] for v in r])
        axs[i].hist(x*x/b**2,bins=12,color="#56859e",edgecolor="white")
        axs[i].axhline(len(r)/12,color="gray",linestyle="--",label="uniform b²")
        axs[i].set(xlabel="b² / bmax²",ylabel="Collision count",
                   title=f"bmax={b:g} bohr | {len(r)} trajectories")
        axs[i].legend()
    fig.tight_layout()
    fig.savefig(out/"impact_sampling.png",dpi=160);plt.close(fig)
    print("IMPACT_SUMMARY",json.dumps({
        "per_bmax":{
            key:{
                "counts":value["rate_summary"]["counts"],
                "channels":value["rate_summary"]["channels"],
                "total_reaction":value["rate_summary"]["total_reaction"],
                "outer_shell":value["outer_6_to_10_bohr"],
                "b2_mean":value["impact_mean_normalized_b2"]
            } for key,value in results.items()},
        "parent_counts":seed_ledger
    }),flush=True)


if __name__=="__main__":
    main()
