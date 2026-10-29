"""Run one independently-seeded 64-collision H + HO2 pilot shard.

The four CPU cores work on four independent parent chains, not four beads.
The 20-ps maximum collision time is a *pilot timeout*; unresolved trajectories
must NOT be counted as nonreactive or silently discarded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from rpmdrate.drpmd import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("examples/drpmd/hhoo_smoke.json"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--bmax",type=float,required=True)
    parser.add_argument("--shard",type=int,required=True)
    parser.add_argument("--trajectories",type=int,default=64)
    parser.add_argument("--parent-chains",type=int,default=4)
    parser.add_argument("--thermal-ps",type=float,default=20.)
    parser.add_argument("--sampling-fs",type=float,default=100.)
    parser.add_argument("--max-collision-ps",type=float,default=20.)
    parser.add_argument("--save-fs",type=float,default=10.)
    args=parser.parse_args()
    if args.shard<0 or args.trajectories<1 or args.parent_chains<1:
        parser.error("Need nonnegative shard and positive trajectory/parent counts")
    if args.trajectories % args.parent_chains:
        parser.error("For a balanced pilot use trajectories divisible by parent-chains")
    src=args.input.resolve()
    cfg=json.loads(src.read_text())
    dt=float(cfg["dt_fs"])
    for name,duration in (
        ("thermal",args.thermal_ps*1000),
        ("sampling",args.sampling_fs),
        ("collision",args.max_collision_ps*1000),
        ("save",args.save_fs)
    ):
        steps=duration/dt
        if not(steps>=1 and abs(steps-round(steps))<1e-8):
            parser.error(name+" interval not an integer multiple of dt")
    if not(0 < args.bmax < float(cfg["initial_separation_bohr"])):
        parser.error("bmax must be between zero and initial separation")
    if args.bmax not in (6.,10.):
        parser.error("This fixed experimental comparison uses bmax=6 or 10 bohr")
    cfg["nbeads"]=32
    cfg["thermalization_steps"]=round(args.thermal_ps*1000/dt)
    cfg["sampling_steps"]=round(args.sampling_fs/dt)
    cfg["evolution_steps"]=round(args.max_collision_ps*1000/dt)
    cfg["save_every"]=round(args.save_fs/dt)
    cfg["trajectories"]=args.trajectories
    cfg["bmax_bohr"]=args.bmax
    cfg["parallel"]={"trajectory_workers":args.parent_chains}
    # Each bmax/shard uses a disjoint independent seed stream.
    master=int(cfg["seed"])
    cfg["seed"]=int(np.random.SeedSequence(master,spawn_key=(int(args.bmax),args.shard)).generate_state(1,dtype=np.uint64)[0])
    cfg["description"]=(
        "Exploratory HHOO 32-bead collision batch after 20ps thermalization. "
        "Unresolved after 20ps are explicitly unknown; not a converged reaction rate."
    )
    args.out.parent.mkdir(parents=True,exist_ok=True)
    print("SHARD_CONFIGURATION",json.dumps({
        "bmax":args.bmax,"shard":args.shard,"seed":cfg["seed"],
        "thermal_ps":args.thermal_ps,"sampling_fs":args.sampling_fs,
        "max_collision_ps":args.max_collision_ps,
        "nbeads":32,"trajectories":args.trajectories,
        "parent_chains":args.parent_chains
    }),flush=True)
    result=run(cfg,base=src.parent,mode="serial",workers=1,
               trajectory_workers=args.parent_chains,output=args.out)
    if result["requested_trajectories"]!=args.trajectories:
        raise RuntimeError("Wrong requested trajectory count")
    counts=result["counts"]
    print("SHARD_RESULT",json.dumps({
        "bmax":args.bmax,"shard":args.shard,"counts":counts,
        "wall_seconds":result["timing_seconds"]["wall"],
        "rate_estimate":result["total_reaction"],
        "output":str(args.out)
    }),flush=True)
    if counts["failed"] or result.get("controller_error") or result["failed_parent_chains"]:
        raise RuntimeError("Collision shard had failures or missing trajectories")


if __name__=="__main__":
    main()
