"""Constrained 32-bead H + HO2 thermal equilibration diagnostic.

Uses the same Dynamics.thermalize() and initial SeedSequence stream as dRPMD,
not a separate thermostat. Four independent thermal parent chains are run
concurrently. The pilot is NOT an acceptance certificate for equilibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from rpmdrate import drpmd_core as core
from rpmdrate.drpmd_backends import PotentialSpec, SerialPotential
from rpmdrate.drpmd_parallel import make_plan


OBSERVABLES = (
    "potential_per_bead_hartree", "kinetic_per_bead_hartree",
    "spring_per_bead_hartree", "H1_O2_centroid_bohr",
    "H1_O3_centroid_bohr", "O2_O3_centroid_bohr",
    "HO2_ring_spread_bohr", "H_HO2_COM_bohr"
)


def observables(q, p, masses, beta, potential):
    nbeads = q.shape[2]
    energy, _ = potential.evaluate(q)
    omega = nbeads / beta
    spring = 0.5 * np.sum(masses[None, :, None] * omega**2 *
                          (q - np.roll(q, 1, axis=2))**2) / nbeads
    kinetic = np.sum(p*p / (2. * masses[None, :, None])) / nbeads
    centroid = q.mean(axis=2)
    spread = np.sqrt(np.mean(
        (q[:, 1:4] - centroid[:, 1:4, None])**2))
    fragment_mass = masses[1:].sum()
    com = np.sum(centroid[:, 1:] * masses[None, 1:], axis=1) / fragment_mass
    distances = [
        float(np.linalg.norm(centroid[:, i] - centroid[:, j]))
        for i, j in ((1, 2), (1, 3), (2, 3))
    ]
    values = (
        float(energy.mean()), float(kinetic), float(spring),
        *distances, float(spread),
        float(np.linalg.norm(centroid[:, 0] - com))
    )
    if not all(math.isfinite(x) for x in values):
        raise FloatingPointError("Nonfinite thermal diagnostic")
    return values


def worker(task):
    index, chain_seed, config, weights, root, nsteps, stride = task
    root = Path(root)
    os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    from threadpoolctl import threadpool_limits
    masses = np.array(config["masses_amu"], dtype=float) * core.AMU
    fragments = core.validate_fragments(config["fragments"], len(masses))
    nbeads = int(config["nbeads"])
    temperature = float(config["temperature_K"])
    dt = float(config["dt_fs"]) / core.TIME_FS
    beta = 1 / (core.KB * temperature)
    state_rng = np.random.default_rng(np.random.SeedSequence(chain_seed).spawn(2)[0])
    q = np.asfortranarray(np.repeat(
        np.asarray(config["initial_geometry_bohr"], dtype=float).T[:, :, None],
        nbeads, axis=2))
    separation = float(config["initial_separation_bohr"])
    for fragment, target in zip(fragments, (np.zeros(3), np.array([0.,0.,separation]))):
        idx = list(fragment)
        q[:, idx, :] += (
            target - core.fragment_com(q, masses, fragment)
        )[:, None, None]
    p = core.sample_momenta(masses, nbeads, beta, state_rng)
    rows = []
    spec = PotentialSpec("rpmdrate.drpmd_hhoo", initializer_kwargs={"weights_path":weights})
    with threadpool_limits(limits=1, user_api="blas"), SerialPotential(spec) as potential:
        dynamics = core.Dynamics(masses, temperature, nbeads, dt, potential, threads=1)
        clock = 0
        started = time.perf_counter()
        for stop in range(stride, nsteps + 1, stride):
            _, clock = dynamics.thermalize(q, p, fragments, stride,
                                             int(config["andersen_interval"]),
                                             state_rng, clock)
            row = [stop * float(config["dt_fs"]) / 1000.] + list(
                observables(q, p, masses, beta, potential))
            rows.append(row)
        core.ConnectivityClassifier(
            config["atoms"], fragments, masses,
            config["bond_cutoffs_bohr"], config["channels"],
            config["exit_distance_bohr"]
        ).validate_reactants(q)
        elapsed = time.perf_counter() - started
    path = root / f"chain_{index:02d}.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_ps", *OBSERVABLES])
        writer.writerows(rows)
    result = {
        "chain": index, "seed": chain_seed, "steps": nsteps,
        "frames": len(rows), "seconds": elapsed,
        "initial_HO2_seed": config["initial_geometry_bohr"],
        "final_observables": dict(zip(OBSERVABLES, rows[-1][1:]))
    }
    (root / f"chain_{index:02d}.json").write_text(json.dumps(result,indent=2))
    print("THERMAL_CHAIN", json.dumps(result), flush=True)
    return result


def split_rhat(chains):
    """Conventional split-R-hat; potential multimodality is not ruled out."""
    halves = np.concatenate([chains[:, :chains.shape[1]//2, :],
                            chains[:, chains.shape[1]//2:, :]], axis=0)
    _, n, _ = halves.shape
    chain_mean = halves.mean(axis=1)
    B = n * chain_mean.var(axis=0, ddof=1)
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    mixed = (n - 1.) / n * W + B / n
    result = np.sqrt(mixed / np.maximum(W, 1e-30))
    return result


def autocorr_ess(chains, component):
    """Conservative positive-lag integrated autocorrelation ESS approximation."""
    x = chains[:, :, component].copy()
    m, n = x.shape
    x -= x.mean(axis=1,keepdims=True)
    variance = np.sum(x*x, axis=1)
    if np.any(variance <= 0):
        return 0., None
    maxlag = min(n//2, 250)
    rho = []
    for lag in range(1, maxlag+1):
        r = np.mean(np.sum(x[:,:n-lag]*x[:,lag:],axis=1) / variance *
                    n / (n-lag))
        if r <= 0:
            break
        rho.append(float(r))
    tau = 1. + 2*sum(rho)
    return float(m*n/tau), float(tau)


def report(root, fraction):
    files = sorted(root.glob("chain_*.csv"))
    if len(files) != 4:
        raise AssertionError("Exactly four thermal chains are required")
    datasets = []
    for path in files:
        with path.open(newline="") as f:
            rd = csv.DictReader(f)
            datasets.append([[float(row[k]) for k in OBSERVABLES] for row in rd])
    data = np.asarray(datasets, dtype=np.float64)
    if data.ndim != 3 or len(data[0]) < 20:
        raise ValueError("Insufficient thermal observations")
    burn = int(data.shape[1]*fraction)
    post = data[:,burn:,:]
    split = split_rhat(post)
    diagnostics = []
    for key in OBSERVABLES:
        col = OBSERVABLES.index(key)
        values = post[:,:,col]
        means = values.mean(axis=1)
        ess, tau = autocorr_ess(post,col)
        first = post[:,:len(post[0])//2,col].mean()
        last = post[:,len(post[0])//2:,col].mean()
        obs = {
            "observable": key,
            "split_rhat": float(split[col]),
            "approx_ess": ess,
            "autocorr_factor": tau,
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "parent_chain_means": [float(x) for x in means],
            "first_half_mean": float(first),
            "last_half_mean": float(last),
            "post_burn_samples": int(values.size),
        }
        diagnostics.append(obs)
        print("THERMAL_DIAGNOSTIC",json.dumps(obs),flush=True)
    # Diagnostic flags are deliberately non-binary certification:
    # no absolute convergence can be inferred from a short fixed pilot.
    watch = ("potential_per_bead_hartree","spring_per_bead_hartree",
             "H1_O2_centroid_bohr","O2_O3_centroid_bohr")
    warnings = [
        f"{r['observable']}: split_Rhat={r['split_rhat']:.3f}"
        for r in diagnostics
        if r["observable"] in watch and r["split_rhat"] > 1.05
    ]
    summaries = {
        "format":"RPMDrate.HHOO.thermal.pilot.v1",
        "chains":4, "thermal_length_ps":float(data.shape[1]*json.loads((root/"spec.json").read_text())["stride_fs"]/1000),
        "post_burn_fraction":fraction,
        "note":"Pilot diagnostics are neither proof of equilibration nor independent sampling. Autocorrelation ESS is approximate.",
        "diagnostics":diagnostics,"warnings":warnings,
    }
    (root/"convergence_report.json").write_text(json.dumps(summaries,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(1,data.shape[1]+1)*json.loads((root/"spec.json").read_text())["stride_fs"]/1000
    fig, axs = plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,name in zip(axs.flat,watch):
        idx=OBSERVABLES.index(name)
        for i in range(4):
            ax.plot(t,data[i,:,idx],linewidth=0.7,alpha=0.65,label=f"chain{i+1}")
        ax.axvline(t[burn],color="black",linestyle="--",alpha=.5)
        ax.set_title(name);ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2,fontsize=8)
    axs[-1,0].set_xlabel("Thermalization time (ps)")
    axs[-1,1].set_xlabel("Thermalization time (ps)")
    fig.tight_layout()
    fig.savefig(root/"thermal_observables.png",dpi=160)
    plt.close(fig)
    print("THERMAL_REPORT",json.dumps({
        "warnings":warnings,
        "results":[{"name":r["observable"],"rhat":r["split_rhat"],
                    "ess":r["approx_ess"]} for r in diagnostics]
    }),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,default=Path("examples/drpmd/hhoo_smoke.json"))
    ap.add_argument("--output",type=Path,default=Path("thermal-pilot-32b"))
    ap.add_argument("--nbeads",type=int,default=32)
    ap.add_argument("--length-ps",type=float,default=20.)
    ap.add_argument("--stride-fs",type=float,default=10.)
    ap.add_argument("--burn-fraction",type=float,default=0.5)
    args=ap.parse_args()
    config=json.loads(args.input.read_text())
    if config["potential"].get("type") != "hhoo":
        raise ValueError("Only the supplied HHOO model is benchmarked")
    config["nbeads"]=args.nbeads
    steps=args.length_ps*1000/config["dt_fs"]
    stride=args.stride_fs/config["dt_fs"]
    if not(steps>=100 and stride>=1 and math.isclose(steps,round(steps)) and
           math.isclose(stride,round(stride)) and round(steps)%round(stride)==0 and
           0.1<=args.burn_fraction<0.8):
        raise ValueError("Invalid length/stride/burn_fraction")
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    weight=str((args.input.resolve().parent/config["potential"]["weights"]).resolve())
    if not Path(weight).is_file():
        raise FileNotFoundError(weight)
    spec={"nbeads":args.nbeads,"length_ps":args.length_ps,
          "stride_fs":args.stride_fs,"dt_fs":config["dt_fs"],
          "seed":config["seed"],"temperature_K":config["temperature_K"]}
    (root/"spec.json").write_text(json.dumps(spec,indent=2)+"\n")
    plan=make_plan(4,mode="serial",trajectory_workers=4,seed=config["seed"])
    tasks=[(chain.chain_id,chain.seed,config,weight,str(root),round(steps),round(stride))
           for chain in plan["chains"]]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context("spawn")) as pool:
        results=list(pool.map(worker,tasks))
    (root/"chains.json").write_text(json.dumps(results,indent=2)+"\n")
    report(root,args.burn_fraction)
    print("THERMAL_PILOT_DONE",json.dumps({"chains":4,"total_frames":4*round(steps)//round(stride)}),flush=True)


if __name__=="__main__":
    main()
