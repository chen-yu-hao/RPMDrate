"""Analyze 100 fs HHOO centroid trajectories, no trajectory interpolation."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path("four-hhoo-results")
masses = np.array([1.00782503223, 1.00782503223, 15.99491461957, 15.99491461957])
header = ["step", "time_fs", "H0-H1", "O2-O3", "H0-O2", "H0-O3",
          "H1-O2", "H1-O3", "H-O_min", "H_HO2_COM"]
color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
all_summary = {}
fig, axes = plt.subplots(4, 3, figsize=(15.5, 13.2), sharex=True)
for n in range(1, 5):
    filename = folder / f"trajectory_{n:06d}.csv"
    with filename.open(newline="") as f:
        rows = list(csv.DictReader(f))
    time = np.array([float(r["time_fs"]) for r in rows])
    xyz = np.array([[[float(r[f"{a}{i}_{axis}_bohr"]) for axis in "xyz"]
                     for i, a in enumerate(["H", "H", "O", "O"])] for r in rows])
    def bond(i,j):
        return np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1)
    d = [bond(0,1), bond(2,3), bond(0,2), bond(0,3), bond(1,2), bond(1,3)]
    com = np.sum(xyz[:, 1:4] * masses[None, 1:4, None], axis=1) / masses[1:4].sum()
    rcom = np.linalg.norm(xyz[:, 0] - com, axis=1)
    ho_min = np.min(np.stack(d[2:]),axis=0)
    series = [*d,ho_min,rcom]
    with (folder / f"distances_{n:06d}.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(header)
        writer.writerows(zip(np.arange(len(time)),time,*series))
    vals = {}
    for name, arr in zip(header[2:],series):
        idx=int(np.argmin(arr))
        vals[name]={"initial":float(arr[0]),"final":float(arr[-1]),
                    "minimum":float(arr[idx]),"min_time_fs":float(time[idx]),
                    "maximum":float(np.max(arr))}
    samples = [{"time_fs":float(time[j]), "R_H_HO2_COM_bohr":float(rcom[j]),
                "H_O_min_bohr":float(ho_min[j]),"H_H_bohr":float(d[0][j]),
                "O_O_bohr":float(d[1][j])}
               for j in np.arange(0,len(time),200)]
    all_summary[str(n)]={"saved_frames":len(time),"distances":vals,"samples":samples}
    row=axes[n-1]
    row[0].plot(time,rcom,color=color[0],label="H--HO2 COM")
    row[0].axhline(20,color="gray",linestyle=":",alpha=.45)
    row[1].plot(time,d[0],label="H0-H1")
    row[1].plot(time,d[1],label="O2-O3")
    for j,label in enumerate(["H0-O2","H0-O3","H1-O2","H1-O3"]):
        row[2].plot(time,d[j+2],label=label,linewidth=1.15)
    for ax in row:
        ax.grid(alpha=.25)
        ax.set_ylabel(f"Trajectory {n}\nDistance (bohr)")
        ax.set_xlim(time[0],time[-1])
    row[0].legend(loc="best",fontsize=8)
    row[1].legend(loc="best",fontsize=8)
    row[2].legend(loc="best",fontsize=8,ncol=2)
for j,title in enumerate(["Reactant centroid COM","H--H / O--O","All four H--O bonds"]):
    axes[0,j].set_title(title)
    axes[-1,j].set_xlabel("Time (fs)")
fig.suptitle("H + HO2 | 4 trajectories | 300 K, 8 beads, 0.1 fs",fontsize=15)
fig.tight_layout(rect=(0,0,1,.975))
fig.savefig(folder/"distances_4x3.png",dpi=180)
plt.close(fig)
(folder/"distance_analysis.json").write_text(json.dumps(all_summary,indent=2)+"\n")
for k,v in all_summary.items():
    print("ANALYSIS_TRAJECTORY",k,json.dumps({
       "frames":v["saved_frames"],
       "R_COM":v["distances"]["H_HO2_COM"],
       "min_HO":v["distances"]["H-O_min"],
       "H-H":v["distances"]["H0-H1"],
       "O-O":v["distances"]["O2-O3"],
       "samples":v["samples"]},separators=(",",":")))
