"""Post-process 4 direct-RPMD HHOO trajectories using saved centroid coordinates.

All six interatomic distances and the H0--HO2 fragment COM distance are written
for EVERY saved frame. Plotting may subsample only for readable graphics.
Time-series classification remains owned by dRPMD; bonds here are diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATOMS = ("H", "H", "O", "O")
MASSES = (1.00782503223, 1.00782503223, 15.99491461957, 15.99491461957)
PAIRS = (
    ("H0_H1", 0, 1),
    ("O2_O3", 2, 3),
    ("H0_O2", 0, 2),
    ("H0_O3", 0, 3),
    ("H1_O2", 1, 2),
    ("H1_O3", 1, 3),
)
KEYS = tuple(name for name, _, _ in PAIRS) + ("H0_HO2_COM",)
HEADERS = ("step", "time_fs", "time_ps", "energy_per_bead_hartree") + KEYS


def atom_pos(row, i):
    return tuple(float(row[f"{ATOMS[i]}{i}_{axis}_bohr"]) for axis in "xyz")


def geometry(row):
    xyz = [atom_pos(row, i) for i in range(4)]
    values = {name: math.dist(xyz[i], xyz[j]) for name, i, j in PAIRS}
    center = tuple(
        sum(MASSES[i] * xyz[i][axis] for i in (1, 2, 3)) / sum(MASSES[1:])
        for axis in range(3)
    )
    values["H0_HO2_COM"] = math.dist(xyz[0], center)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite centroid bond/COM distance")
    return values


def extract_trace(root, output, chain, expected, downsample=5):
    src = root / "chains" / f"chain_{chain:04d}" / "trajectory_000001.csv"
    if not src.is_file():
        raise FileNotFoundError(f"No centroid trajectory saved for chain {chain}: {src}")
    target = output / f"trajectory_{chain:04d}_distances.csv"
    series = {"time_ps": [], **{key: [] for key in KEYS}}
    extrema = {}
    rows = 0
    initial = final = None
    last_step = None
    last_time = None
    last_values = None
    with src.open(newline="", encoding="utf-8") as stream, \
            target.open("w", newline="", encoding="utf-8") as out:
        source = csv.DictReader(stream)
        sink = csv.DictWriter(out, fieldnames=list(HEADERS))
        sink.writeheader()
        for row in source:
            step = int(row["step"])
            time_fs = float(row["time_fs"])
            energy = float(row["ring_energy_per_bead_hartree"])
            if not math.isfinite(time_fs) or not math.isfinite(energy):
                raise ValueError(f"Nonfinite trace data at chain {chain}, step {step}")
            if last_step is not None and (step <= last_step or time_fs <= last_time):
                raise ValueError(f"Non-monotone trace at chain {chain}, step {step}")
            values = geometry(row)
            sink.writerow({"step": step, "time_fs": time_fs,
                           "time_ps": time_fs / 1000.,
                           "energy_per_bead_hartree": energy, **values})
            if rows == 0:
                initial = dict(values)
                extrema = {
                    key: {"min_bohr": value, "min_time_fs": time_fs,
                          "max_bohr": value, "max_time_fs": time_fs}
                    for key, value in values.items()
                }
            for key, value in values.items():
                item = extrema[key]
                if value < item["min_bohr"]:
                    item.update(min_bohr=value, min_time_fs=time_fs)
                if value > item["max_bohr"]:
                    item.update(max_bohr=value, max_time_fs=time_fs)
            if rows % downsample == 0:
                series["time_ps"].append(time_fs / 1000.)
                for key in KEYS:
                    series[key].append(values[key])
            final, last_values = dict(values), values
            last_step, last_time = step, time_fs
            rows += 1

    if rows == 0:
        raise ValueError(f"Empty centroid trace from chain {chain}")
    if last_step != int(expected["steps"]):
        raise ValueError(
            f"Chain {chain} trace ends at {last_step}, but result says {expected['steps']} steps"
        )
    if series["time_ps"][-1] != last_time / 1000.:
        series["time_ps"].append(last_time / 1000.)
        for key in KEYS:
            series[key].append(last_values[key])
    return {
        "trajectory": chain,
        "status": expected["status"],
        "channel": expected.get("channel") or None,
        "steps": last_step,
        "time_fs": last_time,
        "saved_frames": rows,
        "initial_distances_bohr": initial,
        "final_distances_bohr": final,
        "distance_extrema": extrema,
        "distance_csv": str(target.relative_to(root)),
        "centroid_csv": str(src.relative_to(root)),
    }, series


def make_plots(analyses, series, output, duration_ps):
    fig, axs = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    fig.suptitle("H + HO2: 32-bead centroid distances | 4 concurrent CPU trajectories",
                 fontsize=15)
    for i in range(4):
        entry = analyses[i]
        data = series[i]
        t = data["time_ps"]
        a, b, c = axs[i]
        a.plot(t, data["H0_HO2_COM"], linewidth=0.85, label="H0-HO2 COM")
        b.plot(t, data["H0_H1"], linewidth=0.85, label="H0-H1")
        b.plot(t, data["O2_O3"], linewidth=0.85, label="O2-O3")
        for name in ("H0_O2", "H0_O3", "H1_O2", "H1_O3"):
            c.plot(t, data[name], linewidth=0.72, label=name.replace("_", "-"))
        for axis in (a, b, c):
            axis.grid(alpha=0.2)
            axis.set_xlim(0, duration_ps)
            axis.set_ylabel(f"Trajectory {i + 1} ({entry['status']})\nDistance (bohr)", fontsize=9)
            axis.legend(loc="upper right", fontsize=7, ncol=2)
    for j, title in enumerate(("Reactant centroid COM", "H-H and O-O", "All four H-O pairs")):
        axs[0, j].set_title(title)
        axs[3, j].set_xlabel("Time (ps)")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output / "four_trajectories_distances.png", dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i in range(4):
        ax.plot(series[i]["time_ps"], series[i]["H0_HO2_COM"], linewidth=1,
                label=f"Trajectory {i + 1} ({analyses[i]['status']})")
    ax.set(xlabel="Time (ps)", ylabel="H-HO2 centroid COM distance (bohr)",
           title="H + HO2 scattering: four independent 32-bead trajectories")
    ax.set_xlim(0, duration_ps)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "four_trajectories_COM.png", dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected-trajectories", type=int, default=4)
    parser.add_argument("--duration-ps", type=float, default=200.)
    args = parser.parse_args()
    root = args.results.resolve()
    output = root / "geometry"
    output.mkdir(parents=True, exist_ok=True)
    with (root / "trajectories.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != args.expected_trajectories:
        raise RuntimeError(
            f"Expected {args.expected_trajectories} result records, got {len(records)}"
        )
    by_id = {int(record["trajectory"]): record for record in records}
    if sorted(by_id) != list(range(1, args.expected_trajectories + 1)):
        raise RuntimeError("Missing or duplicated trajectory number")

    summaries, plots = [], []
    for n in range(1, args.expected_trajectories + 1):
        item, trace = extract_trace(root, output, n, by_id[n])
        summaries.append(item)
        plots.append(trace)
        print("GEOMETRY_TRAJECTORY", json.dumps({
            "trajectory": n, "status": item["status"], "steps": item["steps"],
            "saved_frames": item["saved_frames"],
            "COM_min": item["distance_extrema"]["H0_HO2_COM"]["min_bohr"],
            "COM_final": item["final_distances_bohr"]["H0_HO2_COM"],
            "OO_max": item["distance_extrema"]["O2_O3"]["max_bohr"],
        }), flush=True)

    make_plots(summaries, plots, output, args.duration_ps)
    (output / "geometry_summary.json").write_text(
        json.dumps({"note": "Only saved centroid frames; not bead-resolved geometry.",
                    "trajectories": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    print("GEOMETRY_COMPLETE", json.dumps({
        "trajectories": len(summaries),
        "frames": sum(x["saved_frames"] for x in summaries),
        "output": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
