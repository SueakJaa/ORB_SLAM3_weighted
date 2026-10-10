"""
Fig. 2 for the paper: ATE RMSE of every run, per method, as a box plot.
 
Reads the evo .eval files written by run_v5_gpu.sh:
    results_v5_gpu/<sequence>/<mode>_<run>.eval
 
Usage (in WSL):
    cd ~/ORB_SLAM3
    python3 plot_fig2.py                                  # Walking XYZ -> fig2_ate_boxplot.pdf/.png
    python3 plot_fig2.py --seq walking_static --out fig_ws
    python3 plot_fig2.py --linear                         # linear y-axis instead of log
 
It also prints mean, std, median and the median run of each method
(use those run numbers for Fig. 3).
"""
import argparse
import glob
import os
import re
import statistics as st
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
MODES = [("baseline", "Baseline"), ("yolo", "YOLO"), ("geo", "Geometry"), ("proposed", "Proposed")]
 
 
def read_runs(folder, mode):
    """Return {run_number: rmse} for one mode."""
    runs = {}
    for path in glob.glob(os.path.join(folder, f"{mode}_*.eval")):
        m = re.search(rf"{mode}_(\d+)\.eval$", os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "rmse":
                    runs[int(m.group(1))] = float(parts[1])
                    break
    return runs
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/ORB_SLAM3/results_v5_gpu"))
    ap.add_argument("--seq", default="walking_xyz")
    ap.add_argument("--out", default="fig2_ate_boxplot")
    ap.add_argument("--linear", action="store_true", help="linear y-axis (default: log)")
    args = ap.parse_args()
 
    folder = os.path.join(args.root, args.seq)
    names, data = [], []
    print(f"== {args.seq}  ({folder})")
    for mode, label in MODES:
        runs = read_runs(folder, mode)
        if not runs:
            print(f"  {label:9s} no .eval files found")
            continue
        vals = [runs[k] for k in sorted(runs)]
        ordered = sorted(runs.items(), key=lambda kv: kv[1])
        med_run = ordered[(len(ordered) - 1) // 2][0]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:9s} n={len(vals):2d}  mean={st.mean(vals):.4f}  std={sd:.4f}  "
              f"median={st.median(vals):.4f}  median-run={med_run}")
        names.append(label)
        data.append(vals)
 
    if not data:
        raise SystemExit("No data found - check --root and --seq.")
 
    # ---- figure: one IEEE column wide (3.5 in) ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,   # embed TrueType fonts (IEEE PDF checks)
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
 
    ax.boxplot(
        data, widths=0.5, patch_artist=True, showfliers=False,
        medianprops=dict(color="#0d366b", linewidth=1.5),
        boxprops=dict(facecolor="#cde2fb", edgecolor="#256abf", linewidth=0.8),
        whiskerprops=dict(color="#256abf", linewidth=0.8),
        capprops=dict(color="#256abf", linewidth=0.8),
    )
    # every run as a dot, spread slightly so they do not overlap
    for i, vals in enumerate(data, start=1):
        n = len(vals)
        xs = [i + (-0.12 + 0.24 * k / max(n - 1, 1)) for k in range(n)]
        ax.scatter(xs, sorted(vals), s=10, color="#2a78d6",
                   edgecolor="white", linewidth=0.4, zorder=3)
 
    if not args.linear:
        ax.set_yscale("log")
        ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        lo = min(min(v) for v in data) * 0.8
        hi = max(max(v) for v in data) * 1.3
        ax.set_ylim(lo, hi)
        ax.set_yticks([t for t in ticks if lo <= t <= hi])
        ax.set_yticklabels([f"{t:g}" for t in ticks if lo <= t <= hi])
        ax.set_ylabel("ATE RMSE [m] (log scale)")
    else:
        ax.set_ylim(bottom=0)
        ax.set_ylabel("ATE RMSE [m]")
 
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
 
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)
    print(f"Saved {args.out}.pdf and {args.out}.png")
 
 
if __name__ == "__main__":
    main()