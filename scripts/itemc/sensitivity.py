#!/usr/bin/env python3
"""Item C: how much do the results move when ONE threshold of the final method changes?

Usage (from ~/ORB_SLAM3), after run_experiments.py + evaluate.py on scripts/itemc/itemc_sens.yaml:
    python3 scripts/itemc/sensitivity.py scripts/itemc/itemc_sens.yaml

Per parameter, a table: each tried value against the final setting (DEFAULT, marked *), median ATE per
sequence with failed runs in brackets; then the "stable range" - the values for which every sequence stays
within +-TOL (default 20 %) of DEFAULT's median ATE (or within 2 mm, for the already very accurate ones) and
has no more failed runs than DEFAULT. With 3 runs per point this is descriptive, not a significance test.
Figure: <experiment>/report/sensitivity.png - one panel per parameter, median ATE relative to DEFAULT.
"""
import csv
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from common import load_config, exp_dir, short  # noqa: E402

TOL, ABS_MM = 0.20, 2.0


def main(cfg_path):
    cfg = load_config(cfg_path)
    params = json.load(open(os.path.join(HERE, "params.json")))
    thr = cfg["fail_threshold_m"]
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(exp_dir(cfg), "metrics.csv"))):
        ok = r["status"] == "ok" and r["ate_rmse"]
        by[(r["variant"], r["sequence"])].append(float(r["ate_rmse"]) if ok else float("inf"))
    seqs = cfg["sequences"]
    stat = lambda v, s: (st.median(by[(v, s)]), sum(1 for x in by[(v, s)] if x > thr)) if by.get((v, s)) else (None, None)
    name = lambda s: short(s).replace("fr3/", "").replace("bonn/", "bonn_")[:16]
    rows_fig = []
    for p in params:
        vals = sorted(p["values"] + [p["default"]])
        print(f"\n== {p['name']}: {p['meaning']} (final: {p['default']})")
        print(f"{'value':>8s}" + "".join(f"{name(s):>19s}" for s in seqs))
        stable, line = [], []
        for v in vals:
            var = "DEFAULT" if v == p["default"] else f"{p['name']}_{v}"
            cells, ok_all, rel = [], True, []
            for s in seqs:
                m, f = stat(var, s)
                m0, f0 = stat("DEFAULT", s)
                if m is None or m0 is None:
                    cells.append(f"{'-':>19s}")
                    ok_all = False
                    rel.append(float("nan"))
                    continue
                cells.append(f"{m:12.4f} ({f}/{len(by[(var, s)])})")
                rel.append(m / m0 if m0 > 0 else float("nan"))
                within = abs(m - m0) <= max(TOL * m0, ABS_MM / 1000)
                ok_all &= within and f <= f0
            mark = "*" if v == p["default"] else " "
            print(f"{v:>7}{mark}" + "".join(cells) + ("" if v == p["default"] else ("   stable" if ok_all else "   CHANGES")))
            if ok_all or v == p["default"]:
                stable.append(v)
            line.append((v, rel))
        print(f"   stable for: {stable}")
        rows_fig.append((p, line))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), squeeze=False)
    for ax, (p, line) in zip(axes.ravel(), rows_fig):
        xs = list(range(len(line)))
        for k, s in enumerate(seqs):
            ys = [rel[k] for _, rel in line]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color=colors[k % len(colors)], label=name(s))
        ax.axhspan(1 - TOL, 1 + TOL, color="#dddddd", alpha=0.6, linewidth=0)
        ax.axhline(1, color="#888888", linewidth=0.8)
        d = [v for v, _ in line].index(p["default"])
        ax.axvline(d, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v, _ in line], fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.4, 5)
        ax.set_title(f"{p['name']}: {p['meaning']}", fontsize=8)
    axes[0, 0].set_ylabel("median ATE / final setting")
    axes[1, 0].set_ylabel("median ATE / final setting")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Sensitivity of GEOK2Z_MF: one threshold changed at a time (dashed = final value, grey band = +-20 %)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    rep = os.path.join(exp_dir(cfg), "report")
    os.makedirs(rep, exist_ok=True)
    fig.savefig(os.path.join(rep, "sensitivity.png"), dpi=120)
    print(f"\nfigure: {rep}/sensitivity.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
