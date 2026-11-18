#!/usr/bin/env python3
"""Item C: how much do the results move when ONE threshold of the final method changes?

Usage (from ~/ORB_SLAM3), after run_experiments.py + evaluate.py on scripts/itemc/itemc_sens.yaml:
    python3 scripts/itemc/sensitivity.py scripts/itemc/itemc_sens.yaml

Per parameter, a table: each tried value against the final setting (DEFAULT, marked *), median ATE per
sequence with failed runs in brackets, and whether it CHANGES the result.

What counts as a change. Each tried value has 3 runs, and a 3-run median of a noisy sequence moves a lot by
chance. If DEFAULT has >= 8 runs (run_experiments.py ... --runs 10 --variant DEFAULT), the noise band of a
3-run median is measured directly: the medians of EVERY group of 3 DEFAULT runs, 2.5-97.5th percentile,
widened by 1 mm. A value CHANGES a sequence if its 3-run median lies outside that band, or if it fails >= 2 of 3 runs while DEFAULT
fails at most a third of its runs. With fewer DEFAULT runs the old rule is used (+-20 % or 2 mm, no extra
failures). The "stable range" lists the values that change no sequence.
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
FLOOR_M = 0.001


def main(cfg_path):
    cfg = load_config(cfg_path)
    params = json.load(open(os.path.join(HERE, "params.json")))
    thr = cfg["fail_threshold_m"]
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(exp_dir(cfg), "metrics.csv"))):
        ok = r["status"] == "ok" and r["ate_rmse"]
        by[(r["variant"], r["sequence"])].append(float(r["ate_rmse"]) if ok else float("inf"))
    seqs = cfg["sequences"]
    name = lambda s: short(s).replace("fr3/", "").replace("bonn/", "bonn_")[:16]
    stat = lambda v, s: (st.median(by[(v, s)]), sum(1 for x in by[(v, s)] if x > thr)) if by.get((v, s)) else (None, None)
    from itertools import combinations
    import numpy as np
    band = {}
    for s in seqs:
        d = by.get(("DEFAULT", s), [])
        if len(d) >= 8:
            meds = [st.median(c) for c in combinations(d, 3)]
            band[s] = (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)), sum(1 for x in d if x > thr) / len(d))
    if band:
        print("noise band of a 3-run median of the final setting (2.5-97.5th percentile over all groups of 3 runs;\n"
              f"a difference below {1000 * FLOOR_M:.0f} mm is ignored as irrelevant; with this many comparisons about 1 in 20\n"
              "cells is flagged by chance, so a single flag on a noisy sequence is weak evidence):")
        for s in seqs:
            if s in band:
                print(f"   {name(s):18s} {band[s][0]:.4f} - {band[s][1]:.4f}   (fails {100 * band[s][2]:.0f} % of runs)")
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
                if s in band and v != p["default"]:
                    lo, hi, frate = band[s]
                    within = lo - FLOOR_M <= m <= hi + FLOOR_M
                    ok_all &= within and not (f >= 2 and frate <= 1 / 3)
                else:
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
        if not band:
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
    fig.suptitle("Sensitivity of GEOK2Z_MF: one threshold changed at a time (dashed = final value; "
                 + ("3-run medians, noise bands in the text output)" if band else "grey band = +-20 %)"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    rep = os.path.join(exp_dir(cfg), "report")
    os.makedirs(rep, exist_ok=True)
    fig.savefig(os.path.join(rep, "sensitivity.png"), dpi=120)
    print(f"\nfigure: {rep}/sensitivity.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
