#!/usr/bin/env python3
"""Offline test of the keyframe-switch options on the per-frame CSVs from run_v16.sh (PHASE=diag).

Usage (from ~/ORB_SLAM3):
    python3 scripts/geo_switch_sim.py results_v16_diag

For every sequence it shows, in frames where YOLO sees something:
  - how many matched map points lie inside the boxes
  - the outlier ratio INSIDE the boxes vs OUTSIDE (people who walk -> high inside, low outside;
    a false detection on a static object -> about the same inside and outside)
Then it replays the switch (hold 30 frames) for several settings and prints the share of frames
with the cap switched on. Good settings: high on the walking sequences, near 0 on the fr1 ones.
"""
import csv
import glob
import os
import sys

import numpy as np

HOLD, YMIN = 30, 0.05


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if None in r or any(v is None or v.strip() == "" for v in r.values()):
                continue
            rows.append({k: float(v) for k, v in r.items()})
    if not rows:
        return None
    g = lambda k: np.array([r[k] for r in rows])
    d = {"yf": g("yolo_frac"), "m": g("matched"), "o": g("outliers"), "oib": g("outliers_in_box"), "iib": g("inliers_in_box")}
    d["nbox"] = d["oib"] + d["iib"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["rin"] = np.where(d["nbox"] > 0, d["oib"] / d["nbox"], np.nan)
        nout = d["m"] - d["nbox"]
        d["rout"] = np.where(nout > 0, (d["o"] - d["oib"]) / nout, 0.0)
    return d


def simulate(d, persist=1, geo=0.0, rel=0.0, pts=10, keep=0, minconf=1):
    hit = d["yf"] >= YMIN
    last, lastgeo, streak, on = -10 ** 9, -10 ** 9, 0, 0
    confs = []
    for i in range(len(hit)):
        streak = streak + 1 if hit[i] else 0
        fire = hit[i] and streak >= persist
        if fire and geo > 0:
            conf = d["nbox"][i] >= pts and d["rin"][i] >= geo and d["rin"][i] >= d["rout"][i] + rel
            confs = [c for c in confs if i - c <= keep]
            recent = keep > 0 and i - lastgeo <= keep and len(confs) >= minconf
            if conf:
                lastgeo = i
                confs.append(i)
            fire = conf or recent
        if fire:
            last = i
        on += (i - last) <= HOLD
    return 100.0 * on / len(hit)


def kind(seq):
    if "walking" in seq:
        return "people"
    if "sitting" in seq:
        return "sitting"
    return "static"


def main(root):
    data = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "DIAG_*_1_diag.csv"))):
        seq, var = os.path.basename(os.path.dirname(p)), os.path.basename(p).split("_1_diag")[0]
        d = load(p)
        if d is not None:
            data[(seq, var)] = d
    if not data:
        print("no DIAG_*_1_diag.csv files under", root)
        return

    print("=== Frames where YOLO sees something (yolo_frac >= 5 %)")
    print(f"{'sequence':22s} {'run':8s} {'yolo %':>7s} {'pts in box':>10s} {'<10 pts':>8s}"
          f" {'outl. inside':>13s} {'outl. outside':>14s}")
    for (seq, var), d in sorted(data.items()):
        h = d["yf"] >= YMIN
        if not h.any():
            print(f"{seq:22s} {var:8s} {0:7.1f}")
            continue
        ok = h & (d["nbox"] >= 10)
        rin = d["rin"][ok]
        q = lambda a: f"{np.median(a):.2f} [{np.percentile(a, 25):.2f}-{np.percentile(a, 75):.2f}]" if len(a) else "-"
        print(f"{seq:22s} {var:8s} {100 * h.mean():7.1f} {np.median(d['nbox'][h]):10.0f} {100 * (~ok[h]).mean():7.0f}%"
              f" {q(rin):>13s} {q(d['rout'][ok]):>14s}")
    print("  (median [25-75 %] over those frames; outlier ratio = share of matched map points the pose")
    print("   optimisation rejected. Walking people should be clearly higher INSIDE than OUTSIDE.)")

    configs = [("yolo only (SW_G10)", 1, 0.0, 0.0, 0, 1), ("persist 3", 3, 0.0, 0.0, 0, 1)]
    for g in (0.3, 0.4, 0.5):
        for r in (0.1, 0.2, 0.3):
            configs.append((f"geo {g:.1f} rel {r:.1f}", 1, g, r, 0, 1))
    for k in (90, 150):
        for m in (1, 2, 3, 5):
            configs.append((f"geo 0.3 rel 0.2 keep {k} conf {m}", 1, 0.3, 0.2, k, m))

    for var in sorted({v for _, v in data}):
        seqs = sorted({s for s, v in data if v == var}, key=lambda s: (kind(s), s))
        print(f"\n=== Replay on {var} runs: % of frames with the cap switched on (hold {HOLD})")
        print(f"{'setting':32s}" + "".join(f"{s.replace('freiburg1_', 'fr1_')[:13]:>14s}" for s in seqs) + f"{'score':>8s}")
        scored = []
        for name, pe, g, r, k, m in configs:
            on = {s: simulate(data[(s, var)], pe, g, r, keep=k, minconf=m) for s in seqs}
            ppl = [on[s] for s in seqs if kind(s) == "people"]
            st = [on[s] for s in seqs if kind(s) == "static"]
            score = (np.mean(ppl) if ppl else 0) - (np.mean(st) if st else 0)
            scored.append((score, name))
            print(f"{name:32s}" + "".join(f"{on[s]:14.1f}" for s in seqs) + f"{score:8.1f}")
        scored.sort(reverse=True)
        print("  best (mean on walking - mean on static fr1):", ", ".join(f"{n} ({s:.0f})" for s, n in scored[:3]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
