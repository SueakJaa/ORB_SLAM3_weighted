#!/usr/bin/env python3
"""Summarise per-frame diagnostics written with DSLAM_DIAG=1.

Usage:
    python3 scripts/analyze_diag.py results_v12/walking_xyz/DIAG_A0_1_diag.csv [more.csv ...]

For each file it prints:
  - how many outliers there are per frame and how many lie inside YOLO boxes
  - how often a keyframe is created, and how many new map points each one adds
    (and how many of those lie inside YOLO boxes, i.e. on people)
  - the same split into frames WITH people in view vs WITHOUT
  - a per-second timeline of outliers and keyframes, to line up with the ATE error plot
"""
import csv
import sys
from collections import defaultdict

PEOPLE_FRAC = 0.05   # a frame "has people" if >= 5 % of its keypoints lie in YOLO boxes


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def block(name, rows):
    if not rows:
        print(f"  {name}: no frames")
        return
    n = len(rows)
    out = sum(r["outliers"] for r in rows)
    out_box = sum(r["outliers_in_box"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    kfs = [r for r in rows if r["kf_created"]]
    new_mp = sum(r["new_mp"] for r in kfs)
    new_mp_box = sum(r["new_mp_in_box"] for r in kfs)
    print(f"  {name}: {n} frames")
    print(f"    outliers / frame          {mean([r['outliers'] for r in rows]):7.1f}"
          f"   ({pct(out, matched):.1f} % of matched map points)")
    print(f"    outliers inside YOLO box  {pct(out_box, out):7.1f} %")
    print(f"    keyframes created         {len(kfs):7d}   (one every {n / max(len(kfs), 1):.1f} frames)")
    print(f"    new map points / keyframe {mean([r['new_mp'] for r in kfs]):7.1f}"
          f"   ({pct(new_mp_box, new_mp):.1f} % inside YOLO boxes)")
    print(f"    tracked / non-tracked close points (mean)"
          f"  {mean([r['tracked_close'] for r in rows if r['tracked_close'] >= 0]):.0f}"
          f" / {mean([r['nontracked_close'] for r in rows if r['nontracked_close'] >= 0]):.0f}")


def main(paths):
    for path in paths:
        with open(path, newline="") as f:
            rows = []
            for r in csv.DictReader(f):
                if None in r or any(v is None or v.strip() == '' for v in r.values()):
                    continue  # skip incomplete lines
                if None in r or any(v is None or v.strip() == '' for v in r.values()):
                    continue  # skip incomplete lines
                rr = {k: float(v) for k, v in r.items()}
                for k in ("matched", "outliers", "outliers_in_box", "inliers_in_box", "new_mp",
                          "new_mp_in_box", "tracked_close", "nontracked_close"):
                    rr[k] = int(rr[k])
                rr["kf_created"] = int(rr["kf_created"]) == 1
                rows.append(rr)
        if not rows:
            print(f"{path}: empty")
            continue

        t0 = rows[0]["t"]
        print(f"\n=== {path}")
        yf = [r["yolo_frac"] for r in rows]
        print(f"  YOLO: keypoints in boxes mean {100*mean(yf):.1f} %, frames >= 5 %: {pct(sum(1 for v in yf if v >= 0.05), len(yf)):.1f} %, frames with any box: {pct(sum(1 for v in yf if v > 0), len(yf)):.1f} %")
        yf = [r["yolo_frac"] for r in rows]
        print(f"  YOLO: keypoints in boxes mean {100*mean(yf):.1f} %, frames >= 5 %: {pct(sum(1 for v in yf if v >= 0.05), len(yf)):.1f} %, frames with any box: {pct(sum(1 for v in yf if v > 0), len(yf)):.1f} %")
        block("all frames", rows)
        block(f"frames WITH people (yolo_frac >= {PEOPLE_FRAC})", [r for r in rows if r["yolo_frac"] >= PEOPLE_FRAC])
        block("frames WITHOUT people", [r for r in rows if r["yolo_frac"] < PEOPLE_FRAC])

        print("  timeline (per second): t | outliers/frame | % in box | keyframes | new pts | % new pts in box")
        sec = defaultdict(list)
        for r in rows:
            sec[int(r["t"] - t0)].append(r)
        for s in sorted(sec):
            rs = sec[s]
            out = sum(r["outliers"] for r in rs)
            kfs = [r for r in rs if r["kf_created"]]
            nm = sum(r["new_mp"] for r in kfs)
            print(f"    {s:3d}s | {mean([r['outliers'] for r in rs]):6.1f} | {pct(sum(r['outliers_in_box'] for r in rs), out):5.1f} %"
                  f" | {len(kfs):3d} | {nm:5d} | {pct(sum(r['new_mp_in_box'] for r in kfs), nm):5.1f} %")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
