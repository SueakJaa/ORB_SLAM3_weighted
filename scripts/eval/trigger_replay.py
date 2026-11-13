#!/usr/bin/env python3
"""Offline replay of the keyframe switch with different geometric tests, on logged per-frame diagnostics.

Usage (from ~/ORB_SLAM3), any mix of folders that contain diag_frames.csv (DSLAM_DIAG=1 runs) or
the older results_v16_diag/<seq>/<VAR>_<i>_diag.csv files:
    python3 scripts/eval/trigger_replay.py results/diag_fr1room results/v19d_diag_xyz results_v16_diag

Why: on fr1_room the switch fires on a real but STANDING person (legs next to the cabinet, 25-29 s).
The box holds only ~10-20 matched points, fr1_room's outlier ratio outside the box is already 0.3-0.6
(fast motion, blur), so "inside >= 0.3 and >= outside + 0.2" is passed by chance in a few frames, the
cap then blocks keyframes at the hardest part of the sequence and tracking is lost for ~3 s.
A fixed margin ignores how many points the ratio is based on. The replacement tested here is a
one-sided two-proportion z-test: inside ratio significantly above the outside ratio, given the
point counts. At z=3.09 (outside ratio 0.45, 200 points outside) the needed margin is ~0.50 with
10 points in the box and ~0.19 with 100.

The replay re-runs the exact switch state machine of Tracking.cc (YOLO hit >= 5 %, short hold 30,
geometric confirmations, long hold 150 after 2 confirmations) on the logged counts. It is exact up to
the first frame where the replayed switch differs from the logged one (after that the real run would
have taken other keyframes), so treat it as a screen, then confirm with real runs.

Output: % of frames with the switch on per sequence for each rule (mean over the files found), and a
check that the replay of the current rule reproduces the logged switch (agreement %).
Good rule: walking_* unchanged (high), people-free fr1_* near 0.
"""
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict, deque

import numpy as np

YMIN, HOLD, KEEP, MINCONF = 0.05, 30, 150, 2


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if k is not None})
            except (TypeError, ValueError):
                continue
    return rows if rows and "outliers_in_box" in rows[0] else None


def seq_of(path):
    for part in reversed(path.split(os.sep)):
        m = re.match(r"(?:rgbd_dataset_|rgbd_)?((?:freiburg\d_|bonn_)\w+)", part)
        if m:
            return m.group(1)
    return None


def z_two_prop(oin, nin, oout, nout):
    if nin <= 0 or nout <= 0:
        return 0.0
    p = (oin + oout) / (nin + nout)
    se = math.sqrt(max(p * (1 - p), 1e-9) * (1.0 / nin + 1.0 / nout))
    return (oin / nin - oout / nout) / se


def make_rules():
    def margin(geo, rel, pts):
        return lambda oin, nin, oout, nout: nin >= pts and oin / nin >= geo and oin / nin >= oout / max(nout, 1) + rel

    def ztest(geo, zc, pts):
        return lambda oin, nin, oout, nout: nin >= pts and oin / nin >= geo and z_two_prop(oin, nin, oout, nout) >= zc

    return [("current  (0.3, +0.2, >=10 pts)", margin(0.3, 0.2, 10)),
            ("margin, >=20 pts", margin(0.3, 0.2, 20)),
            ("margin, >=30 pts", margin(0.3, 0.2, 30)),
            ("margin +0.3", margin(0.3, 0.3, 10)),
            ("z >= 1.64 (p<0.05)", ztest(0.3, 1.64, 10)),
            ("z >= 2.33 (p<0.01)", ztest(0.3, 2.33, 10)),
            ("z >= 3.09 (p<0.001)", ztest(0.3, 3.09, 10)),
            ("z >= 2.33, >=20 pts", ztest(0.3, 2.33, 20))]


def replay(rows, rule):
    last_dyn, last_geo, streak = -10 ** 9, -10 ** 9, 0
    confs = deque()
    on = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        fid = int(r.get("frame", i))
        hit = r["yolo_frac"] >= YMIN
        streak = streak + 1 if hit else 0
        while confs and fid - confs[0] > KEEP:
            confs.popleft()
        recent = fid - last_geo <= KEEP and len(confs) >= MINCONF
        if hit and recent:
            last_dyn = fid
        active = fid - last_dyn <= HOLD
        nin = r["outliers_in_box"] + r["inliers_in_box"]
        if hit and nin > 0:
            oin = r["outliers_in_box"]
            nout = r["matched"] - nin
            oout = r["outliers"] - oin
            if rule(oin, nin, oout, nout):
                last_dyn = last_geo = fid
                confs.append(fid)
                active = True
        on[i] = active
    return on


def kind(seq):
    return "walking" if "walking" in seq else "sitting" if "sitting" in seq else \
        "people" if ("person" in seq or seq.startswith("bonn_")) else "static"


def main(paths):
    files = []
    for root in paths:
        files += glob.glob(os.path.join(root, "**", "diag_frames.csv"), recursive=True)
        files += glob.glob(os.path.join(root, "**", "*_diag.csv"), recursive=True)
    data = defaultdict(list)
    for f in sorted(set(files)):
        s = seq_of(os.path.abspath(f))
        rows = load(f) if s else None
        if rows:
            data[s].append((f, rows))
    if not data:
        sys.exit("no per-frame CSVs with outlier columns found under " + ", ".join(paths))
    seqs = sorted(data, key=lambda s: (["walking", "sitting", "people", "static"].index(kind(s)), s))
    rules = make_rules()

    # does the replay of the current rule reproduce what the runs logged? (only runs that used it)
    agree = []
    for s in seqs:
        for f, rows in data[s]:
            if "GEOK2" in f and "kf_rule_active" in rows[0]:
                logged = np.array([r["kf_rule_active"] >= 1 for r in rows])
                agree.append(100 * np.mean(replay(rows, rules[0][1]) == logged))
    if agree:
        print(f"replay check on {len(agree)} GEOK2 runs: current rule reproduces the logged switch in "
              f"{np.mean(agree):.1f} % of frames (min {min(agree):.1f} %)\n")

    short = lambda s: s.replace("freiburg1_", "fr1_").replace("freiburg2_", "fr2_").replace("freiburg3_", "")[:14]
    print("% of frames with the switch on (mean over files; n = number of logged runs)")
    print(f"{'rule':32s}" + "".join(f"{short(s):>15s}" for s in seqs))
    print(f"{'':32s}" + "".join(f"{kind(s) + ' n=' + str(len(data[s])):>15s}" for s in seqs))
    for name, rule in rules:
        line = f"{name:32s}"
        for s in seqs:
            line += f"{np.mean([100 * replay(rows, rule).mean() for _, rows in data[s]]):15.1f}"
        print(line)
    print("\nwant: walking columns stay as high as 'current'; 'static' columns (nobody walks) go to ~0.")
    print("sitting / people columns: people present but mostly not walking - lower is fine.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
