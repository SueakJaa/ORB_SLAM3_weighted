#!/usr/bin/env python3
"""Was a detected loop real? Ground-truth distance and view-angle difference between the two keyframes
a loop linked (from the "[Loop] current KF .. t=..  matched KF .. t=.." lines in run logs).

Usage (from ~/ORB_SLAM3):
    grep -h "\\[Loop\\]" results/itemb_loop/bonn_moving_obstructing_box2/*/run_*/log.txt | \\
        python3 scripts/itemb/loop_check.py ~/datasets/rgbd_bonn_moving_obstructing_box2
A real loop: the camera was really back near the same place, looking the same way (distance ~< 0.5 m,
angle ~< 30 deg). Far apart = the system took something it saw twice (a moving object) for a revisited place.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
import tum_metrics as tm  # noqa: E402

d = sys.argv[1]
t, p, q = tm.load_tum(os.path.join(d, "groundtruth.txt"))
p, q = tm.gt_transform(p, q, "bonn_tm" if "bonn" in d else None)
T = tm.to_se3(p, q)
t_start = float(next(l for l in open(os.path.join(d, "depth.txt")) if l.strip() and l[0] != "#").split()[0])


def at(ts):
    return T[int(np.argmin(np.abs(t - ts)))]


print(f"{'current t':>10s} {'matched t':>10s} {'GT distance m':>14s} {'view angle deg':>15s}  verdict")
for line in sys.stdin:
    m = re.search(r"current KF \d+ t=([\d.]+)\s+matched KF \d+ t=([\d.]+)", line)
    if not m:
        continue
    a, b = float(m.group(1)), float(m.group(2))
    A, B = at(a), at(b)
    dist = np.linalg.norm(A[:3, 3] - B[:3, 3])
    za, zb = A[:3, 2], B[:3, 2]   # viewing directions
    ang = np.degrees(np.arccos(np.clip(za @ zb, -1, 1)))
    verdict = "real place revisit" if dist < 0.5 and ang < 30 else "FALSE (different place or view)"
    print(f"{a - t_start:10.1f} {b - t_start:10.1f} {dist:14.2f} {ang:15.1f}  {verdict}")
