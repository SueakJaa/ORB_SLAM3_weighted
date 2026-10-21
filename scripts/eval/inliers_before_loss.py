#!/usr/bin/env python3
"""How many map points are still tracked just before ORB-SLAM3 loses tracking? (to choose DSLAM_KF_CAP_BYPASS)

Usage (from ~/ORB_SLAM3):
    python3 scripts/eval/inliers_before_loss.py results_v16_diag

Reads the per-frame CSVs written with DSLAM_DIAG=1 (column inliers_track = tracked map points used for the
tracking test, state = 2 OK / 3 RECENTLY_LOST). For each sequence it prints the tracked-point count
  - in normal frames (state OK, not within 10 frames of a loss)
  - in the 1-5 frames right before tracking is lost
A good bypass threshold is above most "before loss" values but below most "normal" values.
"""
import csv
import glob
import os
import sys

import numpy as np


def main(root):
    print(f"{'sequence':24s} {'run':8s} {'losses':>6s} | {'normal frames: p10 / median':>28s} | "
          f"{'1-5 frames before loss: median / p90':>36s}")
    for p in sorted(glob.glob(os.path.join(root, "*", "DIAG_*_1_diag.csv"))):
        seq, run = os.path.basename(os.path.dirname(p)), os.path.basename(p).split("_1_diag")[0]
        st, inl = [], []
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    st.append(int(float(r["state"])))
                    inl.append(float(r["inliers_track"]))
                except (TypeError, ValueError, KeyError):
                    continue
        st, inl = np.array(st), np.array(inl)
        loss = [i for i in range(1, len(st)) if st[i] != 2 and st[i - 1] == 2]
        near = set()
        for i in loss:
            near.update(range(max(0, i - 10), i + 1))
        normal = [inl[i] for i in range(len(st)) if st[i] == 2 and i not in near]
        before = [inl[j] for i in loss for j in range(max(0, i - 5), i) if st[j] == 2]
        nm = f"{np.percentile(normal, 10):.0f} / {np.median(normal):.0f}" if normal else "-"
        bf = f"{np.median(before):.0f} / {np.percentile(before, 90):.0f}" if before else "-"
        print(f"{seq:24s} {run:8s} {len(loss):6d} | {nm:>28s} | {bf:>36s}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
