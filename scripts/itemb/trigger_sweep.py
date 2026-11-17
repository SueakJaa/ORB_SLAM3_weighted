#!/usr/bin/env python3
"""Item B: try trigger rules on the per-frame files of depth_motion_scan.py (no rescanning).

A frame "fires" if the largest region of arrived pixels is >= AREA % of the valid pixels and focus >= FOCUS.
The trigger turns ON only after PERSIST scanned frames in a row fire (like the 2 confirmations of the
YOLO trigger), so one-off false regions are dropped while a moving object, which keeps moving, stays.

Usage (from ~/ORB_SLAM3):
    python3 scripts/itemb/trigger_sweep.py results/itemb_dev/itemb_gt
    python3 scripts/itemb/trigger_sweep.py results/itemb_dev/itemb_GEOK2Z_MF      (poses from the system)
Prints, per rule, the % of scanned frames with the trigger on, for every sequence found. Rules are only
judged on the development sequences; nothing here touches the Bonn test set.
Want: 'static' columns (nobody moves: fr1_room, fr1_desk) ~ 0; the others as high as possible.
"""
import csv
import glob
import os
import sys

import numpy as np

AREAS = [1, 2, 3, 5, 8]
PERSIST = [1, 2, 3, 5]
FOCUS = 0.5
STATIC = ("freiburg1_room", "freiburg1_desk")


def load(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    return np.array([float(x["largest"]) * 100 for x in r]), np.array([float(x["focus"]) for x in r])


def on_fraction(lg, fc, area, persist):
    hit = (lg >= area) & (fc >= FOCUS)
    run, on = 0, np.zeros(len(hit), bool)
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        on[i] = run >= persist
    return 100 * on.mean()


def main(folder):
    files = sorted(glob.glob(os.path.join(folder, "frames_*.csv")))
    if not files:
        sys.exit(f"no frames_*.csv in {folder} - run depth_motion_scan.py first")
    seqs = [os.path.basename(f)[7:-4] for f in files]
    data = [load(f) for f in files]
    name = lambda s: (s.replace("freiburg1_", "fr1_").replace("freiburg3_", "").replace("bonn_", "")
                      .replace("moving_obstructing_box2", "obst_box2"))[:12]
    order = sorted(range(len(seqs)), key=lambda i: (seqs[i] in STATIC, seqs[i]))
    print(f"{folder}\n% of scanned frames with the trigger ON (focus >= {FOCUS})")
    print(f"{'area% persist':>14s}" + "".join(f"{name(seqs[i]):>13s}" for i in order) + f"{'static max':>12s}")
    for area in AREAS:
        for p in PERSIST:
            vals = [on_fraction(*data[i], area, p) for i in order]
            stat = max([v for i, v in zip(order, vals) if seqs[i] in STATIC] or [float("nan")])
            print(f"{area:>6d} {p:>7d}" + "".join(f"{v:13.1f}" for v in vals) + f"{stat:12.1f}")
    print("\nstatic = fr1_room / fr1_desk (nobody moves). A good rule keeps 'static max' near 0 and the rest high.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
