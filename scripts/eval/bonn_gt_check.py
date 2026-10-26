#!/usr/bin/env python3
"""Which Bonn ground-truth correction is right? Compare the three options on finished runs.

Usage (from ~/ORB_SLAM3):  python3 scripts/eval/bonn_gt_check.py scripts/eval/configs/bonn_baseline.yaml

For every Bonn run: ATE and RPE rotation (deg per second) with the ground truth used
  none         as in groundtruth.txt (marker pose, axis-swapped)
  bonn_tm      camera = T_ros * G * T_ros * T_m   (formula on the dataset page)
  bonn_tm_inv  the same with T_m inverted (in case the calibration is the other direction)
The right option makes the rotation error of good runs small (a few deg/s, like on TUM) - the
wrong ones leave a constant mismatch of tens of degrees. Set it in the config:  bonn_gt: <option>
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, exp_dir, short  # noqa: E402
import tum_metrics as tm  # noqa: E402

KINDS = ("none", "bonn_tm", "bonn_tm_inv")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    root = exp_dir(cfg)
    res = {k: {"ate": [], "rot": []} for k in KINDS}
    print(f"{'sequence':28s} {'variant':12s} " + " ".join(f"{k + ' ATE/rot':>22s}" for k in KINDS))
    for seq in sorted(os.listdir(root)):
        if not seq.startswith("bonn_"):
            continue
        gt = os.path.join(seq_dir(cfg, seq), "groundtruth.txt")
        for var in sorted(os.listdir(os.path.join(root, seq))):
            vd = os.path.join(root, seq, var)
            runs = sorted(r for r in os.listdir(vd) if os.path.exists(os.path.join(vd, r, "traj.txt")))[:2]
            for rd in runs:
                traj = os.path.join(vd, rd, "traj.txt")
                cells = []
                for k in KINDS:
                    a, _ = tm.ate(gt, traj, cfg["max_time_diff_s"], gt_kind=k)
                    _, r = tm.rpe(gt, traj, 1.0, "s", cfg["max_time_diff_s"], gt_kind=k)
                    res[k]["ate"].append(a["rmse"])
                    res[k]["rot"].append(r["rmse"])
                    cells.append(f"{a['rmse']:.4f} / {r['rmse']:6.2f}")
                print(f"{short(seq):28s} {var:12s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\nmedian over all runs: " + "   ".join(
        f"{k}: ATE {st.median(v['ate']):.4f} m, RPE rot {st.median(v['rot']):.2f} deg/s" for k, v in res.items() if v["ate"]))


if __name__ == "__main__":
    main()
