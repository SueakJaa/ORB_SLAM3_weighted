#!/usr/bin/env python3
"""Compare this pipeline's ATE with evo's on real trajectories, and explain any difference.

Usage (from ~/ORB_SLAM3):
    python3 scripts/eval/check_vs_evo.py CONFIG [--seq freiburg3_walking_halfsphere] [--variant SW_G10]

For every run: ATE RMSE from tum_metrics.py and from the evo Python API (same settings: |dt| <= 0.01 s,
SE(3) alignment), number of matched poses in each, and simple checks of the trajectory file
(duplicate or unsorted timestamps, gaps). Prints only runs that differ by more than 0.1 mm, then a total.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, exp_dir, short  # noqa: E402
import tum_metrics as tm  # noqa: E402


def evo_ate(gt, est, max_diff):
    from evo.core import sync, metrics
    from evo.tools import file_interface
    r = file_interface.read_tum_trajectory_file(gt)
    e = file_interface.read_tum_trajectory_file(est)
    r, e = sync.associate_trajectories(r, e, max_diff=max_diff)
    e.align(r, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((r, e))
    return ape.get_statistic(metrics.StatisticsType.rmse), e.num_poses


def file_checks(path):
    t = []
    with open(path) as f:
        for line in f:
            if line.strip() and line[0] != "#":
                t.append(float(line.split()[0]))
    t = np.array(t)
    d = np.diff(t)
    return {"poses": len(t), "unsorted": int((d < 0).sum()), "duplicates": int((d == 0).sum()),
            "max_gap_s": float(d.max()) if len(d) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seq", action="append")
    ap.add_argument("--variant", action="append")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        import evo
        print("evo version", evo.__version__)
    except ImportError:
        sys.exit("evo is not importable from this python3 (pip install evo)")
    root = exp_dir(cfg)
    n = bad = 0
    worst = 0.0
    for seq in sorted(os.listdir(root)):
        if not seq.startswith("freiburg") or (a.seq and seq not in a.seq):
            continue
        gt = os.path.join(seq_dir(cfg, seq), "groundtruth.txt")
        for var in sorted(os.listdir(os.path.join(root, seq))):
            if a.variant and var not in a.variant:
                continue
            vd = os.path.join(root, seq, var)
            for rd in sorted(os.listdir(vd)):
                traj = os.path.join(vd, rd, "traj.txt")
                if not os.path.exists(traj):
                    continue
                mine, info = tm.ate(gt, traj, cfg["max_time_diff_s"])
                try:
                    ev, ev_n = evo_ate(gt, traj, cfg["max_time_diff_s"])
                except Exception as ex:  # evo refuses some files (e.g. unsorted stamps)
                    ev, ev_n = float("nan"), 0
                    print(f"  evo failed on {short(seq)} {var} {rd}: {ex}")
                n += 1
                diff = abs(mine["rmse"] - ev)
                worst = max(worst, diff if diff == diff else 0)
                old = os.path.join(vd, rd, "log.txt")
                if not (diff <= 1e-4):
                    bad += 1
                    fc = file_checks(traj)
                    print(f"{short(seq):22s} {var:12s} {rd}: mine {mine['rmse']:.4f} ({mine['n']} matched) | "
                          f"evo {ev:.4f} ({ev_n} matched) | file: {fc['poses']} poses, unsorted {fc['unsorted']}, "
                          f"duplicate stamps {fc['duplicates']}, largest gap {fc['max_gap_s']:.2f} s")
    print(f"\n{n} runs compared, {bad} differ by more than 0.1 mm (largest difference {worst:.4f} m)")
    if bad:
        print("Expected where the file has duplicate stamps: ORB-SLAM3 copies the last pose and timestamp while\n"
              "RECENTLY_LOST. This pipeline scores each timestamp once (like the TUM tools) and reports the copies\n"
              "as lost frames; evo scores every copy. Runs without duplicate stamps must match exactly.")


if __name__ == "__main__":
    main()
