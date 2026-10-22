#!/usr/bin/env python3
"""Compare per-frame diagnostics of variants (runs made with DSLAM_DIAG=1), e.g. with and without map filter.

Usage (from ~/ORB_SLAM3), after run_experiments.py and evaluate.py:
    python3 scripts/eval/diag_compare.py scripts/eval/configs/v19d_diag_xyz.yaml

Per sequence and variant (mean over runs):
  ATE            median ATE RMSE of the runs [m]
  dyn%           share of keypoints the map filter keeps out of the map, in frames with the switch on
  filt/KF        close points not mapped per new keyframe because of the filter (Tracking)
  newMP/KF       close points mapped per new keyframe
  tri filt%      share of LocalMapping triangulation candidates rejected by the filter
  map pts        map points at the end of the sequence
  inl med / p10  tracked map points per frame (median / 10th percentile)
  inl<100%       frames with fewer than 100 tracked points (thin tracking)
Figure report/diag_<seq>.png: tracked points, map size and ATE over time for the median run of each variant.
"""
import csv
import json
import os
import re
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, exp_dir, short  # noqa: E402
import tum_metrics as tm  # noqa: E402
from report import SERIES, style, MUTED  # noqa: E402


def read_diag(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
    return rows


def run_summary(rd):
    d = read_diag(os.path.join(rd, "diag_frames.csv"))
    if not d:
        return None
    on = [r for r in d if r["kf_rule_active"] >= 1]
    kfs = [r for r in d if r["kf_created"] >= 1]
    ok = [r["inliers_track"] for r in d if r["state"] == 2]
    s = {"dyn": 100 * st.mean(r["dyn_map_frac"] for r in on) if on else 0.0,
         "filt_kf": st.mean(r["mp_filtered"] for r in kfs) if kfs else 0.0,
         "new_kf": st.mean(r["new_mp"] for r in kfs) if kfs else 0.0,
         "map_pts": d[-1]["map_points"], "inl_med": float(np.median(ok)) if ok else 0.0,
         "inl_p10": float(np.percentile(ok, 10)) if ok else 0.0,
         "thin": 100 * sum(1 for v in ok if v < 100) / len(ok) if ok else 0.0, "rows": d, "tri": None}
    with open(os.path.join(rd, "meta.json")) as f:
        stats = json.load(f).get("dynamic_stats", "")
    m = re.search(r"triangulation_candidates=(\d+) triangulation_filtered=(\d+)", stats)
    if m and int(m.group(1)):
        s["tri"] = 100 * int(m.group(2)) / int(m.group(1))
    return s


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    root = exp_dir(cfg)
    metrics = {}
    mp = os.path.join(root, "metrics.csv")
    if os.path.exists(mp):
        for r in csv.DictReader(open(mp)):
            if r["status"] == "ok":
                metrics[(r["sequence"], r["variant"], int(r["run"]))] = float(r["ate_rmse"])
    variants = list(cfg["variants"])
    rep = os.path.join(root, "report")
    os.makedirs(rep, exist_ok=True)
    print(f"{'sequence':20s} {'variant':14s} {'ATE':>7s} {'dyn%':>6s} {'filt/KF':>8s} {'newMP/KF':>9s} "
          f"{'tri filt%':>9s} {'map pts':>8s} {'inl med/p10':>12s} {'inl<100%':>9s}")
    plots = {}
    for s_ in cfg["sequences"]:
        for v in variants:
            vd = os.path.join(root, s_, v)
            if not os.path.isdir(vd):
                continue
            sums = []
            for rd in sorted(os.listdir(vd)):
                p = os.path.join(vd, rd)
                if os.path.exists(os.path.join(p, "diag_frames.csv")):
                    x = run_summary(p)
                    if x:
                        x["run"] = int(rd.split("_")[1])
                        x["ate"] = metrics.get((s_, v, x["run"]))
                        sums.append(x)
            if not sums:
                continue
            m = lambda k: st.mean(x[k] for x in sums if x[k] is not None) if any(x[k] is not None for x in sums) else float("nan")
            ates = [x["ate"] for x in sums if x["ate"] is not None]
            print(f"{short(s_):20s} {v:14s} {st.median(ates) if ates else float('nan'):7.3f} {m('dyn'):6.1f} "
                  f"{m('filt_kf'):8.1f} {m('new_kf'):9.1f} {m('tri'):9.1f} {m('map_pts'):8.0f} "
                  f"{m('inl_med'):5.0f} / {m('inl_p10'):4.0f} {m('thin'):9.1f}")
            withate = sorted((x for x in sums if x["ate"] is not None), key=lambda x: x["ate"])
            if withate:
                plots.setdefault(s_, []).append((v, withate[(len(withate) - 1) // 2]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    style(plt)
    color = {v: SERIES[i % len(SERIES)] for i, v in enumerate(variants)}
    for s_, lst in plots.items():
        fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        gt = os.path.join(seq_dir(cfg, s_), "groundtruth.txt")
        for v, x in lst:
            d = x["rows"]
            t0 = d[0]["t"]
            t = [r["t"] - t0 for r in d]
            lab = f"{v} (run {x['run']}, ATE {x['ate']:.3f} m)"
            axes[0].plot(t, [r["inliers_track"] for r in d], color=color[v], linewidth=1.2, label=lab)
            axes[1].plot(t, [r["map_points"] for r in d], color=color[v], linewidth=1.5)
            traj = os.path.join(root, s_, v, f"run_{x['run']:02d}", "traj.txt")
            _, det = tm.ate(gt, traj, cfg["max_time_diff_s"])
            axes[2].plot(det["t"] - t0, det["err"], color=color[v], linewidth=1.5)
        axes[0].axhline(30, color=MUTED, linestyle="--", linewidth=1)
        axes[0].set_ylabel("tracked map points")
        axes[1].set_ylabel("map points in map")
        axes[2].set_ylabel("position error [m]")
        axes[2].set_xlabel("time [s]")
        axes[0].set_title(f"{short(s_)} - median run per variant (dashed: ORB-SLAM3 loses tracking below 30)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8, loc="lower center", ncol=1)
        fig.tight_layout(rect=(0, 0.04 + 0.03 * len(l), 1, 1))
        fig.savefig(os.path.join(rep, f"diag_{s_}.png"))
        plt.close(fig)
    print(f"figures: {rep}/diag_<sequence>.png")


if __name__ == "__main__":
    main()
