#!/usr/bin/env python3
"""Compute ATE and RPE for every run of an experiment and write one table: <experiment>/metrics.csv

Usage (from ~/ORB_SLAM3):
    python3 scripts/eval/evaluate.py scripts/eval/configs/baseline_tum_dynamic.yaml

One row per run (crashed runs included, with status=crash), columns:
  sequence, variant, run, status, ate_rmse, ate_mean, ate_median, ate_max, coverage,
  tracked_pct (frames with a real pose estimate / frames given), lost_frames (ORB-SLAM3 RECENTLY_LOST
  copies: repeated timestamps, not scored), rpe_t_rmse [m per delta], rpe_r_rmse [deg per delta],
  keyframes, median_tracking_ms, switch_on_pct
ATE = SE(3)-aligned, identical to `evo_ape tum gt est -a`. RPE delta from the config (default 1 s).
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, exp_dir  # noqa: E402
import tum_metrics as tm  # noqa: E402

FIELDS = ["sequence", "variant", "run", "status", "ate_rmse", "ate_mean", "ate_median", "ate_max", "coverage",
          "tracked_pct", "lost_frames",
          "rpe_t_rmse", "rpe_r_rmse", "keyframes", "median_tracking_ms", "switch_on_pct", "cap_blocked", "cap_bypassed", "note"]


_N_IMG = {}


def n_images(cfg, seq):
    """Number of RGB-D frames given to SLAM (lines of the association file)."""
    if seq not in _N_IMG:
        p = os.path.join(seq_dir(cfg, seq), "association_new.txt")
        _N_IMG[seq] = sum(1 for l in open(p) if l.strip() and l[0] != "#") if os.path.exists(p) else None
    return _N_IMG[seq]


def evaluate_run(cfg, seq, var, rd):
    row = {"sequence": seq, "variant": var, "run": int(rd.rsplit("_", 1)[1]), "status": "ok", "note": ""}
    meta = {}
    if os.path.exists(os.path.join(rd, "meta.json")):
        with open(os.path.join(rd, "meta.json")) as f:
            meta = json.load(f)
    if meta.get("median_tracking_s") is not None:
        row["median_tracking_ms"] = round(1000 * meta["median_tracking_s"], 1)
    m = re.search(r"switch_on_frames=\d+ \(([0-9.]+) %\)", meta.get("dynamic_stats", ""))
    if m:
        row["switch_on_pct"] = float(m.group(1))
    for k in ("cap_blocked", "cap_bypassed"):
        m = re.search(rf"{k}=(\d+)", meta.get("dynamic_stats", ""))
        if m:
            row[k] = int(m.group(1))
    kf = os.path.join(rd, "kf.txt")
    if os.path.exists(kf):
        with open(kf) as f:
            row["keyframes"] = sum(1 for l in f if l.strip() and l[0] != "#")
    traj = os.path.join(rd, "traj.txt")
    if not os.path.exists(traj):
        row["status"] = "crash"
        return row
    gt = os.path.join(seq_dir(cfg, seq), "groundtruth.txt")
    try:
        a, _ = tm.ate(gt, traj, cfg["max_time_diff_s"])
        t, r = tm.rpe(gt, traj, cfg["rpe_delta"], cfg["rpe_delta_unit"], cfg["max_time_diff_s"])
    except ValueError as e:
        row["status"], row["note"] = "eval_error", str(e)
        return row
    row.update({"ate_rmse": a["rmse"], "ate_mean": a["mean"], "ate_median": a["median"], "ate_max": a["max"],
                "coverage": round(a["coverage"], 3), "rpe_t_rmse": t["rmse"], "rpe_r_rmse": r["rmse"],
                "lost_frames": a["duplicates"]})
    n_img = n_images(cfg, seq)
    if n_img:
        row["tracked_pct"] = round(100.0 * a["estimated_poses"] / n_img, 1)
    if a["coverage"] < 0.9:
        row["note"] = f"only {100 * a['coverage']:.0f} % of the estimate matched ground truth"
    return row


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    root = exp_dir(cfg)
    if not os.path.isdir(root):
        sys.exit(f"no results yet in {root}")
    rows = []
    for seq in sorted(os.listdir(root)):
        sd = os.path.join(root, seq)
        if not os.path.isdir(sd) or not seq.startswith("freiburg"):
            continue
        for var in sorted(os.listdir(sd)):
            vd = os.path.join(sd, var)
            for rd in sorted(os.listdir(vd)) if os.path.isdir(vd) else []:
                if rd.startswith("run_"):
                    rows.append(evaluate_run(cfg, seq, var, os.path.join(vd, rd)))
    out = os.path.join(root, "metrics.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows)} runs evaluated ({len(bad)} without a usable trajectory) -> {out}")
    print(f"Next: python3 scripts/eval/report.py {sys.argv[1]}")


if __name__ == "__main__":
    main()
