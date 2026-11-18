#!/usr/bin/env python3
"""Why is a sequence worse with the method? Does the keyframe switch fire where nothing moves?

Usage (from ~/ORB_SLAM3):
    # 1) quick look at runs you already have (reads meta.json + metrics.csv, no new runs needed)
    python3 scripts/eval/switch_diag.py scripts/eval/configs/thesis_final.yaml
    # 2) frame-by-frame look (needs runs made with DSLAM_DIAG=1, e.g. configs/diag_fr1room.yaml)
    python3 scripts/eval/switch_diag.py scripts/eval/configs/diag_fr1room.yaml

Part 1, every experiment - per sequence and variant:
  runs / failed   runs, and runs with ATE above fail_threshold_m (or crashed)
  ATE med         median ATE RMSE [m]
  det%            frames where YOLO found a person/vehicle box
  switch%         frames with the keyframe switch on (mean over runs, and min-max)
  tri filt%       LocalMapping triangulation candidates rejected by the map filter
  depth static%   (v21 depth test) share of depth-band points judged static and let into the map;
  unknown%        ... with no evidence (outside the old image, no depth there)
  and, for --seq (default freiburg1_room), one line per run, so failed runs can be compared with good ones.

Part 2, only runs with diag_frames.csv:
  episodes        times the switch turned on, with their length
  err on / off    mean position error while the switch is on vs off [m]
  lost            frames not tracked (state != OK) while on / off
  fail near sw    for a failed run: was the switch on in the 5 s before the error first passed the threshold?
  Figures in <experiment>/report/:
    switch_<seq>.png         position error over time for the worst run of each variant, switch-on time shaded
    switch_frames_<seq>.png  the camera image at the start of each switch episode (what YOLO saw)
"""
import argparse
import csv
import json
import os
import re
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, exp_dir, short, gt_kind  # noqa: E402
import tum_metrics as tm  # noqa: E402


def stats_of(rd):
    p = os.path.join(rd, "meta.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        s = json.load(f).get("dynamic_stats", "")
    out = {}
    m = re.search(r"switch_on_frames=(\d+) \(([\d.]+) %\)", s)
    if m:
        out["sw"] = float(m.group(2))
    m = re.search(r"frames_with_detections=(\d+) \(([\d.]+) %\)", s)
    if m:
        out["det"] = float(m.group(2))
    m = re.search(r"triangulation_candidates=(\d+) triangulation_filtered=(\d+)", s)
    if m and int(m.group(1)):
        out["tri"] = 100 * int(m.group(2)) / int(m.group(1))
    m = re.search(r"depth_test_candidates=(\d+) depth_static=(\d+) depth_moved=(\d+) depth_unknown=(\d+)", s)
    if m and int(m.group(1)):
        out["dstat"] = 100 * int(m.group(2)) / int(m.group(1))
        out["dunk"] = 100 * int(m.group(4)) / int(m.group(1))
    m = re.search(r"depth_trigger_hits=(\d+) depth_trigger_active=(\d+)", s)
    if m:
        out["dthits"], out["dtact"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"cap_blocked=(\d+)", s)
    if m:
        out["blocked"] = int(m.group(1))
    return out


def read_diag(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
    return rows


def episodes(rows):
    eps, start = [], None
    for i, r in enumerate(rows):
        on = r["kf_rule_active"] >= 1
        if on and start is None:
            start = i
        elif not on and start is not None:
            eps.append((start, i - 1))
            start = None
    if start is not None:
        eps.append((start, len(rows) - 1))
    return eps


def nearest_rgb(d, t):
    best, bt = None, 1e9
    with open(os.path.join(d, "rgb.txt")) as f:
        for line in f:
            if line[0] == "#" or not line.strip():
                continue
            ts, fn = line.split()[:2]
            if abs(float(ts) - t) < bt:
                best, bt = os.path.join(d, fn), abs(float(ts) - t)
    return best


def mean(xs):
    return st.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--seq", default="freiburg1_room", help="sequence to list run by run (default fr1_room)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = exp_dir(cfg)
    thr = cfg["fail_threshold_m"]
    metrics = {}
    mp = os.path.join(root, "metrics.csv")
    if not os.path.exists(mp):
        sys.exit(f"{mp} missing - run evaluate.py on this config first")
    for r in csv.DictReader(open(mp)):
        ate = float(r["ate_rmse"]) if r["status"] == "ok" and r["ate_rmse"] else None
        metrics[(r["sequence"], r["variant"], int(r["run"]))] = (r["status"], ate)

    # ---------------- part 1 ----------------
    print(f"\n=== Part 1: switch activity per sequence ({root}) ===")
    print(f"{'sequence':22s} {'variant':12s} {'runs':>4s} {'failed':>6s} {'ATE med':>8s} {'det%':>6s} "
          f"{'switch% mean (min-max)':>24s} {'tri filt%':>9s} {'depth static%':>13s} {'unknown%':>8s}")
    have_diag = False
    for s_ in cfg["sequences"]:
        for v in cfg["variants"]:
            vd = os.path.join(root, s_, v)
            if not os.path.isdir(vd):
                continue
            rs = []
            for rd in sorted(os.listdir(vd)):
                if not rd.startswith("run_"):
                    continue
                n = int(rd.split("_")[1])
                status, ate = metrics.get((s_, v, n), ("missing", None))
                x = stats_of(os.path.join(vd, rd))
                x.update(run=n, status=status, ate=ate,
                         failed=(status == "crash" or (ate is not None and ate > thr)))
                rs.append(x)
                have_diag |= os.path.exists(os.path.join(vd, rd, "diag_frames.csv"))
            if not rs:
                continue
            ates = [x["ate"] for x in rs if x["ate"] is not None]
            sws = [x["sw"] for x in rs if "sw" in x]
            sw = f"{mean(sws):6.1f} ({min(sws):.1f}-{max(sws):.1f})" if sws else "-"
            print(f"{short(s_):22s} {v:12s} {len(rs):4d} {sum(x['failed'] for x in rs):6d} "
                  f"{st.median(ates) if ates else float('nan'):8.3f} {mean([x['det'] for x in rs if 'det' in x]):6.1f} "
                  f"{sw:>24s} {mean([x['tri'] for x in rs if 'tri' in x]):9.1f} "
                  f"{mean([x['dstat'] for x in rs if 'dstat' in x]):13.1f} {mean([x['dunk'] for x in rs if 'dunk' in x]):8.1f}")
    if a.seq in cfg["sequences"]:
        print(f"\n--- {short(a.seq)} run by run ---")
        print(f"{'variant':12s} {'run':>3s} {'status':>6s} {'ATE':>7s} {'switch%':>8s} {'det%':>6s} {'cap_blocked':>11s} "
              f"{'depth trig on':>13s}")
        for v in cfg["variants"]:
            vd = os.path.join(root, a.seq, v)
            if not os.path.isdir(vd):
                continue
            for rd in sorted(x for x in os.listdir(vd) if x.startswith("run_")):
                n = int(rd.split("_")[1])
                status, ate = metrics.get((a.seq, v, n), ("missing", None))
                x = stats_of(os.path.join(vd, rd))
                flag = "  <- FAILED" if status == "crash" or (ate is not None and ate > thr) else ""
                print(f"{v:12s} {n:3d} {status:>6s} {ate if ate is not None else float('nan'):7.3f} "
                      f"{x.get('sw', float('nan')):8.1f} {x.get('det', float('nan')):6.1f} "
                      f"{x.get('blocked', -1):11d} {x.get('dtact', -1):13d}{flag}")
    if not have_diag:
        print("\n(no diag_frames.csv in this experiment - for part 2 run configs/diag_fr1room.yaml)")
        return

    # ---------------- part 2 ----------------
    print(f"\n=== Part 2: frame by frame (runs with DSLAM_DIAG=1) ===")
    print(f"{'sequence':22s} {'variant':12s} {'run':>3s} {'ATE':>7s} {'episodes':>8s} {'longest s':>9s} "
          f"{'err on':>7s} {'err off':>7s} {'lost on/off':>11s} {'fail near sw':>12s}")
    rep = os.path.join(root, "report")
    os.makedirs(rep, exist_ok=True)
    worst = {}
    for s_ in cfg["sequences"]:
        gt = os.path.join(seq_dir(cfg, s_), "groundtruth.txt")
        for v in cfg["variants"]:
            vd = os.path.join(root, s_, v)
            if not os.path.isdir(vd):
                continue
            for rd in sorted(x for x in os.listdir(vd) if x.startswith("run_")):
                p = os.path.join(vd, rd)
                dp = os.path.join(p, "diag_frames.csv")
                n = int(rd.split("_")[1])
                status, ate = metrics.get((s_, v, n), ("missing", None))
                if not os.path.exists(dp) or ate is None:
                    continue
                rows = read_diag(dp)
                if not rows:
                    continue
                t = np.array([r["t"] for r in rows])
                on = np.array([r["kf_rule_active"] >= 1 for r in rows])
                lost = np.array([r["state"] != 2 for r in rows])
                eps = episodes(rows)
                longest = max(((t[b] - t[a_]) for a_, b in eps), default=0.0)
                _, det = tm.ate(gt, os.path.join(p, "traj.txt"), cfg["max_time_diff_s"], gt_kind=gt_kind(cfg, s_))
                et, ee = np.asarray(det["t"]), np.asarray(det["err"])
                idx = np.clip(np.searchsorted(t, et), 0, len(t) - 1)
                e_on, e_off = ee[on[idx]], ee[~on[idx]]
                near = "-"
                if ate > thr:
                    bad = np.nonzero(ee > thr)[0]
                    if len(bad):
                        tf = et[bad[0]]
                        w = (t >= tf - 5) & (t <= tf)
                        near = f"{'YES' if on[w].any() else 'no'} @{tf - t[0]:.0f}s"
                print(f"{short(s_):22s} {v:12s} {n:3d} {ate:7.3f} {len(eps):8d} {longest:9.1f} "
                      f"{e_on.mean() if len(e_on) else float('nan'):7.3f} "
                      f"{e_off.mean() if len(e_off) else float('nan'):7.3f} "
                      f"{int((lost & on).sum()):5d}/{int((lost & ~on).sum()):<5d} {near:>12s}")
                if (s_, v) not in worst or ate > worst[(s_, v)]["ate"]:
                    worst[(s_, v)] = dict(ate=ate, run=n, t=t, on=on, lost=lost, et=et, ee=ee, eps=eps)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        print("matplotlib missing - no figures")
        return
    for s_ in cfg["sequences"]:
        vs = [v for v in cfg["variants"] if (s_, v) in worst]
        if not vs:
            continue
        fig, axes = plt.subplots(len(vs), 1, figsize=(8, 2.4 * len(vs)), sharex=True, squeeze=False)
        for ax, v in zip(axes[:, 0], vs):
            w = worst[(s_, v)]
            t0 = w["t"][0]
            for a_, b in w["eps"]:
                ax.axvspan(w["t"][a_] - t0, w["t"][b] - t0, color="#f2a541", alpha=0.3, linewidth=0)
            ax.plot(w["et"] - t0, w["ee"], color="#2f5d8a", linewidth=1.3)
            lt = w["t"][w["lost"]] - t0
            if len(lt):
                ax.plot(lt, np.zeros_like(lt), "|", color="#c0392b", markersize=8)
            ax.axhline(thr, color="#888888", linestyle="--", linewidth=1)
            ax.set_ylabel("error [m]")
            ax.set_title(f"{v} - worst run (run {w['run']}, ATE {w['ate']:.3f} m); "
                         f"orange = switch on, red ticks = not tracked", fontsize=9)
        axes[-1, 0].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(rep, f"switch_{s_}.png"), dpi=120)
        plt.close(fig)

        # images at the start of each switch episode, for the worst run of the last variant that has any
        for v in reversed(vs):
            w = worst[(s_, v)]
            if w["eps"]:
                break
        else:
            continue
        eps = w["eps"][:12]
        cols = min(4, len(eps))
        rows_ = (len(eps) + cols - 1) // cols
        fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 2.6 * rows_), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (a_, b) in zip(axes.ravel(), eps):
            img = nearest_rgb(seq_dir(cfg, s_), w["t"][a_])
            if img and os.path.exists(img):
                ax.imshow(mpimg.imread(img))
            ax.set_title(f"t={w['t'][a_] - w['t'][0]:.1f}s, on {w['t'][b] - w['t'][a_]:.1f}s", fontsize=8)
        fig.suptitle(f"{short(s_)} {v} run {w['run']}: image when the switch turned on", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(rep, f"switch_frames_{s_}.png"), dpi=110)
        plt.close(fig)
    runs_figure(cfg, root, rep, a.seq, metrics, thr, plt)
    print(f"\nfigures: {rep}/switch_<sequence>.png, switch_frames_<sequence>.png, switch_runs_{a.seq}.png")


def runs_figure(cfg, root, rep, s_, metrics, thr, plt):
    """Every run of one sequence: error over time (switch-on shaded, lost frames in red) and, on a second
    axis, the outlier ratio inside the YOLO boxes vs outside - the two numbers the geometric check compares."""
    if s_ not in cfg["sequences"]:
        return
    gt = os.path.join(seq_dir(cfg, s_), "groundtruth.txt")
    items = []
    for v in cfg["variants"]:
        vd = os.path.join(root, s_, v)
        if not os.path.isdir(vd):
            continue
        for rd in sorted(x for x in os.listdir(vd) if x.startswith("run_")):
            p = os.path.join(vd, rd)
            n = int(rd.split("_")[1])
            status, ate = metrics.get((s_, v, n), ("missing", None))
            if ate is None or not os.path.exists(os.path.join(p, "diag_frames.csv")):
                continue
            items.append((v, n, ate, p))
    if not items:
        return
    fig, axes = plt.subplots(len(items), 1, figsize=(9, 1.9 * len(items)), sharex=True, squeeze=False)
    for ax, (v, n, ate, p) in zip(axes[:, 0], items):
        rows = read_diag(os.path.join(p, "diag_frames.csv"))
        t = np.array([r["t"] for r in rows])
        t0 = t[0]
        for a_, b in episodes(rows):
            ax.axvspan(t[a_] - t0, t[b] - t0, color="#f2a541", alpha=0.3, linewidth=0)
        _, det = tm.ate(gt, os.path.join(p, "traj.txt"), cfg["max_time_diff_s"], gt_kind=gt_kind(cfg, s_))
        ax.plot(np.asarray(det["t"]) - t0, det["err"], color="#2f5d8a", linewidth=1.2)
        lost = np.array([r["state"] != 2 for r in rows])
        if lost.any():
            ax.plot(t[lost] - t0, np.zeros(lost.sum()), "|", color="#c0392b", markersize=7)
        ax.axhline(thr, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, max(0.3, float(np.max(det["err"])) * 1.05))
        ax.set_ylabel("err [m]", fontsize=8)
        ax.set_title(f"{v} run {n}  ATE {ate:.3f} m{'  FAILED' if ate > thr else ''}", fontsize=8, loc="left")
        ax2 = ax.twinx()
        ib = np.array([r["outliers_in_box"] + r["inliers_in_box"] for r in rows])
        rin = np.where(ib >= 5, np.array([r["outliers_in_box"] for r in rows]) / np.maximum(ib, 1), np.nan)
        ob = np.array([r["matched"] for r in rows]) - ib
        rout = np.array([r["outliers"] - r["outliers_in_box"] for r in rows]) / np.maximum(ob, 1)
        ax2.plot(t - t0, rout, color="#999999", linewidth=0.6)
        ax2.plot(t - t0, rin, ".", color="#8e44ad", markersize=2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("outlier ratio", fontsize=8)
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle(f"{short(s_)} all runs - blue: position error, orange: switch on, red: not tracked, "
                 f"purple dots: outlier ratio inside box, grey: outside", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(rep, f"switch_runs_{s_}.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
