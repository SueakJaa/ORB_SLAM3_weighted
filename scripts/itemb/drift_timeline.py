#!/usr/bin/env python3
"""Where exactly does a run go wrong? Local drift over time, next to what the system was doing.

ATE aligns the whole trajectory at once, so one bad stretch makes every frame look wrong. Here the error
is LOCAL: for every frame t, the estimated motion from t to t+DT is compared with the true motion over the
same time (relative pose error, no global alignment). A bad stretch shows up as a spike where it happens.

Usage (from ~/ORB_SLAM3):
    python3 scripts/itemb/drift_timeline.py scripts/itemb/itemb_trig.yaml --seq bonn_moving_obstructing_box2 \
        --variant GEOK2Z_MF_DT --variant DT_ONLY_T
    python3 scripts/itemb/drift_timeline.py scripts/itemb/itemb_dev.yaml --seq bonn_moving_obstructing_box2 \
        --variant A0_orbslam3 --variant GEOK2Z_MF
    options: --run 1   --dt 0.5 (seconds)
One row per variant: drift [cm per DT] (blue), switch on (orange), and on the right axis the depth region
size (% of image, green), YOLO share of keypoints (purple), tracked inliers / 10 (grey); ticks at the
bottom = keyframes created. Written to <experiment>/report/drift_<seq>.png, and the five worst moments of
each run are printed with what was going on then.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from common import load_config, seq_dir, exp_dir, short, gt_kind  # noqa: E402
import tum_metrics as tm  # noqa: E402


def read_diag(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[round(float(r["t"]), 4)] = {k: float(v) for k, v in r.items() if v not in (None, "")}
            except (KeyError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config(a.config)
    gt_path = os.path.join(seq_dir(cfg, a.seq), "groundtruth.txt")
    tg, pg, qg = tm.load_tum(gt_path)
    pg, qg = tm.gt_transform(pg, qg, gt_kind(cfg, a.seq))
    G = tm.to_se3(pg, qg)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for v in a.variant:
        rd = os.path.join(exp_dir(cfg), a.seq, v, f"run_{a.run:02d}")
        tp = os.path.join(rd, "traj.txt")
        if not os.path.exists(tp):
            print(f"skip {v}: no {tp}")
            continue
        te, pe, qe = tm.load_tum(tp)
        E = tm.to_se3(pe, qe)
        ig, ie = tm.associate(tg, te, cfg["max_time_diff_s"])
        t = te[ie]
        Gm, Em = G[ig], E[ie]
        drift_t, drift_r, tt, mv = [], [], [], []
        for k in range(len(t)):
            j = int(np.searchsorted(t, t[k] + a.dt))
            if j >= len(t) or t[j] - t[k] > 1.5 * a.dt:
                continue
            dG = np.linalg.inv(Gm[k]) @ Gm[j]
            dE = np.linalg.inv(Em[k]) @ Em[j]
            err = np.linalg.inv(dG) @ dE
            rot = lambda M: np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1)))
            mv.append((100 * np.linalg.norm(dG[:3, 3]), rot(dG), 100 * np.linalg.norm(dE[:3, 3]), rot(dE)))
            drift_t.append(100 * np.linalg.norm(err[:3, 3]))
            drift_r.append(np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))))
            tt.append(t[k])
        tt, drift_t, drift_r = np.array(tt), np.array(drift_t), np.array(drift_r)
        diag = read_diag(os.path.join(rd, "diag_frames.csv"))
        # stretches without evaluation: no ground-truth pose (or no estimate) near the frame times
        gaps = [(t[k], t[k + 1]) for k in range(len(t) - 1) if t[k + 1] - t[k] > 0.1]
        rows.append((v, tt, drift_t, drift_r, diag, mv, gaps))
    if not rows:
        sys.exit("nothing to plot")
    t0 = min(r[1][0] for r in rows if len(r[1]))
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.6 * len(rows)), sharex=True, squeeze=False)
    for ax, (v, tt, dtr, drr, diag, mv, gaps) in zip(axes[:, 0], rows):
        for g0, g1 in gaps:
            ax.axvspan(g0 - t0, g1 - t0, color="#bbbbbb", alpha=0.5, linewidth=0, zorder=0)
        ts = np.array(sorted(diag))
        if len(ts):
            on = np.array([diag[x].get("kf_rule_active", 0) >= 1 for x in ts])
            for k in np.nonzero(on)[0]:
                ax.axvspan(ts[k] - t0 - 0.017, ts[k] - t0 + 0.017, color="#f2a541", alpha=0.3, linewidth=0)
            kf = np.array([diag[x].get("kf_created", 0) >= 1 for x in ts])
            ax.plot(ts[kf] - t0, np.zeros(kf.sum()), "|", color="#555555", markersize=6)
        ax.plot(tt - t0, dtr, color="#2f5d8a", linewidth=1.3, label="drift [cm per %.1f s]" % a.dt)
        ax.set_ylabel("drift [cm]")
        ax.set_ylim(0, max(10.0, float(np.percentile(dtr, 99)) * 1.1) if len(dtr) else 10)
        ax.set_title(f"{short(a.seq)} {v} run {a.run}: median drift {np.median(dtr):.1f} cm, "
                     f"max {dtr.max():.1f} cm per {a.dt:g} s", fontsize=9, loc="left")
        if len(ts):
            ax2 = ax.twinx()
            g = lambda key: np.array([diag[x].get(key, np.nan) for x in ts])
            if "dt_largest" in diag[ts[0]]:
                ax2.plot(ts - t0, 100 * g("dt_largest"), color="#1baf7a", linewidth=0.9, label="depth region %")
            ax2.plot(ts - t0, 100 * g("yolo_frac"), color="#8e44ad", linewidth=0.7, alpha=0.7, label="YOLO keypoints %")
            ax2.plot(ts - t0, g("inliers_track") / 10, color="#999999", linewidth=0.7, label="tracked inliers / 10")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("%", fontsize=8)
            if ax is axes[0, 0]:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=4)
        # five worst moments, at least 1 s apart
        total = sum(g1 - g0 for g0, g1 in gaps)
        print(f"\n{v} run {a.run}: {len(gaps)} stretches without ground truth/estimate, {total:.1f} s in total"
              + "".join(f"\n    {g0 - t0:5.1f} - {g1 - t0:5.1f} s" for g0, g1 in gaps if g1 - g0 > 0.3))
        print(f"worst moments (drift over {a.dt:g} s); GT moved / est moved = how far each says the camera went")
        print(f"{'time s':>7s} {'drift cm':>9s} {'rot deg':>8s} | {'GT moved cm/deg':>15s} {'est moved cm/deg':>16s} | "
              f"{'switch':>6s} {'depth reg%':>10s} {'YOLO%':>6s} {'inliers':>8s} {'kf':>3s}")
        order, used = np.argsort(-dtr), []
        for k in order:
            if any(abs(tt[k] - u) < 1.0 for u in used):
                continue
            used.append(tt[k])
            d = diag.get(round(tt[k], 4)) or (diag[ts[int(np.argmin(np.abs(ts - tt[k])))]] if len(ts) else {})
            gm, gr, em, er = mv[k]
            print(f"{tt[k] - t0:7.1f} {dtr[k]:9.1f} {drr[k]:8.1f} | {gm:8.1f}/{gr:5.1f}  {em:9.1f}/{er:5.1f}  | "
                  f"{int(d.get('kf_rule_active', -1)):6d} "
                  f"{100 * d.get('dt_largest', float('nan')):10.1f} {100 * d.get('yolo_frac', float('nan')):6.1f} "
                  f"{int(d.get('inliers_track', -1)):8d} {int(d.get('kf_created', -1)):3d}")
            if len(used) == 5:
                break
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle("local drift (no global alignment); orange = switch on; grey = no ground truth/estimate; ticks = keyframes",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    rep = os.path.join(exp_dir(cfg), "report")
    os.makedirs(rep, exist_ok=True)
    out = os.path.join(rep, f"drift_{a.seq}.png")
    fig.savefig(out, dpi=110)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
