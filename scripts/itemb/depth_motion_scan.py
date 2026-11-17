#!/usr/bin/env python3
"""Item B prototype: find moving things WITHOUT YOLO, from depth + camera motion only (offline).

Idea. For frame t and an earlier frame r = t - GAP, move every pixel of t's depth image into r with the
relative camera pose and compare its predicted depth with what r actually measured there. A static
point agrees (within the Kinect noise). Something that has MOVED INTO that spot since r is closer now
than what r saw (r saw the background): predicted depth < measured depth by more than the noise. Such
pixels, grouped into one compact region, are the moving object - a box, a person, anything.

    e = z_r_measured - z_r_predicted,  moving if e > K * sqrt(s(z_pred)^2 + s(z_meas)^2) + POSE_TOL,
    s(z) = 1.425e-3 * z^2 (Kinect axial noise, same model as the v21 depth test)

Only "arrived" pixels (e > 0) count: the opposite sign is background that a moving object uncovered,
which is static. Per frame it reports
    incons%   share of valid pixels that disagree (pose error makes this large everywhere)
    largest%  the largest connected region of disagreeing pixels, as a share of valid pixels
    focus     largest / all disagreeing pixels (one compact object -> high; pose error -> low)
    fire      largest% >= MIN_AREA and focus >= MIN_FOCUS

Usage (from ~/ORB_SLAM3):
    python3 scripts/itemb/depth_motion_scan.py scripts/itemb/itemb_dev.yaml                # ground-truth poses
    python3 scripts/itemb/depth_motion_scan.py scripts/itemb/itemb_dev.yaml --pose GEOK2Z_MF  # poses of run 1
        of that variant in this experiment (needs run_experiments.py on the config first): the realistic case
    options: --seq NAME (repeatable) --gap 10 --step 2 --scale 4 --k 3 --pose-tol 0.02 --min-area 1 --min-focus 0.5
Output in <results_root>/<experiment>/itemb_<pose>/:
    frames_<seq>.csv   per-frame numbers
    scan_<seq>.png     largest% and incons% over time, fire shaded
    masks_<seq>.png    camera images with the disagreeing pixels (red) and the largest region (yellow outline)
    and a summary table on screen: % of frames that fire, per sequence.
Ground-truth poses = the detector's best case (pose known); --pose = what the real system would see.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from common import load_config, seq_dir, seq_calib, exp_dir, short, gt_kind  # noqa: E402
import tum_metrics as tm  # noqa: E402

try:
    from scipy import ndimage
except ImportError:
    sys.exit("needs scipy: pip install scipy")


def read_depth(path, factor):
    try:
        from PIL import Image
        a = np.asarray(Image.open(path), dtype=np.float32)
    except ImportError:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return a / factor


def read_rgb(path):
    import matplotlib.image as mpimg
    return mpimg.imread(path)


def intrinsics(yaml_path):
    txt = open(yaml_path).read()

    def get(*keys):
        for k in keys:
            m = re.search(r"^\s*" + re.escape(k) + r"\s*:\s*([-\d.eE+]+)", txt, re.M)
            if m:
                return float(m.group(1))
        raise ValueError(f"{keys[0]} not in {yaml_path}")
    return (get("Camera1.fx", "Camera.fx"), get("Camera1.fy", "Camera.fy"), get("Camera1.cx", "Camera.cx"),
            get("Camera1.cy", "Camera.cy"), get("RGBD.DepthMapFactor", "DepthMapFactor"))


def read_list(d, name):
    out = []
    with open(os.path.join(d, name)) as f:
        for line in f:
            if line.strip() and line[0] != "#":
                ts, fn = line.split()[:2]
                out.append((float(ts), os.path.join(d, fn)))
    return out


def poses_from(path, kind):
    t, p, q = tm.load_tum(path)
    if kind:
        p, q = tm.gt_transform(p, q, kind)
    return t, tm.to_se3(p, q)   # Twc


def nearest(ts, t, tol):
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    j = i if abs(ts[i] - t) < abs(ts[i - 1] - t) else i - 1
    return j if abs(ts[j] - t) <= tol else None


def scan_pair(Dt, Dr, T_rt, K, a):
    fx, fy, cx, cy = K
    s = a.scale
    v, u = np.mgrid[s // 2:Dt.shape[0]:s, s // 2:Dt.shape[1]:s]
    z = Dt[v, u]
    ok = z > 0.1
    X = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], -1)
    Xr = X @ T_rt[:3, :3].T + T_rt[:3, 3]
    zp = Xr[..., 2]
    ok &= zp > 0.1
    zps = np.where(ok, zp, 1.0)
    ur = np.rint(fx * Xr[..., 0] / zps + cx).astype(int)
    vr = np.rint(fy * Xr[..., 1] / zps + cy).astype(int)
    best = np.full(z.shape, np.nan)
    for dv in (-1, 0, 1):          # measured depth in a 3x3 window closest to the prediction
        for du in (-1, 0, 1):
            uu, vv = ur + du, vr + dv
            inb = ok & (uu >= 0) & (vv >= 0) & (uu < Dr.shape[1]) & (vv < Dr.shape[0])
            d = np.full(z.shape, np.nan)
            d[inb] = Dr[vv[inb], uu[inb]]
            d[d <= 0] = np.nan
            take = np.isnan(best) | (np.abs(d - zp) < np.abs(best - zp))
            best = np.where(take & ~np.isnan(d), d, best)
    valid = ok & ~np.isnan(best)
    sig = lambda x: 1.425e-3 * x * x
    tol = a.k * np.sqrt(sig(zp) ** 2 + sig(np.nan_to_num(best)) ** 2) + a.pose_tol
    moved = valid & ((np.nan_to_num(best) - zp) > tol)
    moved = ndimage.binary_opening(moved, structure=np.ones((2, 2)))
    lab, n = ndimage.label(moved, structure=np.ones((3, 3)))
    nv = max(int(valid.sum()), 1)
    if n:
        sizes = ndimage.sum(moved, lab, index=np.arange(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        largest = lab == k
    else:
        largest = np.zeros_like(moved)
    nm = int(moved.sum())
    return dict(valid=nv / moved.size, incons=nm / nv, largest=float(largest.sum()) / nv,
                focus=float(largest.sum()) / nm if nm else 0.0, regions=int(n)), moved, largest, (u, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--seq", action="append")
    ap.add_argument("--pose", default="gt", help="gt (default) or a variant name: poses of its run 1")
    ap.add_argument("--gap", type=int, default=10, help="frames back (10 = 1/3 s at 30 Hz)")
    ap.add_argument("--step", type=int, default=2, help="look at every n-th frame")
    ap.add_argument("--scale", type=int, default=4, help="pixel subsampling (4 -> 160x120)")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--pose-tol", type=float, default=0.02)
    ap.add_argument("--min-area", type=float, default=1.0, help="largest region, %% of valid pixels")
    ap.add_argument("--min-focus", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seqs = a.seq or cfg["sequences"]
    out = os.path.join(exp_dir(cfg), f"itemb_{a.pose}")
    os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = []
    for s in seqs:
        d = seq_dir(cfg, s)
        fx, fy, cx, cy, factor = intrinsics(seq_calib(cfg, s))
        depth, rgb = read_list(d, "depth.txt"), read_list(d, "rgb.txt")
        if a.pose == "gt":
            pt, Tw = poses_from(os.path.join(d, "groundtruth.txt"), gt_kind(cfg, s))
        else:
            tp = os.path.join(exp_dir(cfg), s, a.pose, "run_01", "traj.txt")
            if not os.path.exists(tp):
                print(f"skip {short(s)}: no {tp}")
                continue
            pt, Tw = poses_from(tp, None)
        rts = np.array([x[0] for x in rgb])
        rows, keep = [], []
        for t in range(a.gap, len(depth), a.step):
            ts, tr = depth[t][0], depth[t - a.gap][0]
            i, j = nearest(pt, ts, 0.03), nearest(pt, tr, 0.03)
            if i is None or j is None:
                continue
            T_rt = np.linalg.inv(Tw[j]) @ Tw[i]
            Dt, Dr = read_depth(depth[t][1], factor), read_depth(depth[t - a.gap][1], factor)
            m, moved, largest, uv = scan_pair(Dt, Dr, T_rt, (fx, fy, cx, cy), a)
            m.update(t=ts - depth[0][0], frame=t,
                     fire=int(100 * m["largest"] >= a.min_area and m["focus"] >= a.min_focus))
            rows.append(m)
            keep.append((m["largest"], t, moved, largest, uv))
        if not rows:
            print(f"skip {short(s)}: no frames with poses")
            continue
        with open(os.path.join(out, f"frames_{s}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "t", "valid", "incons", "largest", "focus", "regions", "fire"])
            w.writeheader()
            w.writerows(rows)
        fire = np.array([r["fire"] for r in rows])
        t_ = np.array([r["t"] for r in rows])
        lg = np.array([100 * r["largest"] for r in rows])
        ic = np.array([100 * r["incons"] for r in rows])
        summary.append((s, len(rows), 100 * fire.mean(), np.median(lg), np.percentile(lg, 90), np.median(ic)))
        # time plot
        fig, ax = plt.subplots(figsize=(9, 2.6))
        for k in np.nonzero(fire)[0]:
            ax.axvspan(t_[k] - a.step / 60, t_[k] + a.step / 60, color="#f2a541", alpha=0.35, linewidth=0)
        ax.plot(t_, ic, color="#999999", linewidth=0.8, label="all disagreeing pixels %")
        ax.plot(t_, lg, color="#2f5d8a", linewidth=1.3, label="largest region %")
        ax.axhline(a.min_area, color="#c0392b", linestyle="--", linewidth=0.8)
        ax.set_xlabel("time [s]"), ax.set_ylabel("% of valid pixels")
        ax.set_title(f"{short(s)} - poses: {a.pose}, gap {a.gap}; orange = fires "
                     f"({100 * fire.mean():.0f} % of frames)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout(), fig.savefig(os.path.join(out, f"scan_{s}.png"), dpi=110), plt.close(fig)
        # mask montage: 4 highest + 4 evenly spaced frames
        keep.sort(key=lambda x: -x[0])
        pick = keep[:4] + [keep[i] for i in np.linspace(4, len(keep) - 1, 4).astype(int) if len(keep) > 4]
        pick = sorted({p[1]: p for p in pick}.values(), key=lambda p: p[1])
        fig, axs = plt.subplots(2, 4, figsize=(13, 5.2))
        for ax in axs.ravel():
            ax.axis("off")
        for ax, (lgv, t, moved, largest, (u, v)) in zip(axs.ravel(), pick):
            k = nearest(rts, depth[t][0], 0.05)
            if k is not None:
                ax.imshow(read_rgb(rgb[k][1]))
            ax.scatter(u[moved], v[moved], s=2, c="#e74c3c", linewidths=0)
            ax.contour(u[0], v[:, 0], largest.astype(float), levels=[0.5], colors="#f1c40f", linewidths=1.2)
            ax.set_title(f"t={depth[t][0] - depth[0][0]:.1f}s largest {100 * lgv:.1f}%", fontsize=8)
        fig.suptitle(f"{short(s)}: red = disagrees with the frame {a.gap} back, yellow = largest region "
                     f"(poses: {a.pose})", fontsize=10)
        fig.tight_layout(), fig.savefig(os.path.join(out, f"masks_{s}.png"), dpi=100), plt.close(fig)
    print(f"\nposes: {a.pose}, gap {a.gap} frames, fires if largest region >= {a.min_area} % and focus >= {a.min_focus}")
    print(f"{'sequence':34s} {'frames':>6s} {'fires %':>8s} {'largest% med':>12s} {'p90':>6s} {'incons% med':>11s}")
    for s, n, fp, lm, l90, im in summary:
        print(f"{short(s):34s} {n:6d} {fp:8.1f} {lm:12.2f} {l90:6.2f} {im:11.2f}")
    print(f"\nfigures and CSVs: {out}")


if __name__ == "__main__":
    main()
