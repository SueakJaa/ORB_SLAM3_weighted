#!/usr/bin/env python3
"""Per-module timing from the [Timing] lines of every run's log.txt, for the thesis timing table.

    python3 scripts/eval/timing_table.py --exp thesis_z bonn_baseline --variants A0_orbslam3 GEOK2Z_MF GEOK2Z_MF_PAR_TRT

All numbers are per frame, in the tracking thread, YOLO INCLUDED: total = time from receiving the RGB-D frame to
the pose (ORB extraction + YOLO + tracking; in parallel mode YOLO overlaps ORB extraction, and "YOLO wait" is the
part of YOLO that tracking still had to wait for). Value = median over runs of each run's median (p95 = median
over runs of each run's 95th percentile). FPS = 1000 / total. Local mapping runs in its own thread, per keyframe.
Frames are fed at the camera rate (30 Hz TUM/Bonn) when the system is faster, so FPS is processing capacity, not
the rate frames arrived at.
"""
import argparse
import os
import re
import statistics as st
import sys

KEYS = ["orb_ms", "yolo_ms", "yolo_wait_ms", "track_ms", "total_ms", "localmap_per_kf_ms"]
COLS = [("orb_ms", "ORB extraction"), ("yolo_ms", "YOLO (own time)"), ("yolo_wait_ms", "YOLO wait"),
        ("track_ms", "Tracking"), ("total_ms", "Total per frame")]


def parse(log):
    with open(log, errors="replace") as f:
        lines = [l for l in f if l.startswith("[Timing]")]
    if not lines:
        return None
    d = {}
    for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", lines[-1]):
        try:
            d[k] = float(v)
        except ValueError:
            pass
    b = re.search(r"yolo_backend=(\w+)", lines[-1])
    d["backend"] = b.group(1) if b else ("none" if "yolo_ms_median" not in d else "cuda (older log)")
    if "yolo_ms_median" in d and "yolo_wait_ms_median" not in d:   # older logs: serial, wait = whole YOLO
        d["yolo_wait_ms_median"] = d["yolo_ms_median"]
        d["yolo_wait_ms_p95"] = d.get("yolo_ms_p95", float("nan"))
    return d


def med(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    a = ap.parse_args()

    out = []
    for exp in a.exp:
        ed = os.path.join(a.root, exp)
        if not os.path.isdir(ed):
            sys.exit(f"no {ed}")
        out.append(f"**{exp}** (median over runs and sequences; ms per frame, YOLO included)\n")
        out.append("| Variant | runs | " + " | ".join(f"{c} med / p95" for _, c in COLS) +
                   " | FPS (1000/total) | Local mapping per keyframe | YOLO backend |")
        out.append("|---|---|" + "---|" * len(COLS) + "---|---|---|")
        for var in a.variants:
            rs = []
            for seq in sorted(os.listdir(ed)):
                vd = os.path.join(ed, seq, var)
                if not os.path.isdir(vd):
                    continue
                for run in sorted(os.listdir(vd)):
                    lf = os.path.join(vd, run, "log.txt")
                    if os.path.isfile(lf):
                        d = parse(lf)
                        if d:
                            rs.append(d)
            if not rs:
                out.append(f"| {var} | 0 | " + " | ".join("-" for _ in COLS) + " | - | - | - |")
                continue
            cells = []
            for k, _ in COLS:
                m = med([r.get(k + "_median", float("nan")) for r in rs])
                p = med([r.get(k + "_p95", float("nan")) for r in rs])
                cells.append("-" if m is None else f"{m:.1f} / {p:.1f}" if p is not None else f"{m:.1f}")
            tot = med([r.get("total_ms_median", float("nan")) for r in rs])
            lm = med([r.get("localmap_per_kf_ms_median", float("nan")) for r in rs])
            backends = sorted({r["backend"] for r in rs})
            out.append(f"| {var} | {len(rs)} | " + " | ".join(cells) +
                       f" | {1000 / tot:.0f} | {lm:.0f} ms | {', '.join(backends)} |" if tot else " | - | - | - |")
        out.append("")
    txt = "\n".join(out)
    print(txt)
    p = os.path.join(a.root, "timing_table.md")
    with open(p, "w") as f:
        f.write(txt + "\n")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
