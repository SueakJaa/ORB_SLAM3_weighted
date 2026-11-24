#!/usr/bin/env python3
"""Find runs in which ORB-SLAM3 reset or started a new map, from the saved log.txt files.

Usage (from ~/ORB_SLAM3):
    python3 scripts/eval/reset_scan.py                 # all experiments under results/
    python3 scripts/eval/reset_scan.py --exp thesis_z  # one experiment
    python3 scripts/eval/reset_scan.py --runs          # also list every affected run

Events counted per run (the text ORB-SLAM3 prints):
    full_reset   "System Reseting"        Tracking::Reset(): frame ids restart at 0
    map_reset    "Active map Reseting"    Tracking::ResetActiveMap(): lost with < 10 keyframes, map erased
    new_map      "First frame id in map"  Tracking::CreateMapInAtlas(): lost, a new map is started
Before commit "clear depth history on map change", variants using DSLAM_MAP_DEPTH_TEST or DSLAM_DEPTH_TRIG
kept depth frames from the old map after map_reset / new_map; those runs are marked DEPTH.
"""
import argparse
import os
from collections import defaultdict

EVENTS = {"full_reset": "System Reseting",
          "map_reset": "Active map Reseting",
          "new_map": "First frame id in map"}


def uses_depth_history(log_text):
    # the DynamicConfig settings line prints these fields; 1 / nonzero means the feature was on
    for key in (" map_depth_test=", " depth_trig="):
        i = log_text.find(key)
        if i >= 0:
            v = log_text[i + len(key):i + len(key) + 8].split("(")[0].split()[0]
            if v not in ("0", "false", "0.0"):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--runs", action="store_true")
    a = ap.parse_args()

    exps = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    if a.exp:
        exps = [e for e in exps if e == a.exp]
    total_runs = 0
    affected = []
    table = defaultdict(lambda: defaultdict(int))   # (exp, variant) -> counters
    for exp in exps:
        ed = os.path.join(a.root, exp)
        for seq in sorted(os.listdir(ed)):
            sd = os.path.join(ed, seq)
            if not os.path.isdir(sd):
                continue
            for var in sorted(os.listdir(sd)):
                vd = os.path.join(sd, var)
                if not os.path.isdir(vd):
                    continue
                for run in sorted(os.listdir(vd)):
                    lf = os.path.join(vd, run, "log.txt")
                    if not os.path.isfile(lf):
                        continue
                    total_runs += 1
                    with open(lf, errors="replace") as f:
                        txt = f.read()
                    c = {k: txt.count(s) for k, s in EVENTS.items()}
                    t = table[(exp, var)]
                    t["runs"] += 1
                    if any(c.values()):
                        t["affected"] += 1
                        for k, v in c.items():
                            t[k] += v
                        depth = uses_depth_history(txt) and (c["map_reset"] or c["new_map"])
                        if depth:
                            t["depth"] += 1
                        affected.append((exp, seq, var, run, c, depth))

    print(f"{total_runs} runs scanned, {len(affected)} with a reset or new map\n")
    print(f"{'experiment':22s} {'variant':22s} {'runs':>5s} {'affected':>8s} {'full':>5s} {'mapRst':>6s} {'newMap':>6s} {'DEPTH':>5s}")
    for (exp, var), t in sorted(table.items()):
        if t["affected"] == 0:
            continue
        print(f"{exp:22s} {var:22s} {t['runs']:5d} {t['affected']:8d} {t['full_reset']:5d} "
              f"{t['map_reset']:6d} {t['new_map']:6d} {t['depth']:5d}")
    if a.runs:
        print()
        for exp, seq, var, run, c, depth in affected:
            ev = " ".join(f"{k}={v}" for k, v in c.items() if v)
            print(f"{exp}/{seq}/{var}/{run}: {ev}{'  DEPTH' if depth else ''}")


if __name__ == "__main__":
    main()
