#!/usr/bin/env python3
"""Copy results of the old shell scripts (results_v16/, results_v18_unseen/, ...) into the new layout,
so they can be evaluated and plotted without re-running anything.

Usage (from ~/ORB_SLAM3):
    python3 scripts/eval/import_legacy.py results_v16 CONFIG
    python3 scripts/eval/import_legacy.py results_v18_unseen CONFIG --variant GEOK2_G10 --variant A0_orbslam3

Old layout: <dir>/<sequence>/<VARIANT>_<i>.txt (+ _<i>_kf.txt, _<i>.log). Short fr3 names
(walking_xyz) become freiburg3_walking_xyz. Existing runs in the new layout are not overwritten.
"""
import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, exp_dir  # noqa: E402
from run_experiments import parse_log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy_dir")
    ap.add_argument("config")
    ap.add_argument("--variant", action="append")
    a = ap.parse_args()
    cfg = load_config(a.config)
    n = 0
    for seq in sorted(os.listdir(a.legacy_dir)):
        sd = os.path.join(a.legacy_dir, seq)
        if not os.path.isdir(sd):
            continue
        new_seq = seq if seq.startswith("freiburg") else "freiburg3_" + seq
        for f in sorted(os.listdir(sd)):
            m = re.fullmatch(r"(.+)_(\d+)\.log", f)
            if not m:
                continue
            var, i = m.group(1), int(m.group(2))
            if a.variant and var not in a.variant:
                continue
            rd = os.path.join(exp_dir(cfg), new_seq, var, f"run_{i:02d}")
            if os.path.exists(rd):
                continue
            os.makedirs(rd)
            shutil.copy(os.path.join(sd, f), os.path.join(rd, "log.txt"))
            for src, dst in ((f"{var}_{i}.txt", "traj.txt"), (f"{var}_{i}_kf.txt", "kf.txt")):
                if os.path.exists(os.path.join(sd, src)):
                    shutil.copy(os.path.join(sd, src), os.path.join(rd, dst))
            with open(os.path.join(rd, "log.txt"), errors="replace") as lf:
                meta = {"imported_from": os.path.abspath(sd), "sequence": new_seq, "variant": var, "run": i,
                        "trajectory": os.path.exists(os.path.join(rd, "traj.txt")), **parse_log(lf.read())}
            with open(os.path.join(rd, "meta.json"), "w") as mf:
                json.dump(meta, mf, indent=1)
            n += 1
    print(f"imported {n} runs into {exp_dir(cfg)}. Next: evaluate.py and report.py with the same config.")


if __name__ == "__main__":
    main()
