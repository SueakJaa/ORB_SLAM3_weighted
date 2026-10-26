#!/usr/bin/env python3
"""Reproduce an occasional crash under gdb and save the backtrace (where in the code it happens).

Usage (from ~/ORB_SLAM3; needs gdb: sudo apt install -y gdb):
    python3 scripts/eval/crash_hunt.py scripts/eval/configs/bonn_baseline.yaml \
        --seq bonn_static --variant GEOK2_MF --tries 15 --timeout 120

Runs the variant again and again, each try at most --timeout seconds (crashes that happen early show up
fast; runs that survive are stopped and count as "no crash"). The first crash is saved to
results/crash_hunt/<seq>_<variant>_bt.txt with the full backtrace of every thread.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, seq_calib, ensure_association  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tries", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    if not shutil.which("gdb"):
        sys.exit("gdb not found: sudo apt install -y gdb")
    cfg = load_config(a.config)
    d = seq_dir(cfg, a.seq)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in {**cfg.get("common_env", {}), **(cfg["variants"][a.variant] or {})}.items()})
    cmd = ["gdb", "-batch", "-ex", "set pagination off", "-ex", "run", "-ex", "thread apply all bt 25", "--args",
           cfg["executable"], cfg["vocabulary"], seq_calib(cfg, a.seq), d, ensure_association(cfg["orbslam_root"], d)]
    out_dir = os.path.join(cfg["results_root"], "crash_hunt")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, a.tries + 1):
        t0 = time.time()
        with tempfile.TemporaryDirectory() as wd:
            try:
                p = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True, errors="replace",
                                   timeout=a.timeout)
                out = p.stdout + p.stderr
            except subprocess.TimeoutExpired as e:
                subprocess.run(["pkill", "-f", os.path.basename(cfg["executable"])], capture_output=True)
                print(f"try {i}: no crash in {a.timeout} s")
                continue
        crashed = any(s in out for s in ("SIGABRT", "SIGSEGV", "SIGFPE"))
        print(f"try {i}: {'CRASH' if crashed else 'finished without crash'} after {time.time() - t0:.0f} s")
        if crashed:
            path = os.path.join(out_dir, f"{a.seq}_{a.variant}_bt.txt")
            with open(path, "w") as f:
                f.write(out)
            frames = [l for l in out.splitlines() if l.startswith("#") and ("ORB_SLAM3::" in l or "Sophus" in l)]
            print("frames from ORB-SLAM3 / Sophus (first 15):")
            for l in frames[:15]:
                print("   " + l[:200])
            print(f"full backtrace: {path}")
            return
    print("no crash reproduced")


if __name__ == "__main__":
    main()
