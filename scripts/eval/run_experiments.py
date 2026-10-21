#!/usr/bin/env python3
"""Run SLAM variants on TUM RGB-D sequences, N times each, from one config file.

Usage (from ~/ORB_SLAM3):
    python3 scripts/eval/run_experiments.py scripts/eval/configs/baseline_tum_dynamic.yaml
    python3 scripts/eval/run_experiments.py CONFIG --runs 3 --seq freiburg3_walking_xyz --variant A0_orbslam3
    python3 scripts/eval/run_experiments.py CONFIG --dry-run          # print what would run

Layout written:  <results_root>/<experiment>/<sequence>/<variant>/run_01/
    traj.txt  (camera trajectory, TUM format)   kf.txt   log.txt   meta.json
Finished runs are skipped, so an interrupted experiment can simply be restarted. A crashed run
counts as a result (the crash rate belongs in the thesis); --retry-crashed re-runs only those,
--force re-runs everything. Every run gets its own working directory (rgbd_tum writes fixed file names).
Runs are strictly sequential on purpose: ORB-SLAM3's keyframe decisions depend on processing speed,
so two runs sharing the machine would change each other's results.
"""
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, seq_calib, ensure_association, run_dirs  # noqa: E402


def git_info(repo):
    def g(*a):
        try:
            return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "--short", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [l[3:] for l in g("status", "--porcelain", "--untracked-files=no").splitlines()]}


def parse_log(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("median tracking time:"):
            out["median_tracking_s"] = float(line.split(":")[1])
        elif line.startswith("mean tracking time:"):
            out["mean_tracking_s"] = float(line.split(":")[1])
        elif "[DynamicConfig]" in line:
            out["dynamic_config"] = line.split("[DynamicConfig]", 1)[1].strip()
        elif "[DynamicStats]" in line:
            out["dynamic_stats"] = line.split("[DynamicStats]", 1)[1].strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--runs", type=int, help="override runs per variant")
    ap.add_argument("--seq", action="append", help="only these sequences (repeatable)")
    ap.add_argument("--variant", action="append", help="only these variants (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-run runs that already finished")
    ap.add_argument("--retry-crashed", action="store_true",
                    help="re-run finished runs without trajectory (by default a crash is kept as a result)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    runs = a.runs or cfg["runs"]
    seqs = [s for s in cfg["sequences"] if not a.seq or s in a.seq]
    variants = {k: v for k, v in cfg["variants"].items() if not a.variant or k in a.variant}
    repo = cfg["orbslam_root"]
    exe, voc = cfg["executable"], cfg["vocabulary"]
    for p in (exe, voc):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}  (check orbslam_root / executable / vocabulary in the config)")
    gi = git_info(repo)
    if gi["dirty_files"]:
        print(f"!! uncommitted changes in {repo}: {', '.join(gi['dirty_files'])} - results will not match a commit")

    jobs, missing = [], []
    for s in seqs:
        d = seq_dir(cfg, s)
        if not os.path.isdir(d):
            missing.append(s)
            continue
        for v, env in variants.items():
            for i, rd in enumerate(run_dirs(cfg, s, v, runs), 1):
                done = os.path.exists(os.path.join(rd, "meta.json"))   # finished, with or without trajectory
                crashed = done and not os.path.exists(os.path.join(rd, "traj.txt"))
                if done and not a.force and not (crashed and a.retry_crashed):
                    continue
                jobs.append((s, v, env, i, rd))
    if missing:
        print(f"!! datasets not found in {cfg['data_root']} (skipped): {', '.join(missing)}")
    print(f"experiment '{cfg['name']}': {len(jobs)} runs to do  (commit {gi['commit']} on {gi['branch']})")
    if a.dry_run:
        for s, v, env, i, rd in jobs:
            print(f"  {s:32s} {v:18s} run {i:2d} -> {rd}")
        return

    t_start = time.time()
    for k, (s, v, env, i, rd) in enumerate(jobs, 1):
        d = seq_dir(cfg, s)
        assoc = ensure_association(repo, d)
        calib = seq_calib(cfg, s)
        if os.path.isdir(rd):
            shutil.rmtree(rd)
        os.makedirs(rd)
        run_env = dict(os.environ)
        run_env.update({kk: str(vv) for kk, vv in {**cfg.get("common_env", {}), **(env or {})}.items()})
        cmd = [exe, voc, calib, d, assoc]
        eta = ""
        if k > 1:
            per = (time.time() - t_start) / (k - 1)
            eta = f"  (~{per * (len(jobs) - k + 1) / 60:.0f} min left)"
        print(f"[{time.strftime('%H:%M')}] {k}/{len(jobs)}  {s}  {v}  run {i}{eta}", flush=True)
        t0 = time.time()
        with open(os.path.join(rd, "log.txt"), "w") as log:
            try:
                rc = subprocess.run(cmd, cwd=rd, env=run_env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=cfg.get("timeout_s", 3600)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        dur = time.time() - t0
        ok = os.path.exists(os.path.join(rd, "CameraTrajectory.txt"))
        if ok:
            os.replace(os.path.join(rd, "CameraTrajectory.txt"), os.path.join(rd, "traj.txt"))
        if os.path.exists(os.path.join(rd, "KeyFrameTrajectory.txt")):
            os.replace(os.path.join(rd, "KeyFrameTrajectory.txt"), os.path.join(rd, "kf.txt"))
        with open(os.path.join(rd, "log.txt"), errors="replace") as f:
            info = parse_log(f.read())
        meta = {"experiment": cfg["name"], "sequence": s, "variant": v, "run": i, "returncode": rc,
                "trajectory": ok, "duration_s": round(dur, 1), "command": cmd, "calibration": calib,
                "env": {kk: run_env[kk] for kk in sorted(run_env) if kk.startswith("DSLAM_")},
                "git": gi, "host": socket.gethostname(), "platform": platform.platform(),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        with open(os.path.join(rd, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if not ok:
            print(f"   !! no trajectory (return code {rc}) - see {rd}/log.txt")
    print(f"done in {(time.time() - t_start) / 60:.0f} min. Next: python3 scripts/eval/evaluate.py {a.config}")


if __name__ == "__main__":
    main()
