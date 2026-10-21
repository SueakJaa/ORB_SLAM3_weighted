"""Shared helpers: config loading, dataset paths, calibration choice, run directories."""
import os
import subprocess
import sys


def _expand(p, base):
    p = os.path.expanduser(os.path.expandvars(str(p)))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load_config(path):
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            sys.exit("PyYAML missing: pip install pyyaml  (or use a .json config with the same keys)")
        with open(path) as f:
            cfg = yaml.safe_load(f)
    else:
        import json
        with open(path) as f:
            cfg = json.load(f)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg.setdefault("runs", 10)
    cfg.setdefault("fail_threshold_m", 0.1)
    cfg.setdefault("max_time_diff_s", 0.01)
    cfg.setdefault("rpe_delta", 1.0)
    cfg.setdefault("rpe_delta_unit", "s")
    cfg.setdefault("baseline", next(iter(cfg["variants"])))
    root = _expand(cfg.get("orbslam_root", "."), os.getcwd())
    cfg["orbslam_root"] = root
    cfg["executable"] = _expand(cfg.get("executable", "Examples/RGB-D/rgbd_tum"), root)
    cfg["vocabulary"] = _expand(cfg.get("vocabulary", "Vocabulary/ORBvoc.txt"), root)
    cfg["data_root"] = _expand(cfg.get("data_root", "~/datasets"), root)
    cfg["results_root"] = _expand(cfg.get("results_root", "results"), root)
    cfg.setdefault("calibration", {"freiburg1": "Examples/RGB-D/TUM1.yaml",
                                   "freiburg2": "Examples/RGB-D/TUM2.yaml",
                                   "freiburg3": "Examples/RGB-D/TUM3.yaml"})
    cfg["sequences"] = [s if s.startswith("freiburg") else "freiburg3_" + s for s in cfg["sequences"]]
    return cfg


def seq_dir(cfg, seq):
    return os.path.join(cfg["data_root"], "rgbd_dataset_" + seq)


def seq_calib(cfg, seq):
    cam = seq.split("_")[0]
    return _expand(cfg["calibration"][cam], cfg["orbslam_root"])


def short(seq):
    return seq.replace("freiburg1_", "fr1/").replace("freiburg2_", "fr2/").replace("freiburg3_", "fr3/")


def ensure_association(repo, d):
    assoc = os.path.join(d, "association_new.txt")
    if not (os.path.exists(assoc) and os.path.getsize(assoc) > 0):
        out = subprocess.run([sys.executable, os.path.join(repo, "evaluation", "associate.py"),
                              os.path.join(d, "rgb.txt"), os.path.join(d, "depth.txt")],
                             capture_output=True, text=True, check=True).stdout
        with open(assoc, "w") as f:
            f.write(out)
        print(f"   created {assoc}")
    return assoc


def exp_dir(cfg):
    return os.path.join(cfg["results_root"], cfg["name"])


def run_dirs(cfg, seq, variant, runs):
    return [os.path.join(exp_dir(cfg), seq, variant, f"run_{i:02d}") for i in range(1, runs + 1)]
