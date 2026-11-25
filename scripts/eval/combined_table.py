#!/usr/bin/env python3
"""One Markdown table across experiments: ORB-SLAM3, GEOK2_G10, GEOK2_MF, GEOK2Z_MF on TUM (12) and Bonn (10).

Usage (from ~/ORB_SLAM3), after evaluate.py has written metrics.csv for thesis_z, thesis_final, bonn_baseline:
    python3 scripts/eval/combined_table.py                       # prints the table, writes results/combined_table.md
    python3 scripts/eval/combined_table.py --tum-runs 10 --bonn-runs 5

Numbers come from each experiment's metrics.csv, so the scoring is the pipeline's own: one pose per timestamp
(repeated timestamps written while RECENTLY_LOST are dropped, not scored), association |dt| <= 10 ms, ATE = RMSE
after SE(3) Umeyama alignment per run; the table shows the median over runs. A run fails if ATE > 0.1 m or it
crashed. Only runs 1..N are used (N = --tum-runs / --bonn-runs), so every variant has the same N.
p = two-sided Mann-Whitney U on per-run ATE, GEOK2Z_MF vs GEOK2_MF, then Holm over the sequences of each dataset.
"""
import argparse
import csv
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import mann_whitney_p  # noqa: E402
from common import short  # noqa: E402

TUM = ["freiburg3_walking_xyz", "freiburg3_walking_static", "freiburg3_walking_halfsphere", "freiburg3_walking_rpy",
       "freiburg3_sitting_xyz", "freiburg3_sitting_static", "freiburg3_sitting_halfsphere",
       "freiburg2_desk_with_person", "freiburg1_room", "freiburg1_desk", "freiburg1_desk2", "freiburg1_360"]
BONN = ["bonn_balloon", "bonn_balloon2", "bonn_crowd", "bonn_crowd2", "bonn_person_tracking", "bonn_person_tracking2",
        "bonn_moving_nonobstructing_box", "bonn_moving_obstructing_box", "bonn_synchronous", "bonn_static"]
VARIANTS = [("A0_orbslam3", "ORB-SLAM3"), ("GEOK2_G10", "GEOK2_G10"), ("GEOK2_MF", "GEOK2_MF"),
            ("GEOK2Z_MF", "GEOK2Z_MF")]
# where each variant's runs live: TUM GEOK2_G10 was only run in the ablation experiment
TUM_SRC = {"A0_orbslam3": "thesis_z", "GEOK2_G10": "thesis_final", "GEOK2_MF": "thesis_z", "GEOK2Z_MF": "thesis_z"}
BONN_SRC = {v: "bonn_baseline" for v, _ in VARIANTS}
FAIL_M = 0.1


def load(root, exp):
    p = os.path.join(root, exp, "metrics.csv")
    if not os.path.exists(p):
        sys.exit(f"missing {p}: run  python3 scripts/eval/evaluate.py scripts/eval/configs/{exp}.yaml  first")
    with open(p) as f:
        return list(csv.DictReader(f))


def holm(ps):
    order = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    out, run_max, m = [None] * len(ps), 0.0, len(order)
    for k, (p, i) in enumerate(order):
        run_max = max(run_max, min(1.0, (m - k) * p))
        out[i] = run_max
    return out


def cell_stats(rows, seq, var, n):
    rs = [r for r in rows if r["sequence"] == seq and r["variant"] == var and int(r["run"]) <= n]
    ok = [r for r in rs if r["status"] == "ok" and r["ate_rmse"]]
    ate = [float(r["ate_rmse"]) for r in ok]
    tr = [float(r["tracked_pct"]) for r in ok if r.get("tracked_pct")]
    return {"n": len(rs), "ate": ate, "med": st.median(ate) if ate else None,
            "fail": sum(a > FAIL_M for a in ate) + (len(rs) - len(ok)),
            "track": st.mean(tr) if tr else None}


def fmt_p(p):
    if p is None:
        return "-"
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"


def block(title, seqs, src, n, data):
    lines = [f"**{title}** ({n} runs per variant)", "",
             "| Sequence | " + " | ".join(f"{lab} ATE / fail / track%" for _, lab in VARIANTS) +
             " | p (Z vs MF) | p Holm |",
             "|---|" + "---|" * len(VARIANTS) + "---|---|"]
    cells, ps = [], []
    for s in seqs:
        c = {v: cell_stats(data[src[v]], s, v, n) for v, _ in VARIANTS}
        a, b = c["GEOK2Z_MF"]["ate"], c["GEOK2_MF"]["ate"]
        ps.append(mann_whitney_p(a, b) if len(a) >= 3 and len(b) >= 3 else None)
        cells.append(c)
    ph = holm(ps)
    short_n = []
    for s, c, p, h in zip(seqs, cells, ps, ph):
        row = []
        for v, _ in VARIANTS:
            d = c[v]
            if d["n"] < n:
                short_n.append(f"{short(s)} {v}: {d['n']}/{n} runs")
            med = f"{d['med']:.4f}" if d["med"] is not None else "-"
            tr = f"{d['track']:.1f}" if d["track"] is not None else "-"
            row.append(f"{med} / {d['fail']}/{d['n']} / {tr}")
        lines.append(f"| {short(s)} | " + " | ".join(row) + f" | {fmt_p(p)} | {fmt_p(h)} |")
    return lines, short_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--tum-runs", type=int, default=10)
    ap.add_argument("--bonn-runs", type=int, default=5)
    a = ap.parse_args()
    data = {e: load(a.root, e) for e in set(TUM_SRC.values()) | set(BONN_SRC.values())}
    out = ["ATE = median over runs of ATE RMSE [m]; fail = runs with ATE > 0.1 m or crashed; "
           "track% = mean share of frames with a pose. p = Mann-Whitney U, GEOK2Z_MF vs GEOK2_MF.", ""]
    l1, w1 = block("TUM RGB-D", TUM, TUM_SRC, a.tum_runs, data)
    l2, w2 = block("Bonn RGB-D Dynamic", BONN, BONN_SRC, a.bonn_runs, data)
    out += l1 + [""] + l2
    if w1 + w2:
        out += ["", "Missing runs: " + "; ".join(w1 + w2)]
    txt = "\n".join(out) + "\n"
    print(txt)
    with open(os.path.join(a.root, "combined_table.md"), "w") as f:
        f.write(txt)
    print(f"-> {os.path.join(a.root, 'combined_table.md')}")


if __name__ == "__main__":
    main()
