#!/usr/bin/env python3
"""Summary tables, significance tests and figures for an evaluated experiment.

Usage (from ~/ORB_SLAM3), after evaluate.py:
    python3 scripts/eval/report.py scripts/eval/configs/baseline_tum_dynamic.yaml

Writes into <results_root>/<experiment>/report/:
    summary.md         tables per sequence (ATE, RPE, failures, keyframes, timing) + significance vs baseline
    summary.csv        the same numbers, machine-readable
    ate_box.png        ATE RMSE of every run, one panel per sequence (log scale, failure line dashed)
    rpe_box.png        RPE translation RMSE of every run
    traj_<seq>.png     top view: ground truth + the median run of each variant (SE(3)-aligned)
    error_<seq>.png    ATE over time for the median run of each variant
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, seq_dir, exp_dir, short  # noqa: E402
import tum_metrics as tm  # noqa: E402

# categorical palette, fixed order (slot n = n-th variant in the config) - validated for adjacent pairs
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def mann_whitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ImportError:
        n1, n2 = len(a), len(b)
        allv = sorted((v, k) for k, v in enumerate(a + b))
        ranks = [0.0] * (n1 + n2)
        i = 0
        while i < len(allv):
            j = i
            while j < len(allv) and allv[j][0] == allv[i][0]:
                j += 1
            for k in range(i, j):
                ranks[allv[k][1]] = (i + j + 1) / 2
            i = j
        u = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
        mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (abs(u - mu) - 0.5) / sd if sd else 0
        return math.erfc(z / math.sqrt(2))


def q(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p
    f = math.floor(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


def fnum(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("ate_rmse", "rpe_t_rmse", "rpe_r_rmse", "keyframes", "median_tracking_ms", "switch_on_pct",
                  "tracked_pct"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
        r["run"] = int(r["run"])
    return rows


def summarize(cfg, rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["sequence"], r["variant"])].append(r)
    thr, base = cfg["fail_threshold_m"], cfg["baseline"]
    out = {}
    for (s, v), rs in by.items():
        ok = [r for r in rs if r["status"] == "ok"]
        ate = [r["ate_rmse"] for r in ok]
        mean_of = lambda k: (st.mean([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else None)
        out[(s, v)] = {
            "sequence": s, "variant": v, "runs": len(rs), "crashes": len(rs) - len(ok),
            "fails": sum(1 for a in ate if a > thr) + (len(rs) - len(ok)),
            "ate_median": st.median(ate) if ate else None, "ate_mean": st.mean(ate) if ate else None,
            "ate_std": st.stdev(ate) if len(ate) > 1 else 0.0 if ate else None,
            "ate_q1": q(ate, 0.25) if ate else None, "ate_q3": q(ate, 0.75) if ate else None,
            "rpe_t_median": st.median([r["rpe_t_rmse"] for r in ok]) if ok else None,
            "rpe_r_median": st.median([r["rpe_r_rmse"] for r in ok]) if ok else None,
            "keyframes": mean_of("keyframes"), "tracking_ms": mean_of("median_tracking_ms"),
            "tracked_pct": mean_of("tracked_pct"),
            "switch_on_pct": mean_of("switch_on_pct"), "_ate": ate}
    for (s, v), d in out.items():
        b = out.get((s, base))
        d["p_vs_baseline"] = None
        if v != base and b and len(d["_ate"]) >= 3 and len(b["_ate"]) >= 3:
            d["p_vs_baseline"] = mann_whitney_p(d["_ate"], b["_ate"])
    return out


def write_tables(cfg, summ, seqs, variants, rep):
    keys = ["sequence", "variant", "runs", "crashes", "fails", "ate_median", "ate_mean", "ate_std", "ate_q1",
            "ate_q3", "rpe_t_median", "rpe_r_median", "tracked_pct", "keyframes", "tracking_ms", "switch_on_pct",
            "p_vs_baseline"]
    with open(os.path.join(rep, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for s in seqs:
            for v in variants:
                if (s, v) in summ:
                    w.writerow([summ[(s, v)][k] if not isinstance(summ[(s, v)][k], float) else f"{summ[(s, v)][k]:.6f}"
                                for k in keys])
    thr, base = cfg["fail_threshold_m"], cfg["baseline"]
    unit = "s" if cfg["rpe_delta_unit"] == "s" else "frame(s)"
    L = [f"# {cfg['name']}", "",
         f"ATE: RMSE after SE(3) alignment (as `evo_ape -a`), in m. RPE: RMSE over {cfg['rpe_delta']:g} {unit}, "
         f"translation in m, rotation in deg. A run fails if ATE > {thr:g} m or it produced no trajectory. "
         f"p: two-sided Mann-Whitney U test of ATE against **{base}** (p < 0.05 = significant). "
         f"Tracked %: frames with a real pose estimate; frames ORB-SLAM3 spent RECENTLY_LOST (repeated "
         f"timestamps in its output) are not scored, as in the official TUM tools.", "",
         "![ATE per run](ate_box.png)", "", "![RPE per run](rpe_box.png)", ""]
    for s in seqs:
        L += [f"## {short(s)}", "",
              "| Variant | Runs | Fails | ATE median | ATE mean ± std | ATE IQR | RPE trans | RPE rot | Tracked % | Keyframes | Track ms | p vs baseline |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in variants:
            d = summ.get((s, v))
            if not d:
                continue
            p = d["p_vs_baseline"]
            ptxt = "-" if p is None else (f"**{p:.4f}**" if p < 0.05 else f"{p:.4f}")
            if p is not None and p < 0.05:
                ptxt += " better" if d["ate_median"] < summ[(s, base)]["ate_median"] else " worse"
            L.append(f"| {v} | {d['runs']} | {d['fails']}" + (f" ({d['crashes']} crash)" if d["crashes"] else "") +
                     f" | {fnum(d['ate_median'])} | {fnum(d['ate_mean'])} ± {fnum(d['ate_std'])} | "
                     f"{fnum(d['ate_q1'])}–{fnum(d['ate_q3'])} | {fnum(d['rpe_t_median'])} | {fnum(d['rpe_r_median'], 2)} | "
                     f"{fnum(d['tracked_pct'], 1)} | "
                     f"{fnum(d['keyframes'], 0)} | {fnum(d['tracking_ms'], 1)} | {ptxt} |")
        L += ["", f"![trajectories](traj_{s}.png)", "", f"![error over time](error_{s}.png)", ""]
    with open(os.path.join(rep, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "axes.titlecolor": TEXT, "axes.titlesize": 10, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def box_figure(plt, cfg, rows, seqs, variants, key, ylabel, path):
    color = {v: SERIES[i % len(SERIES)] for i, v in enumerate(variants)}
    ncol = min(4, len(seqs))
    nrow = math.ceil(len(seqs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow + 0.5), squeeze=False)
    for ax, s in zip(axes.flat, seqs):
        data = [[r[key] for r in rows if r["sequence"] == s and r["variant"] == v and r["status"] == "ok"] for v in variants]
        pos = [i for i, d in enumerate(data) if d]
        if pos:
            bp = ax.boxplot([data[i] for i in pos], positions=pos, widths=0.55, patch_artist=True, showfliers=False,
                            medianprops={"color": TEXT, "linewidth": 1.5}, whiskerprops={"color": MUTED},
                            capprops={"color": MUTED}, boxprops={"linewidth": 0})
            for patch, i in zip(bp["boxes"], pos):
                patch.set_facecolor(color[variants[i]])
                patch.set_alpha(0.35)
            for i in pos:
                xs = [i + (k - (len(data[i]) - 1) / 2) * 0.04 for k in range(len(data[i]))]
                ax.scatter(xs, data[i], s=12, color=color[variants[i]], zorder=3, linewidths=0)
        crashed = [sum(1 for r in rows if r["sequence"] == s and r["variant"] == v and r["status"] != "ok") for v in variants]
        for i, c in enumerate(crashed):
            if c:
                ax.annotate(f"{c} crashed", (i, 0.0), xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=7, color=TEXT2)
        if key == "ate_rmse":
            ax.axhline(cfg["fail_threshold_m"], color=MUTED, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        fmt = plt.FuncFormatter(lambda y, _: f"{y:g}")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}" if f"{y:.0e}"[0] in "25" else ""))
        ax.set_title(short(s))
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([""] * len(variants))
        ax.set_xlim(-0.6, len(variants) - 0.4)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(seqs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=color[v], label=v) for v in variants]
    if key == "ate_rmse":
        handles.append(plt.Line2D([], [], linestyle="--", color=MUTED, label=f"failure line ({cfg['fail_threshold_m']:g} m)"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def median_run(rows, s, v):
    ok = sorted((r for r in rows if r["sequence"] == s and r["variant"] == v and r["status"] == "ok"),
                key=lambda r: r["ate_rmse"])
    return ok[(len(ok) - 1) // 2] if ok else None


def trajectory_figures(plt, cfg, rows, seqs, variants, rep):
    color = {v: SERIES[i % len(SERIES)] for i, v in enumerate(variants)}
    for s in seqs:
        gt = os.path.join(seq_dir(cfg, s), "groundtruth.txt")
        if not os.path.exists(gt):
            continue
        fig1, ax1 = plt.subplots(figsize=(5.2, 4.6))
        fig2, ax2 = plt.subplots(figsize=(6.4, 3.8))
        gt_drawn = False
        for v in variants:
            r = median_run(rows, s, v)
            if not r:
                continue
            traj = os.path.join(exp_dir(cfg), s, v, f"run_{r['run']:02d}", "traj.txt")
            _, d = tm.ate(gt, traj, cfg["max_time_diff_s"])
            if not gt_drawn:
                ax1.plot(d["gt_all"][:, 0], d["gt_all"][:, 1], color=TEXT2, linestyle="--", linewidth=1.2,
                         label="ground truth", zorder=5)
                gt_drawn = True
            lab = f"{v} (run {r['run']}, ATE {r['ate_rmse']:.3f} m)"
            ax1.plot(d["est"][:, 0], d["est"][:, 1], color=color[v], linewidth=1.5, label=lab)
            ax2.plot(d["t"] - d["t"][0], d["err"], color=color[v], linewidth=1.5, label=lab)
        if not gt_drawn:
            plt.close(fig1), plt.close(fig2)
            continue
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.set_xlabel("x [m]"), ax1.set_ylabel("y [m]")
        ax1.set_title(f"{short(s)} - median run of each variant (top view)")
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=8)
        fig1.tight_layout()
        fig1.savefig(os.path.join(rep, f"traj_{s}.png"))
        ax2.axhline(cfg["fail_threshold_m"], color=MUTED, linestyle="--", linewidth=1)
        ax2.set_xlabel("time [s]"), ax2.set_ylabel("position error [m]")
        ax2.set_title(f"{short(s)} - ATE over time, median run")
        ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
        fig2.tight_layout()
        fig2.savefig(os.path.join(rep, f"error_{s}.png"))
        plt.close(fig1), plt.close(fig2)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    root = exp_dir(cfg)
    mpath = os.path.join(root, "metrics.csv")
    if not os.path.exists(mpath):
        sys.exit(f"{mpath} missing - run evaluate.py first")
    rows = load_rows(mpath)
    seqs = [s for s in cfg["sequences"] if any(r["sequence"] == s for r in rows)]
    present = {r["variant"] for r in rows}
    variants = [v for v in cfg["variants"] if v in present] + sorted(present - set(cfg["variants"]))
    rep = os.path.join(root, "report")
    os.makedirs(rep, exist_ok=True)
    summ = summarize(cfg, rows)
    write_tables(cfg, summ, seqs, variants, rep)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing - tables written, figures skipped (pip install matplotlib)")
    else:
        style(plt)
        box_figure(plt, cfg, rows, seqs, variants, "ate_rmse", "ATE RMSE [m]", os.path.join(rep, "ate_box.png"))
        box_figure(plt, cfg, rows, seqs, variants, "rpe_t_rmse", "RPE trans. RMSE [m]", os.path.join(rep, "rpe_box.png"))
        trajectory_figures(plt, cfg, rows, seqs, variants, rep)
    # console summary
    print(f"{'sequence':26s} {'variant':16s} {'n':>3s} {'fail':>5s} {'ATE med':>8s} {'RPE t':>8s} {'RPE r':>7s} "
          f"{'track%':>7s} {'p':>7s}")
    for s in seqs:
        for v in variants:
            d = summ.get((s, v))
            if d:
                p = d["p_vs_baseline"]
                print(f"{short(s):26s} {v:16s} {d['runs']:3d} {d['fails']:5d} {fnum(d['ate_median']):>8s} "
                      f"{fnum(d['rpe_t_median']):>8s} {fnum(d['rpe_r_median'], 2):>7s} {fnum(d['tracked_pct'], 1):>7s} "
                      f"{fnum(p) if p is not None else '-':>7s}")
    print(f"report: {rep}/summary.md (+ figures)")


if __name__ == "__main__":
    main()
