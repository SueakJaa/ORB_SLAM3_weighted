#!/usr/bin/env python3
"""Is variant A really better than variant B? Two-sided Mann-Whitney U test per sequence.

Usage (from ~/ORB_SLAM3):
    python3 scripts/compare_variants.py results_v16 GEOK2_G10 SW_G10
    python3 scripts/compare_variants.py results_v16 GEOK2_G10 A0_orbslam3 results_v16_heldout

Reads <root>/<sequence>/<variant>_<run>.eval (evo_ape output, line "rmse <value>").
p < 0.05: the difference is unlikely to be chance. With 10 runs each, small differences
(e.g. 0.025 vs 0.027 m) usually do NOT reach significance - report them as "no difference".
"""
import glob
import itertools
import math
import os
import re
import sys


def rmse_values(root, seq, var):
    vals = []
    for p in glob.glob(os.path.join(root, seq, f"{var}_*.eval")):
        if not re.search(rf"{re.escape(var)}_\d+\.eval$", p):
            continue
        for line in open(p):
            if line.strip().startswith("rmse"):
                vals.append(float(line.split()[1]))
    return sorted(vals)


def mann_whitney_p(a, b):
    """Two-sided p-value. Exact (all splits) when small, otherwise normal approximation."""
    try:
        from scipy.stats import mannwhitneyu
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ImportError:
        pass
    allv = a + b
    ranks = {}
    order = sorted(allv)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j < len(order) and order[j] == order[i]:
            j += 1
        ranks[order[i]] = (i + j + 1) / 2.0
        i = j
    r = [ranks[v] for v in allv]
    na, n = len(a), len(allv)
    obs = sum(r[:na])
    mean = na * (n + 1) / 2.0
    if math.comb(n, na) <= 300000:
        dev = abs(obs - mean)
        hits = total = 0
        for idx in itertools.combinations(range(n), na):
            total += 1
            if abs(sum(r[k] for k in idx) - mean) >= dev - 1e-9:
                hits += 1
        return hits / total
    sd = math.sqrt(na * (n - na) * (n + 1) / 12.0)
    z = (abs(obs - mean) - 0.5) / sd
    return math.erfc(z / math.sqrt(2))


def median(v):
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def main(root, va, vb, *more_roots):
    print(f"{'sequence':24s} {va:>12s} {vb:>12s}   (median ATE m, n)   p-value   verdict")
    for r in (root,) + more_roots:
        for seq in sorted(d for d in os.listdir(r) if os.path.isdir(os.path.join(r, d))):
            a, b = rmse_values(r, seq, va), rmse_values(r, seq, vb)
            if len(a) < 3 or len(b) < 3:
                continue
            p = mann_whitney_p(a, b)
            if p < 0.05:
                verdict = f"{va if median(a) < median(b) else vb} better"
            else:
                verdict = "no significant difference"
            print(f"{seq:24s} {median(a):9.4f} n={len(a):<2d} {median(b):9.4f} n={len(b):<2d}"
                  f"             {p:7.4f}   {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
