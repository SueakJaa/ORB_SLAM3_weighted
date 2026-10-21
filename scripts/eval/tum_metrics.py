"""Trajectory metrics for TUM RGB-D, without external SLAM-evaluation packages.

Conventions (match evo, which earlier results in this project used):
  - timestamps associated nearest-neighbour, |dt| <= max_diff (default 0.01 s, evo's default)
  - ATE: SE(3) Umeyama alignment of the estimate to ground truth (no scale, RGB-D), then RMSE etc.
    of the translational error  -> same number as `evo_ape tum gt est -a`
  - RPE: relative pose error over a fixed delta, no alignment needed.
      delta in frames  (delta_unit="f", evo_rpe default, delta=1)
      delta in seconds (delta_unit="s", TUM benchmark convention, delta=1.0 -> drift per second)
    translational error in m, rotational error in degrees.
"""
import numpy as np


def load_tum(path):
    """Return (stamps[N], positions[N,3], quaternions[N,4] as x,y,z,w). Skips comments/empty lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            v = line.replace(",", " ").split()
            if len(v) >= 8:
                rows.append([float(x) for x in v[:8]])
    if not rows:
        raise ValueError(f"no poses in {path}")
    a = np.array(rows)
    order = np.argsort(a[:, 0], kind="stable")
    a = a[order]
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def quat_to_rot(q):
    """q[..., 4] as x,y,z,w -> R[..., 3, 3]"""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def to_se3(p, q):
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = quat_to_rot(q)
    T[:, :3, 3] = p
    return T


def associate(t_ref, t_est, max_diff=0.01):
    """Indices (i_ref, i_est) of matched stamps: for every estimate stamp the nearest reference stamp
    within max_diff; each reference stamp used at most once (closest wins)."""
    idx = np.searchsorted(t_ref, t_est)
    idx = np.clip(idx, 1, len(t_ref) - 1)
    left, right = t_ref[idx - 1], t_ref[idx]
    nearest = np.where(np.abs(t_est - left) <= np.abs(t_est - right), idx - 1, idx)
    diff = np.abs(t_ref[nearest] - t_est)
    ok = diff <= max_diff
    best = {}
    for j in np.nonzero(ok)[0]:
        i = nearest[j]
        if i not in best or diff[j] < diff[best[i]]:
            best[i] = j
    pairs = sorted((i, j) for i, j in best.items())
    if not pairs:
        return np.array([], int), np.array([], int)
    a = np.array(pairs)
    order = np.argsort(a[:, 1])
    return a[order, 0], a[order, 1]


def umeyama(src, dst, with_scale=False):
    """Least-squares R, t, s with dst ~ s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def _stats(e):
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mean": float(np.mean(e)), "median": float(np.median(e)),
            "std": float(np.std(e)), "min": float(np.min(e)), "max": float(np.max(e)), "n": int(len(e))}


def ate(gt_path, est_path, max_diff=0.01, with_scale=False):
    """Absolute trajectory error. Returns (stats dict, details dict for plotting)."""
    tg, pg, qg = load_tum(gt_path)
    te, pe, qe = load_tum(est_path)
    ig, ie = associate(tg, te, max_diff)
    if len(ig) < 3:
        raise ValueError(f"only {len(ig)} matched poses between {gt_path} and {est_path}")
    R, t, s = umeyama(pe[ie], pg[ig], with_scale)
    pe_al = (s * (R @ pe[ie].T)).T + t
    err = np.linalg.norm(pe_al - pg[ig], axis=1)
    st = _stats(err)
    st["coverage"] = float(len(ie) / len(te))
    return st, {"t": te[ie], "gt": pg[ig], "est": pe_al, "err": err, "gt_all": pg}


def rpe(gt_path, est_path, delta=1.0, delta_unit="s", max_diff=0.01):
    """Relative pose error. delta_unit "s": pairs ~delta seconds apart (TUM convention),
    "f": pairs delta matched frames apart (evo default). Returns (trans stats [m], rot stats [deg])."""
    tg, pg, qg = load_tum(gt_path)
    te, pe, qe = load_tum(est_path)
    ig, ie = associate(tg, te, max_diff)
    G, E = to_se3(pg[ig], qg[ig]), to_se3(pe[ie], qe[ie])
    ts = te[ie]
    pairs = []
    if delta_unit == "f":
        d = int(delta)
        pairs = [(i, i + d) for i in range(len(ts) - d)]
    else:
        j = 0
        for i in range(len(ts)):
            j = max(j, i + 1)
            while j < len(ts) and ts[j] - ts[i] < delta:
                j += 1
            if j < len(ts):
                pairs.append((i, j))
    if not pairs:
        raise ValueError("trajectory too short for this RPE delta")
    i, j = np.array(pairs).T
    rel_g = np.linalg.inv(G[i]) @ G[j]
    rel_e = np.linalg.inv(E[i]) @ E[j]
    err = np.linalg.inv(rel_g) @ rel_e
    trans = np.linalg.norm(err[:, :3, 3], axis=1)
    cosang = np.clip((np.trace(err[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
    rot = np.degrees(np.arccos(cosang))
    return _stats(trans), _stats(rot)
