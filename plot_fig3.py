import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evo.core import sync
from evo.tools import file_interface
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=os.path.expanduser("~/datasets/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt"))
ap.add_argument("--baseline", default="baseline_6.txt")
ap.add_argument("--yolo", default="yolo_9.txt")
ap.add_argument("--geo", default="geo_3.txt")
ap.add_argument("--proposed", default="proposed_6.txt")
ap.add_argument("--out", default="fig3_traj")
a = ap.parse_args()
runs = [("Baseline", a.baseline, "#2a78d6", 0.9, 2), ("YOLO", a.yolo, "#eda100", 0.9, 3),
        ("Geometry", a.geo, "#1baf7a", 1.0, 4), ("Proposed", a.proposed, "#eb6834", 1.0, 5)]
ref = file_interface.read_tum_trajectory_file(a.gt)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 3.0))
g = ref.positions_xyz
ax.plot(g[:, 0], g[:, 1], color="black", linestyle="--", linewidth=1.0, label="Ground truth", zorder=6)
for label, path, color, lw, z in runs:
    est = file_interface.read_tum_trajectory_file(path)
    ref_s, est_s = sync.associate_trajectories(ref, est)
    est_s.align(ref_s, correct_scale=False)
    p = est_s.positions_xyz
    ax.plot(p[:, 0], p[:, 1], color=color, linewidth=lw, label=label, zorder=z)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_aspect("equal", adjustable="box")
ax.grid(color="#e5e5e5", linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
          handlelength=1.4, columnspacing=0.9, handletextpad=0.4, fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("Saved", a.out + ".pdf")
