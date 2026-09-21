import numpy as np
import matplotlib.pyplot as plt

# =========================
# UPDATED EXPERIMENT DATA
# =========================

data = {
    "walking_xyz": {
        "Baseline": [
            0.657461, 0.692609, 0.679323, 0.758638, 0.691814
        ],
        "YOLO": [
            0.222953, 0.224134, 0.244606, 0.333463, 0.137360
        ],
        "Geometry": [
            0.040598, 0.040796, 0.041773, 0.036005, 0.049999
        ],
        "Proposed": [
            0.066216, 0.035967, 0.039495, 0.038149, 0.036179
        ],
    },

    "walking_static": {
        "Baseline": [
            0.034167, 0.040230, 0.030429, 0.030708, 0.019348
        ],
        "YOLO": [
            0.023163, 0.028182, 0.039709, 0.032233, 0.031951
        ],
        "Geometry": [
            0.026390, 0.031711, 0.034673, 0.020823, 0.048151
        ],
        "Proposed": [
            0.026605, 0.018345, 0.017936, 0.025901, 0.026861
        ],
    },

    "sitting_static": {
        "Baseline": [
            0.011188, 0.010379, 0.011085, 0.010232, 0.010554
        ],
        "YOLO": [
            0.013119, 0.015204, 0.013994, 0.013288, 0.014237
        ],
        "Geometry": [
            0.009736, 0.011121, 0.010667, 0.010131, 0.010205
        ],
        "Proposed": [
            0.013658, 0.009889, 0.011207, 0.014170, 0.010192
        ],
    }
}

# =========================
# COMPUTE MEAN + STD
# =========================

results = {}

print("\n=== Mean ± Std ===")

for seq, methods in data.items():
    print(f"\n{seq}")
    results[seq] = {}

    for method, values in methods.items():
        mean = np.mean(values)
        std = np.std(values)

        results[seq][method] = (mean, std)

        print(f"{method}: {mean:.4f} ± {std:.4f}")

# =========================
# PLOTS
# =========================

for seq in results:

    methods = ["Baseline", "YOLO", "Geometry", "Proposed"]

    means = [results[seq][m][0] for m in methods]
    stds = [results[seq][m][1] for m in methods]

    x = np.arange(len(methods))

    # =========================
    # BAR PLOT
    # =========================

    plt.figure(figsize=(8, 6))

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        edgecolor="black"
    )

    plt.xticks(x, methods, fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylabel("ATE RMSE (m)", fontsize=16)
    plt.title(
        f"{seq.replace('_', ' ').title()}",
        fontsize=18
    )

    plt.tight_layout()

    plt.savefig(
        f"{seq}_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================
    # ERROR BAR PLOT
    # =========================

    plt.figure(figsize=(8, 6))

    for i, method in enumerate(methods):

        plt.errorbar(
            x[i],
            means[i],
            yerr=stds[i],
            fmt="o",
            markersize=10,
            capsize=6,
            linewidth=2,
            label=method
        )

    plt.xticks(x, methods, fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylabel("ATE RMSE (m)", fontsize=16)
    plt.title(
        f"{seq.replace('_', ' ').title()}",
        fontsize=18
    )

    plt.legend(fontsize=12)

    plt.tight_layout()

    plt.savefig(
        f"{seq}_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nPlots saved.")