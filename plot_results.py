import numpy as np
import matplotlib.pyplot as plt

# =========================
# UPDATED DATA
# =========================

data = {
    "walking_xyz": {
        "Baseline": [0.758870,0.425429,0.788755,0.717999,0.719808],
        "Geometry": [0.418522,0.637031,0.685153,0.833011,0.722118],
        "YOLO": [0.043100,0.045561,0.056734,0.044642,0.106871,
                 0.105466,0.041302,0.053814,0.035242,0.055942],
        "Ours": [0.042774,0.047745,0.052069,0.047831,0.047976,
                 0.033860,0.049766,0.048785,0.057513,0.053886],
    },

    "walking_static": {
        "Baseline": [0.232082,0.165318,0.037212,0.026310,0.273137],
        "Geometry": [0.035628,0.021486,0.022071,0.028976,0.031180],
        "YOLO": [0.014210,0.018311,0.033856,0.041643,0.018515],
        "Ours": [0.017551,0.025077,0.016123,0.028981,0.017450],
    },

    "sitting_static": {
        "Baseline": [0.010065,0.010608,0.011596],
        "Geometry": [0.009670,0.010316,0.010619,0.010375,0.010733],
        "YOLO": [0.009326,0.009386,0.010840,0.010107,0.009309],
        "Ours": [0.009761,0.008617,0.009057,0.010284,0.009949],
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
    methods = list(results[seq].keys())
    means = [results[seq][m][0] for m in methods]
    stds = [results[seq][m][1] for m in methods]

    x = np.arange(len(methods))

    # Bar
    plt.figure()
    plt.bar(x, means)
    plt.xticks(x, methods, rotation=30)
    plt.ylabel("RMSE")
    plt.title(f"{seq} RMSE Comparison")
    plt.tight_layout()
    plt.savefig(f"{seq}_bar.png", dpi=300)
    plt.close()

    # Error bar
    plt.figure()
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5)
    plt.xticks(x, methods, rotation=30)
    plt.ylabel("RMSE")
    plt.title(f"{seq} Mean ± Std")
    plt.tight_layout()
    plt.savefig(f"{seq}_error.png", dpi=300)
    plt.close()

print("\nPlots saved.")