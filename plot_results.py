import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA
# =========================

data = {
    "walking_xyz": {
        "Baseline": [0.455413, 0.449283, 0.329930],
        "Geometry": [0.397071, 0.684214, 0.705764],
        "YOLO": [0.049461, 0.052005, 0.031933],
        "Geo+YOLO": [0.041777, 0.044668, 0.046427],
    },
    "walking_static": {
        "Baseline": [0.022611, 0.048438, 0.031159],
        "Geometry": [0.027505, 0.027933, 0.032431],
        "YOLO": [0.014903, 0.051656, 0.020161],
        "Geo+YOLO": [0.062223, 0.016729, 0.014844],
    },
    "sitting_static": {
        "Baseline": [0.010198, 0.009802, 0.011641],
        "Geometry": [0.008178, 0.009696, 0.011693],
        "YOLO": [0.009450, 0.010088, 0.009450],
        "Geo+YOLO": [0.011480, 0.009948, 0.009412],
    }
}

# =========================
# COMPUTE MEAN + STD
# =========================

results = {}

for seq, methods in data.items():
    results[seq] = {}
    for method, values in methods.items():
        mean = np.mean(values)
        std = np.std(values)
        results[seq][method] = (mean, std)

print("\n=== Mean ± Std ===")
for seq in results:
    print(f"\n{seq}")
    for method in results[seq]:
        m, s = results[seq][method]
        print(f"{method}: {m:.4f} ± {s:.4f}")

# =========================
# PLOTS
# =========================

for seq in results:
    methods = list(results[seq].keys())
    means = [results[seq][m][0] for m in methods]
    stds = [results[seq][m][1] for m in methods]

    x = np.arange(len(methods))

    # Bar chart
    plt.figure()
    plt.bar(x, means)
    plt.xticks(x, methods, rotation=30)
    plt.ylabel("RMSE")
    plt.title(f"{seq} - RMSE")
    plt.tight_layout()
    plt.savefig(f"{seq}_bar.png")
    plt.close()

    # Error bar
    plt.figure()
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5)
    plt.xticks(x, methods, rotation=30)
    plt.ylabel("RMSE")
    plt.title(f"{seq} - Mean ± Std")
    plt.tight_layout()
    plt.savefig(f"{seq}_bar.png")
    plt.close()

print("\nPlots saved successfully.")
