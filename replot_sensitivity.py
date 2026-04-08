"""
Replot figures/sensitivity_analysis.pdf from results/sensitivity_results.csv
without re-running the sweep. Mirrors the figure code in sensitivity_sweep.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "results/sensitivity_results.csv"
OUT_FIG = "figures/sensitivity_analysis.pdf"

KAPPAS = [1e6, 1e12]

METHOD_LABELS = {
    "iter_sketch_mom": "Iter. Sketch + Mom.",
    "sketch_pre_cold": "Sketch\\&Pre (cold)",
    "sketch_pre_warm": "Sketch\\&Pre (warm)",
    "fossils": "FOSSILS",
    "spir": "SPIR",
    "qr_ref": "QR reference",
}
METHOD_COLORS = {
    "iter_sketch_mom": "#999999",
    "sketch_pre_cold": "#fdae61",
    "sketch_pre_warm": "#d7191c",
    "fossils": "#74add1",
    "spir": "#2166ac",
    "qr_ref": "#1a9850",
}
METHOD_MARKERS = {
    "iter_sketch_mom": "s",
    "sketch_pre_cold": "^",
    "sketch_pre_warm": "o",
    "fossils": "D",
    "spir": "v",
    "qr_ref": "*",
}
METHODS_ORDER = ["iter_sketch_mom", "sketch_pre_cold", "sketch_pre_warm",
                 "fossils", "spir", "qr_ref"]

os.makedirs("figures", exist_ok=True)
df = pd.read_csv(IN_CSV)
all_noise = sorted(df["noise_level"].unique())

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle(
    "Sensitivity to perturbation level: backward stability pass rates",
    fontsize=13
)

for ax, kappa in zip(axes, KAPPAS):
    kdf = df[df["kappa"] == kappa]
    for method in METHODS_ORDER:
        mdf = kdf[kdf["method"] == method]
        xs, ys = [], []
        for nl in all_noise:
            sub = mdf[mdf["noise_level"] == nl]
            if len(sub) > 0:
                xs.append(nl)
                ys.append(sub["pass_fail"].mean())
        ax.plot(
            xs, ys,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.8,
            markersize=7,
        )

    ax.set_xscale("symlog", linthresh=0.0008)
    xtick_vals = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(["0\n(b-only)", "0.1%", "0.5%", "1%", "5%", "10%"], fontsize=9)
    ax.set_xlim(-0.0002, 0.12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(
        r"Relative perturbation level $\|\Delta A\|_F / \|A\|_F$",
        fontsize=10,
    )
    kexp = int(round(np.log10(kappa)))
    ax.set_title(f"$\\kappa = 10^{{{kexp}}}$", fontsize=12)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

axes[0].set_ylabel("Backward stability pass rate", fontsize=11)
axes[1].legend(loc="center right", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT_FIG, bbox_inches="tight")
print(f"Saved figure to {OUT_FIG}")

# Print summary tables
for kappa in KAPPAS:
    print(f"\n=== Pass rates at kappa={kappa:.0e} ===")
    kdf = df[df["kappa"] == kappa]
    piv = kdf.pivot_table(values="pass_fail", index="noise_level",
                          columns="method", aggfunc="mean")
    print(piv.to_string())
