"""
Replot figures/real_data_pass_rates.pdf from results/real_data_summary.csv
without re-running the real-data sweep. Mirrors make_figure() in
real_data_experiments.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "results/real_data_summary.csv"
OUT_FIG = "figures/real_data_pass_rates.pdf"

METHODS_FOR_FIG = [
    "sketch_solve",
    "sketch_pre_warm",
    "fossils",
    "spir",
    "qr_ref",
]
METHOD_LABELS = {
    "sketch_solve": "Sketch-and-solve",
    "sketch_pre_warm": "Sketch&Pre (warm)",
    "fossils": "FOSSILS",
    "spir": "SPIR",
    "qr_ref": "QR reference",
}

os.makedirs("figures", exist_ok=True)
summary = pd.read_csv(IN_CSV)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
noise_order = [("b_only", r"$b$-only"), ("a_and_b", r"$A$-and-$b$")]
dataset_order = list(summary["dataset_id"].drop_duplicates())
x = np.arange(len(dataset_order))
width = 0.14

for ax, (noise_model, title) in zip(axes, noise_order):
    noise_df = summary[
        (summary["noise_model"] == noise_model)
        & (summary["method"].isin(METHODS_FOR_FIG))
    ].copy()

    for idx, method in enumerate(METHODS_FOR_FIG):
        method_df = noise_df[noise_df["method"] == method]
        rates = []
        for dataset_id in dataset_order:
            sub = method_df[method_df["dataset_id"] == dataset_id]
            rates.append(sub["pass_rate"].iloc[0] if not sub.empty else 0.0)

        ax.bar(
            x + (idx - 2) * width,
            rates,
            width=width,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_xlabel("Dataset")

axes[0].set_ylabel("Backward stability pass rate")
axes[1].legend(loc="lower right", fontsize=9)
fig.suptitle("Real-data validation: pass rates by dataset and noise model")
plt.tight_layout()
plt.savefig(OUT_FIG, bbox_inches="tight")
print(f"Saved figure to {OUT_FIG}")

# Print summary table
report = summary[summary["method"].isin(METHODS_FOR_FIG)].copy()
print(
    report.pivot_table(
        values="pass_rate",
        index=["dataset_id", "noise_model"],
        columns="method",
    ).to_string()
)
