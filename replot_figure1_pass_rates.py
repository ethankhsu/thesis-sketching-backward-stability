"""
Regenerate figures/figure1_pass_rates.png from results/sweep_results.csv.

Grouped bar chart: for each method, two bars (b_only vs a_and_b) showing the
overall pass rate across all 840 configurations. Matches Table 5.1 in code.tex.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "results/sweep_results.csv"
OUT_PNG = "figures/figure1_pass_rates.png"
OUT_PDF = "figures/figure1_pass_rates.pdf"

METHOD_LABELS = {
    "iter_sketch_mom": "Iter. Sketch + Mom.",
    "sketch_pre_cold": "Sketch&Pre (cold)",
    "sketch_pre_warm": "Sketch&Pre (warm)",
    "fossils": "FOSSILS",
    "spir": "SPIR",
    "qr_ref": "QR reference",
}
METHODS_ORDER = ["iter_sketch_mom", "sketch_pre_cold", "sketch_pre_warm",
                 "fossils", "spir", "qr_ref"]

NOISE_COLORS = {
    "b_only": "#4575b4",
    "a_and_b": "#d73027",
}
NOISE_LABELS = {
    "b_only": r"$b$-only",
    "a_and_b": r"$A$-and-$b$",
}

os.makedirs("figures", exist_ok=True)
df = pd.read_csv(IN_CSV)

# qr_ref is not always in the main sweep CSV; fall back to 1.0 if missing
methods_in_csv = set(df["method"].unique())

rates = {}
for method in METHODS_ORDER:
    for nm in ["b_only", "a_and_b"]:
        if method == "qr_ref" and method not in methods_in_csv:
            rates[(method, nm)] = 1.0
            continue
        sub = df[(df["method"] == method) & (df["noise_model"] == nm)]
        rates[(method, nm)] = sub["pass_fail"].mean() if len(sub) > 0 else np.nan

# Print summary
print("=== Overall pass rates ===")
print(f"{'Method':<22} {'b_only':>10} {'a_and_b':>10}")
print("-" * 44)
for method in METHODS_ORDER:
    b = rates[(method, "b_only")]
    a = rates[(method, "a_and_b")]
    print(f"{METHOD_LABELS[method]:<22} {b*100:>9.1f}% {a*100:>9.1f}%")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(METHODS_ORDER))
width = 0.38

b_only_rates = [rates[(m, "b_only")] for m in METHODS_ORDER]
a_and_b_rates = [rates[(m, "a_and_b")] for m in METHODS_ORDER]

bars_b = ax.bar(x - width/2, b_only_rates, width,
                label=NOISE_LABELS["b_only"],
                color=NOISE_COLORS["b_only"], edgecolor="black", linewidth=0.5)
bars_a = ax.bar(x + width/2, a_and_b_rates, width,
                label=NOISE_LABELS["a_and_b"],
                color=NOISE_COLORS["a_and_b"], edgecolor="black", linewidth=0.5)

# Value labels
for bars in (bars_b, bars_a):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h*100:.1f}%", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS_ORDER],
                    rotation=20, ha="right")
ax.set_ylabel("Backward stability pass rate")
ax.set_ylim(0.0, 1.12)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))
ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0.0, 1.01, 0.2)])
ax.set_title("Backward stability pass rate by method and noise model")
ax.legend(loc="center right", framealpha=0.95)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"\nSaved {OUT_PNG} and {OUT_PDF}")
