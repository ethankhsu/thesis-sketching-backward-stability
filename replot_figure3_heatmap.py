"""
Regenerate figures/figure3_heatmap.png from results/sweep_results.csv.

Pass rate heatmap: rows = methods, columns = kappa, one panel per noise
model. Matches the figure referenced at code.tex line 93.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "results/sweep_results.csv"
OUT_PNG = "figures/figure3_heatmap.png"
OUT_PDF = "figures/figure3_heatmap.pdf"

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
KAPPAS = [1e3, 1e6, 1e9, 1e12, 1e14]

os.makedirs("figures", exist_ok=True)
df = pd.read_csv(IN_CSV)
methods_in_csv = set(df["method"].unique())


def pivot_for_noise(noise_model):
    mat = np.full((len(METHODS_ORDER), len(KAPPAS)), np.nan)
    for i, method in enumerate(METHODS_ORDER):
        for j, kappa in enumerate(KAPPAS):
            if method == "qr_ref" and method not in methods_in_csv:
                mat[i, j] = 1.0
                continue
            sub = df[
                (df["method"] == method)
                & (df["noise_model"] == noise_model)
                & (np.isclose(df["kappa"], kappa, rtol=1e-6))
            ]
            if len(sub) > 0:
                mat[i, j] = sub["pass_fail"].mean()
    return mat


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
noise_panels = [
    ("b_only", r"$b$-only noise"),
    ("a_and_b", r"$A$-and-$b$ noise"),
]

cmap = plt.cm.RdYlGn
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

last_im = None
for ax, (nm, title) in zip(axes, noise_panels):
    mat = pivot_for_noise(nm)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    last_im = im

    ax.set_xticks(np.arange(len(KAPPAS)))
    ax.set_xticklabels([f"$10^{{{int(round(np.log10(k)))}}}$" for k in KAPPAS])
    ax.set_yticks(np.arange(len(METHODS_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS_ORDER])
    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_title(title)

    # Annotate each cell with pass rate
    for i in range(len(METHODS_ORDER)):
        for j in range(len(KAPPAS)):
            v = mat[i, j]
            if np.isnan(v):
                txt = "-"
            else:
                txt = f"{int(round(v*100))}%"
            # Contrasting text color
            tc = "white" if v < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=tc, fontsize=9)

axes[0].set_ylabel("Method")

fig.suptitle("Backward stability pass rate by method and condition number",
             fontsize=13)
cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Pass rate")
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved {OUT_PNG} and {OUT_PDF}")

# Print summary
for nm, _ in noise_panels:
    print(f"\n=== {nm} pass rates (rows=methods, cols=kappa) ===")
    mat = pivot_for_noise(nm)
    header = "method".ljust(22) + "".join([f"{k:>10.0e}" for k in KAPPAS])
    print(header)
    print("-" * len(header))
    for i, method in enumerate(METHODS_ORDER):
        row = METHOD_LABELS[method].ljust(22)
        for j in range(len(KAPPAS)):
            row += f"{mat[i,j]*100:>9.1f}%"
        print(row)
