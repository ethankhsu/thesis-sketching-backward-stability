"""
Plot backward error trajectories over iterations for two perturbation modes
(b_only and a_and_b), using pre-saved per-iteration histories from npz files.

Configuration: aspect=100, kappa=1e12, res_size=1e-6, seed=32439
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
u = np.finfo(float).eps          # machine epsilon ≈ 2.22e-16
pass_threshold = np.sqrt(2) * u  # ≈ 3.14e-16

PHASE_BOUNDARY = 12  # FOSSILS / SPIR split at iteration 12

DATA_DIR = "results/be_histories"

METHOD_STYLES = {
    "iter_sketch_mom":  dict(label="Iter. Sketch + Mom.", color="red",    linestyle="-",  linewidth=1.5),
    "sketch_pre_cold":  dict(label="Sketch & Pre. (cold)", color="orange", linestyle="-",  linewidth=1.5),
    "sketch_pre_warm":  dict(label="Sketch & Pre. (warm)", color="goldenrod", linestyle="-",  linewidth=1.5),
    "fossils":          dict(label="FOSSILS",              color="blue",   linestyle="-",  linewidth=1.5),
    "spir":             dict(label="SPIR",                 color="green",  linestyle="-",  linewidth=1.5),
}

PANEL_INFO = [
    ("b_only",   r"$b$-only perturbation"),
    ("a_and_b",  r"$A$-and-$b$ perturbation"),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_npz(pert_mode):
    fname = f"aspect100_kappa1e+12_res1e-06_{pert_mode}_seed32439.npz"
    path = os.path.join(DATA_DIR, fname)
    return np.load(path)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, (pert_mode, title) in zip(axes, PANEL_INFO):
    data = load_npz(pert_mode)

    # --- iterative methods (26 values each, x = 0..25) ---
    for key, style in METHOD_STYLES.items():
        arr = data[key]
        if key == "spir":
            # SPIR has no initial iterate; 25 values → x = 1..25
            x = np.arange(1, len(arr) + 1)
        else:
            x = np.arange(len(arr))  # 0..25
        ax.semilogy(x, arr, **style)

    # --- QR reference: single value, plotted as a horizontal line ---
    qr_val = float(data["qr_ref"].flat[0])
    ax.axhline(qr_val, color="black", linestyle=":", linewidth=1.8,
               label="QR reference")

    # --- machine epsilon thresholds ---
    ax.axhline(pass_threshold, color="dimgray", linestyle="--", linewidth=1.2,
               label=r"pass threshold $\sqrt{2}\,u \approx 3.1 \times 10^{-16}$")
    ax.axhline(u, color="lightslategray", linestyle="--", linewidth=1.0,
               label=r"machine $\epsilon\, \approx 2.2 \times 10^{-16}$")

    # --- phase boundary for FOSSILS / SPIR ---
    ax.axvline(PHASE_BOUNDARY, color="black", linestyle="--", linewidth=0.9,
               alpha=0.5, label="phase boundary (iter 12)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Backward Error $\eta_{KW}(\hat{x})\,/\,\|A\|_F$")
    ax.set_title(title)
    ax.set_xlim(0, 25)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

fig.suptitle(
    r"Backward Error Trajectories: aspect$=100$, $\kappa=10^{12}$, "
    r"$\delta_{\mathrm{res}}=10^{-6}$, seed$=32439$",
    fontsize=12,
)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/be_trajectories.pdf", dpi=150, bbox_inches="tight")
fig.savefig("figures/be_trajectories.png", dpi=150, bbox_inches="tight")
print("Saved figures/be_trajectories.pdf and figures/be_trajectories.png")
plt.close()
