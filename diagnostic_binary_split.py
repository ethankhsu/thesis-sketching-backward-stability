"""
Diagnostic plot: kappa * res_size vs pass/fail for sketch_pre_warm.

The sweep holds res_size as a fixed parameter, so kappa * res_size spans
many orders of magnitude across configurations. If Corollary 2.10 (forward
stability implies backward stability when kappa * ||b - Ax|| is small)
were the dominant mechanism, we would expect pass/fail to correlate with
this product. This plot tests whether that is Outcome A (correlated) or
Outcome B (not correlated, product spans 11+ orders yet pass rate is flat
within each noise model).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

CSV_PATH = "results/sweep_results.csv"
OUT_PATH = "figures/diagnostic_binary_split.pdf"

os.makedirs("figures", exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Filter to sketch_pre_warm only
warm = df[df["method"] == "sketch_pre_warm"].copy()

# Compute kappa * res_size (this is kappa * ||b - Ax|| in the clean-A case)
warm["kappa_times_res"] = warm["kappa"] * warm["res_size"]

# Separate by noise model
b_only = warm[warm["noise_model"] == "b_only"]
a_and_b = warm[warm["noise_model"] == "a_and_b"]

# Print summary statistics
print("=== Diagnostic: kappa * res_size vs pass/fail for sketch_pre_warm ===\n")
print(f"Total sketch_pre_warm rows: {len(warm)}")
print(f"  b_only: {len(b_only)} rows, pass rate: {b_only['pass_fail'].mean():.1%}")
print(f"  a_and_b: {len(a_and_b)} rows, pass rate: {a_and_b['pass_fail'].mean():.1%}")
print()

for nm, grp in [("b_only", b_only), ("a_and_b", a_and_b)]:
    prod = grp["kappa_times_res"]
    print(f"{nm}: kappa*res_size range: [{prod.min():.2e}, {prod.max():.2e}]")
    print(f"  log10 range: {np.log10(prod.max()) - np.log10(prod.min()):.1f} orders of magnitude")
    pf = grp.groupby("kappa_times_res")["pass_fail"].mean()
    print(f"  pass rate by kappa*res_size:")
    for val, rate in sorted(pf.items()):
        print(f"    {val:.2e}: {rate:.1%}")
    print()

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fig.suptitle(
    r"Diagnostic: $\kappa \cdot \|r\|$ vs backward stability (sketch\&pre warm start)",
    fontsize=13
)

colors = {"pass": "#2166ac", "fail": "#d73027"}
noise_models = [("b_only", b_only, r"$b$-only noise"), ("a_and_b", a_and_b, r"$A$-and-$b$ noise")]

for ax, (nm, grp, title) in zip(axes, noise_models):
    passed = grp[grp["pass_fail"] == 1]
    failed = grp[grp["pass_fail"] == 0]

    # Jitter y positions
    rng = np.random.default_rng(42)
    y_pass = np.ones(len(passed)) + rng.uniform(-0.05, 0.05, len(passed))
    y_fail = np.zeros(len(failed)) + rng.uniform(-0.05, 0.05, len(failed))

    ax.scatter(passed["kappa_times_res"], y_pass, color=colors["pass"],
               alpha=0.4, s=18, label=f"Pass (n={len(passed)})", zorder=3)
    ax.scatter(failed["kappa_times_res"], y_fail, color=colors["fail"],
               alpha=0.4, s=18, label=f"Fail (n={len(failed)})", zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(left=1e-11, right=1e14)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlabel(r"$\kappa \cdot \|r\|$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.3, 1.3)

    # Annotate pass rate
    rate = grp["pass_fail"].mean()
    ax.text(0.02, 0.95, f"Overall pass rate: {rate:.1%}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

axes[0].set_ylabel("Backward stability", fontsize=12)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved figure to {OUT_PATH}")
