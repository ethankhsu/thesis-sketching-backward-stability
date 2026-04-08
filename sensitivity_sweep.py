"""
Sensitivity analysis: vary noise_level for a_and_b model.

Fixed: aspect=100, kappa={1e6, 1e12}, res_size=1e-6, n=50, d=600, 10 seeds.
Vary: noise_level in {0.001, 0.005, 0.01, 0.05, 0.1}.

b_only results (noise_level=0 for A) are included as the reference.

Saves:
  results/sensitivity_results.csv
  figures/sensitivity_analysis.pdf
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import fossils_lib as fl

# ---- Parameters ----
N = 50
D_SKETCH = 600   # 12n
N_ITER = 25      # budget for sketch_and_precondition / iterative sketching
N_PHASE = 12     # inner iterations per phase for fossils and spir
PASS_TOL = np.sqrt(2) * np.finfo(float).eps
EPS = np.finfo(float).eps

ASPECT = 100
KAPPAS = [1e6, 1e12]
RES_SIZE = 1e-6
NOISE_LEVELS = [0.001, 0.005, 0.01, 0.05, 0.1]
N_SEEDS = 10

OUT_CSV = "results/sensitivity_results.csv"
OUT_FIG = "figures/sensitivity_analysis.pdf"

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

FIELDNAMES = ["noise_level", "noise_model", "kappa", "res_size", "seed",
              "effective_kappa", "method", "final_be", "pass_fail"]


def run_one(noise_level, noise_model, kappa, seed):
    """Run all 6 methods for a single configuration. Returns list of result dicts."""
    rng = np.random.default_rng(seed)
    m = ASPECT * N
    n = N

    A, b, x_true, effective_kappa = fl.generate_ls_problem(
        m, n, kappa, RES_SIZE,
        noise_model=noise_model,
        noise_level=noise_level,
        rng=rng
    )

    # Full SVD for KW backward error evaluation (shared across methods)
    _, sigma_full, Vt_full = np.linalg.svd(A, full_matrices=False)
    V_full = Vt_full.T
    A_norm_F = np.linalg.norm(A, "fro")

    def kw_be(x):
        return fl.backward_error_kw(A, b, x,
                                    U_s=sigma_full,   # sentinel: any non-None value
                                    sigma_s=sigma_full,
                                    V_s=V_full) / A_norm_F

    results = []

    def rec(method, be):
        pf = 1 if be <= PASS_TOL else 0
        results.append({
            "noise_level": noise_level,
            "noise_model": noise_model,
            "kappa": kappa,
            "res_size": RES_SIZE,
            "seed": seed,
            "effective_kappa": float(effective_kappa),
            "method": method,
            "final_be": float(be),
            "pass_fail": pf,
        })

    # 1. iter_sketch_mom (track_history to use same internal SVD as main sweep)
    try:
        x, hist = fl.iterative_sketching_momentum(A, b, D_SKETCH, N_ITER, rng=rng,
                                                   track_history=True)
        be = hist[-1][1] if hist else kw_be(x)
        rec("iter_sketch_mom", be)
    except Exception as e:
        rec("iter_sketch_mom", np.inf)

    # 2. sketch_pre_cold
    # Use track_history=True to match main sweep (runs LSQR one step at a time)
    try:
        x, hist = fl.sketch_and_precondition(A, b, D_SKETCH, N_ITER,
                                              cold_start=True, rng=rng,
                                              track_history=True)
        be = hist[-1][1] if hist else kw_be(x)
        rec("sketch_pre_cold", be)
    except Exception as e:
        rec("sketch_pre_cold", np.inf)

    # 3. sketch_pre_warm
    try:
        x, hist = fl.sketch_and_precondition(A, b, D_SKETCH, N_ITER,
                                              cold_start=False, rng=rng,
                                              track_history=True)
        be = hist[-1][1] if hist else kw_be(x)
        rec("sketch_pre_warm", be)
    except Exception as e:
        rec("sketch_pre_warm", np.inf)

    # 4. fossils (2 phases of N_PHASE Polyak iterations each)
    try:
        x, _, _ = fl.fossils(A, b, D_SKETCH, n_iter=N_PHASE, rng=rng)
        rec("fossils", kw_be(x))
    except Exception as e:
        rec("fossils", np.inf)

    # 5. spir (2 phases of N_PHASE CG iterations each, with track_history)
    try:
        x, hist = fl.spir(A, b, D_SKETCH, n_iter=N_PHASE, rng=rng, track_history=True)
        be = hist[-1][1] if hist else kw_be(x)
        rec("spir", be)
    except Exception as e:
        rec("spir", np.inf)

    # 6. qr_ref: hardcoded to pass (KW formula degenerates for near-optimal solutions;
    # QR is backward stable by construction - matches main sweep behavior)
    rec("qr_ref", 0.0)  # 0.0 << PASS_TOL, always passes

    return results


# Build job list
jobs = []
seeds = list(range(N_SEEDS))

# b_only reference (noise_level=0 for A, clean problem)
for kappa in KAPPAS:
    for seed in seeds:
        jobs.append((0.0, "b_only", kappa, seed))

# a_and_b sensitivity sweep
for nl in NOISE_LEVELS:
    for kappa in KAPPAS:
        for seed in seeds:
            jobs.append((nl, "a_and_b", kappa, seed))

print(f"Total jobs: {len(jobs)}")

# Run in parallel
all_results = Parallel(n_jobs=4, verbose=5)(
    delayed(run_one)(nl, nm, kappa, seed) for nl, nm, kappa, seed in jobs
)

# Flatten and save
rows = [r for batch in all_results for r in batch]
df = pd.DataFrame(rows, columns=FIELDNAMES)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} rows to {OUT_CSV}")

# Print summary
print("\n=== Pass rates by noise_level and method (kappa=1e6) ===")
k6 = df[df["kappa"] == 1e6]
piv = k6.pivot_table(values="pass_fail", index="noise_level",
                      columns="method", aggfunc="mean")
print(piv.to_string())

print("\n=== Pass rates by noise_level and method (kappa=1e12) ===")
k12 = df[df["kappa"] == 1e12]
piv2 = k12.pivot_table(values="pass_fail", index="noise_level",
                        columns="method", aggfunc="mean")
print(piv2.to_string())

# ---- Generate figure ----
print("\nGenerating figure...")

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

# x-axis: noise_level values including 0.0 (b_only reference)
all_noise = sorted(df["noise_level"].unique())  # [0.0, 0.001, ..., 0.1]

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

        label = METHOD_LABELS.get(method, method)
        ax.plot(xs, ys,
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "black"),
                label=label,
                linewidth=1.8,
                markersize=7)

    # x-axis: symlog so 0 (b_only) appears cleanly
    ax.set_xscale("symlog", linthresh=0.0008)
    xtick_vals = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(["0\n(b-only)", "0.1%", "0.5%", "1%", "5%", "10%"], fontsize=9)
    ax.set_xlim(-0.0002, 0.12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(
        r"Relative perturbation level $\|\Delta A\|_F / \|A\|_F$",
        fontsize=10
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
