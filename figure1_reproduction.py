"""
Reproduction of Figure 1 from Epperly, Meier, Nakatsukasa (2024):
"Fast randomized least-squares solvers can be just as accurate and stable
as classical direct solvers"

Figure 1: Forward (left) and backward (right) error for different LS solvers
across iterations, on a 4000x50 problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import lstsq, svd, qr, solve_triangular
from fossils_lib import (
    generate_ls_problem, sparse_sign_embedding, sketch_and_solve,
    fossils, iterative_sketching_momentum, sketch_and_precondition,
    backward_error_kw, backward_error_ls
)

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ---------------------------------------------------------------------------
# Problem setup (matching paper: 4000 x 50, moderate difficulty)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(32439)  # match paper seed
m, n = 4000, 50
kappa = 1e12      # match paper exactly
res_size = 1e-6   # match paper exactly

A, b, x_true = generate_ls_problem(m, n, kappa, res_size, rng=rng)
A_norm_fro = np.linalg.norm(A, 'fro')
u = np.finfo(float).eps

# QR reference solution
x_qr, _, _, _ = lstsq(A, b)
fe_qr = np.linalg.norm(x_qr - x_true) / max(np.linalg.norm(x_true), 1e-300)
be_qr = backward_error_ls(A, b, x_qr, theta=np.inf) / A_norm_fro

print(f"Problem: m={m}, n={n}, kappa={kappa:.0e}")
print(f"Residual norm: {np.linalg.norm(b - A @ x_true):.2e}")
print(f"QR forward error:  {fe_qr:.2e}")
print(f"QR backward error: {be_qr:.2e}")
print(f"Machine eps: {u:.2e}")

# ---------------------------------------------------------------------------
# Run solvers and collect per-iteration histories
# ---------------------------------------------------------------------------
n_iter = 25
n_iter_fossils_phase = 12  # split budget: 12 + 13 = 25 total
d_sketch = 12 * n  # d = 600 for sketch dimension
rng2 = np.random.default_rng(42)

print("\nRunning iterative sketching with momentum...")
_, hist_is = iterative_sketching_momentum(
    A, b, d=d_sketch, n_iter=n_iter, rng=np.random.default_rng(42),
    track_history=True, x_true=x_true
)

print("Running sketch-and-precondition (cold start)...")
_, hist_sap_cold = sketch_and_precondition(
    A, b, d=d_sketch, n_iter=n_iter, cold_start=True, rng=np.random.default_rng(42),
    track_history=True, x_true=x_true
)

print("Running sketch-and-precondition (warm start)...")
_, hist_sap_warm = sketch_and_precondition(
    A, b, d=d_sketch, n_iter=n_iter, cold_start=False, rng=np.random.default_rng(42),
    track_history=True, x_true=x_true
)

print("Running FOSSILS...")
# FOSSILS only has 2 refinement steps (each with n_iter Polyak iterations)
# Track the sketch-and-solve init, after step 1, and after step 2
S_fossils = sparse_sign_embedding(m, d_sketch, rng=np.random.default_rng(42))
SA = S_fossils @ A
Sb = S_fossils @ b
U_s, sigma_s, Vt_s = svd(SA, full_matrices=False)
V_s = Vt_s.T
eta = np.sqrt(n / d_sketch)
alpha = (1 - eta**2)**2
beta_pb = eta**2

from scipy.linalg import solve_triangular

def polyak_iter_track(A, b_rhs, V_s, sigma_s, alpha, beta, n_iter, x_base,
                       x_true, A_norm_fro, sigma_s_full=None, V_s_full=None):
    """Run Polyak and track FE/BE at each iteration."""
    inv_sigma = 1.0 / sigma_s

    def apply_P(y):  return V_s @ (inv_sigma * y)
    def apply_Pt(z): return inv_sigma * (V_s.T @ z)

    c = apply_Pt(A.T @ b_rhs)
    dy = c.copy()
    dy_old = c.copy()

    history = []
    for i in range(n_iter):
        Mdy = apply_Pt(A.T @ (A @ apply_P(dy)))
        delta = alpha * (c - Mdy) + beta * (dy - dy_old)
        dy_old = dy.copy()
        dy = dy + delta
        x_cur = x_base + apply_P(dy)
        fe = np.linalg.norm(x_cur - x_true) / max(np.linalg.norm(x_true), 1e-300)
        be = backward_error_kw(A, b, x_cur, theta=np.inf, sigma_s=sigma_s, V_s=V_s)
        history.append((i + 1, fe, be / A_norm_fro))
    return apply_P(dy), history

# FOSSILS step 0: sketch-and-solve init
x0_f = V_s @ (np.diag(1.0 / sigma_s) @ (U_s.T @ Sb))
fe0 = np.linalg.norm(x0_f - x_true) / max(np.linalg.norm(x_true), 1e-300)
be0 = backward_error_kw(A, b, x0_f, theta=np.inf, sigma_s=sigma_s, V_s=V_s) / A_norm_fro

print("  Polyak refinement step 1...")
dx1, hist_fossils_1 = polyak_iter_track(
    A, b - A @ x0_f, V_s, sigma_s, alpha, beta_pb, n_iter_fossils_phase, x0_f,
    x_true, A_norm_fro
)
x1_f = x0_f + dx1

phase2_start = n_iter_fossils_phase + 1
phase2_iters = n_iter - n_iter_fossils_phase  # 13

print("  Polyak refinement step 2...")
dx2, hist_fossils_2 = polyak_iter_track(
    A, b - A @ x1_f, V_s, sigma_s, alpha, beta_pb, phase2_iters, x1_f,
    x_true, A_norm_fro
)
x2_f = x1_f + dx2

# Combine FOSSILS history: 0 = init, 1..12 = phase1, 13..25 = phase2
fossils_iters = [0] + [i for i, _, _ in hist_fossils_1] + \
                [n_iter_fossils_phase + i for i, _, _ in hist_fossils_2]
fossils_fe = [fe0] + [fe for _, fe, _ in hist_fossils_1] + \
             [fe for _, fe, _ in hist_fossils_2]
fossils_be = [be0] + [be for _, _, be in hist_fossils_1] + \
             [be for _, _, be in hist_fossils_2]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Extract histories
def extract(hist):
    iters = [h[0] for h in hist]
    fe    = [h[1] for h in hist]
    be    = [h[2] for h in hist]
    return iters, fe, be

iters_is, fe_is, be_is = extract(hist_is)
iters_cold, fe_cold, be_cold = extract(hist_sap_cold)
iters_warm, fe_warm, be_warm = extract(hist_sap_warm)

# --- Forward error ---
ax1.semilogy(iters_cold, fe_cold, 's-', color='tab:orange',
             label=r'Sketch&Pre ($x_0=0$)', linewidth=1.5, markersize=5)
ax1.semilogy(iters_is, fe_is, 'o-', color='tab:blue',
             label='Iter Sketch + Mom', linewidth=1.5, markersize=5)
ax1.semilogy(iters_warm, fe_warm, 'D-', color='tab:green',
             label='Sketch&Pre', linewidth=1.5, markersize=5)
ax1.semilogy(fossils_iters, fossils_fe, '^-', color='tab:red',
             label='FOSSILS', linewidth=1.5, markersize=5)
ax1.axhline(fe_qr, color='k', linestyle='--', linewidth=1.5, label='QR')
ax1.set_xlabel('Iteration $i$')
ax1.set_ylabel(r'Forward error $\|x - \hat{x}_i\| / \|x\|$')
ax1.set_title('Forward Error')
ax1.legend(loc='upper right')
ax1.set_xlim(0, n_iter)
ax1.grid(True, which='both', alpha=0.3)

# --- Backward error ---
ax2.semilogy(iters_cold, be_cold, 's-', color='tab:orange',
             label=r'Sketch&Pre ($x_0=0$)', linewidth=1.5, markersize=5)
ax2.semilogy(iters_is, be_is, 'o-', color='tab:blue',
             label='Iter Sketch + Mom', linewidth=1.5, markersize=5)
ax2.semilogy(iters_warm, be_warm, 'D-', color='tab:green',
             label='Sketch&Pre', linewidth=1.5, markersize=5)
ax2.semilogy(fossils_iters, fossils_be, '^-', color='tab:red',
             label='FOSSILS', linewidth=1.5, markersize=5)
ax2.axhline(be_qr, color='k', linestyle='--', linewidth=1.5, label='QR')
ax2.set_xlabel('Iteration $i$')
ax2.set_ylabel(r'Backward error $\mathrm{BE}_1(\hat{x}_i) / \|A\|_F$')
ax2.set_title('Backward Error')
ax2.legend(loc='upper right')
ax2.set_xlim(0, n_iter)
ax2.grid(True, which='both', alpha=0.3)

fig.suptitle(
    f'Figure 1 Reproduction: $m={m}$, $n={n}$, $\\kappa={kappa:.0e}$',
    fontsize=13
)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/figure1_reproduction.png', dpi=150, bbox_inches='tight')
print("\nSaved figure1_reproduction.png")
plt.close()

print(f"\nFinal results:")
print(f"  Iter Sketch+Mom  FE={fe_is[-1]:.2e}  BE={be_is[-1]:.2e}")
print(f"  S&P cold         FE={fe_cold[-1]:.2e}  BE={be_cold[-1]:.2e}")
print(f"  S&P warm         FE={fe_warm[-1]:.2e}  BE={be_warm[-1]:.2e}")
print(f"  FOSSILS          FE={fossils_fe[-1]:.2e}  BE={fossils_be[-1]:.2e}")
print(f"  QR reference     FE={fe_qr:.2e}  BE={be_qr:.2e}")
