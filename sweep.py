"""
Parameter sweep: characterising when sketch-and-solve suffices vs
when FOSSILS/SPIR refinement is required.

Run locally:
    python sweep.py              # full sweep
    python sweep.py --dry-run    # 20 configs only, for testing

Output:
    results/sweep_results.csv        # one row per (config, method)
    results/be_histories/<key>.npz   # per-iteration BE traces
"""

import os
# Pin BLAS threads BEFORE importing numpy/scipy to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import csv
import itertools
import numpy as np
from scipy.linalg import lstsq
from filelock import FileLock
from joblib import Parallel, delayed

from fossils_lib import (
    generate_ls_problem, sparse_sign_embedding,
    iterative_sketching_momentum, sketch_and_precondition,
    fossils, spir, backward_error_kw,
)
from scipy.linalg import svd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR    = "results"
CSV_FILE       = os.path.join(RESULTS_DIR, "sweep_results.csv")
LOCK_FILE      = os.path.join(RESULTS_DIR, "sweep_results.lock")
HIST_DIR       = os.path.join(RESULTS_DIR, "be_histories")

N              = 50
D_SKETCH       = 12 * N
N_ITER         = 25
N_PHASE1       = 12
N_PHASE2       = 13
EPS            = np.finfo(np.float64).eps   # per-iteration KW tracking threshold
PASS_TOL       = np.sqrt(2) * EPS           # pass/fail: kw_be <= PASS_TOL means true BE <= 2*eps
NOISE_LEVEL    = 0.01  # ||delta_A||_F / ||A||_F for a_and_b model

ASPECT_RATIOS  = [10, 50, 100, 250]
KAPPAS         = [1e3, 1e6, 1e9, 1e12, 1e14]
RES_SIZES      = [1e-2, 1e-6, 1e-10]
NOISE_MODELS   = ["b_only", "a_and_b"]

SEEDS_BASE     = [32439, 10001, 20002, 30003, 40004]
SEEDS_HIGH     = SEEDS_BASE + [50005, 60006, 70007, 80008, 90009]
HIGH_KAPPA     = {1e12, 1e14}

CSV_FIELDNAMES = [
    "aspect", "kappa", "res_size", "noise_model", "seed",
    "effective_kappa", "method",
    "final_be", "pass_fail", "iters_to_stable",
]


# ---------------------------------------------------------------------------
# Build config list
# ---------------------------------------------------------------------------

def build_configs():
    configs = []
    for aspect, kappa, res_size, noise_model in itertools.product(
        ASPECT_RATIOS, KAPPAS, RES_SIZES, NOISE_MODELS
    ):
        seeds = SEEDS_HIGH if kappa in HIGH_KAPPA else SEEDS_BASE
        for seed in seeds:
            configs.append((aspect, kappa, res_size, noise_model, seed))
    return configs


# ---------------------------------------------------------------------------
# Restart: skip already-completed configs
# ---------------------------------------------------------------------------

def get_completed(csv_file):
    """A config is complete only if ALL expected methods have rows."""
    ALL_METHODS = {"iter_sketch_mom", "sketch_pre_cold", "sketch_pre_warm",
                   "fossils", "spir", "qr_ref"}
    from collections import defaultdict
    methods_done = defaultdict(set)
    if not os.path.exists(csv_file):
        return set()
    with open(csv_file, "r") as f:
        for row in csv.DictReader(f):
            key = (
                float(row["aspect"]), float(row["kappa"]),
                float(row["res_size"]), row["noise_model"], int(row["seed"])
            )
            methods_done[key].add(row["method"])
    return {k for k, v in methods_done.items() if v >= ALL_METHODS}


# ---------------------------------------------------------------------------
# Stability helpers
# ---------------------------------------------------------------------------

def iters_to_stable(be_history, threshold, window=5):
    """
    First iteration where BE drops below threshold and stays there
    for at least `window` consecutive remaining iterations.
    Returns None if never reached.
    """
    n = len(be_history)
    for i in range(n):
        tail = be_history[i: i + window]
        if len(tail) == window and all(be <= threshold for be in tail):
            return i
    return None


# ---------------------------------------------------------------------------
# Run a single configuration
# ---------------------------------------------------------------------------

def run_one_config(aspect, kappa, res_size, noise_model, seed, fossils_only=False):
    m = aspect * N
    rng = np.random.default_rng(seed)

    # Generate problem
    A, b, x_true, effective_kappa = generate_ls_problem(
        m, N, kappa, res_size, rng=rng,
        noise_model=noise_model, noise_level=NOISE_LEVEL
    )
    A_norm_fro = np.linalg.norm(A, 'fro')

    # Precompute full SVD of A once (used for KW tracking across all methods)
    _, sigma_full, Vt_full = svd(A, full_matrices=False)
    V_full = Vt_full.T

    def kw_be(x):
        return backward_error_kw(A, b, x, theta=np.inf,
                                 sigma_s=sigma_full, V_s=V_full) / A_norm_fro

    results = []
    histories = {}

    if not fossils_only:
        # QR reference
        x_qr, _, _, _ = lstsq(A, b)

        # --- Iterative sketching with momentum ---
        _, hist = iterative_sketching_momentum(
            A, b, d=D_SKETCH, n_iter=N_ITER,
            rng=np.random.default_rng(seed + 1), track_history=True
        )
        be_hist = [h[1] for h in hist]
        final_be = be_hist[-1]
        results.append(dict(
            aspect=aspect, kappa=kappa, res_size=res_size,
            noise_model=noise_model, seed=seed,
            effective_kappa=effective_kappa, method="iter_sketch_mom",
            final_be=final_be,
            pass_fail=int(final_be <= PASS_TOL),
            iters_to_stable=iters_to_stable(be_hist, EPS),
        ))
        histories["iter_sketch_mom"] = be_hist

        # --- Sketch-and-precondition cold start ---
        _, hist = sketch_and_precondition(
            A, b, d=D_SKETCH, n_iter=N_ITER, cold_start=True,
            rng=np.random.default_rng(seed + 2), track_history=True
        )
        be_hist = [h[1] for h in hist]
        final_be = be_hist[-1]
        results.append(dict(
            aspect=aspect, kappa=kappa, res_size=res_size,
            noise_model=noise_model, seed=seed,
            effective_kappa=effective_kappa, method="sketch_pre_cold",
            final_be=final_be,
            pass_fail=int(final_be <= PASS_TOL),
            iters_to_stable=iters_to_stable(be_hist, EPS),
        ))
        histories["sketch_pre_cold"] = be_hist

        # --- Sketch-and-precondition warm start ---
        _, hist = sketch_and_precondition(
            A, b, d=D_SKETCH, n_iter=N_ITER, cold_start=False,
            rng=np.random.default_rng(seed + 3), track_history=True
        )
        be_hist = [h[1] for h in hist]
        final_be = be_hist[-1]
        results.append(dict(
            aspect=aspect, kappa=kappa, res_size=res_size,
            noise_model=noise_model, seed=seed,
            effective_kappa=effective_kappa, method="sketch_pre_warm",
            final_be=final_be,
            pass_fail=int(final_be <= PASS_TOL),
            iters_to_stable=iters_to_stable(be_hist, EPS),
        ))
        histories["sketch_pre_warm"] = be_hist

    # --- FOSSILS (Algorithm 4, Epperly et al.) ---
    # Column scaling
    scale_f = np.linalg.norm(A, axis=0)
    scale_f[scale_f == 0] = 1.0
    A_f = A / scale_f

    S = sparse_sign_embedding(m, D_SKETCH, rng=np.random.default_rng(seed + 4))
    U_s, sigma_s, Vt_s = svd(S @ A_f, full_matrices=False)
    V_s = Vt_s.T
    inv_sigma = 1.0 / sigma_s
    eta = np.sqrt(N / D_SKETCH)
    alpha_pb = (1 - eta**2)**2
    beta_pb  = eta**2

    def apply_P(y):  return V_s @ (inv_sigma * y)
    def apply_Pt(z): return inv_sigma * (V_s.T @ z)

    def polyak_phase(b_rhs, x_base, n_steps, iter_offset):
        c = apply_Pt(A_f.T @ b_rhs)
        dy, dy_old = c.copy(), c.copy()
        hist = []
        for i in range(n_steps):
            Mdy = apply_Pt(A_f.T @ (A_f @ apply_P(dy)))
            delta = alpha_pb * (c - Mdy) + beta_pb * (dy - dy_old)
            dy_old = dy.copy()
            dy += delta
            x_cur = x_base + apply_P(dy)
            hist.append((iter_offset + i + 1, kw_be(x_cur / scale_f)))
        return apply_P(dy), hist

    x0_f = apply_P(U_s.T @ (S @ b))
    be0 = kw_be(x0_f / scale_f)
    dx1, h1 = polyak_phase(b - A_f @ x0_f, x0_f, N_PHASE1, 0)
    x1_f = x0_f + dx1
    dx2, h2 = polyak_phase(b - A_f @ x1_f, x1_f, N_PHASE2, N_PHASE1)
    be_hist = [be0] + [be for _, be in h1] + [be for _, be in h2]
    final_be = be_hist[-1]
    results.append(dict(
        aspect=aspect, kappa=kappa, res_size=res_size,
        noise_model=noise_model, seed=seed,
        effective_kappa=effective_kappa, method="fossils",
        final_be=final_be,
        pass_fail=int(final_be <= PASS_TOL),
        iters_to_stable=iters_to_stable(be_hist, EPS),
    ))
    histories["fossils"] = be_hist

    if not fossils_only:
        # --- SPIR ---
        _, hist = spir(
            A, b, d=D_SKETCH, n_iter=N_PHASE1,
            rng=np.random.default_rng(seed + 5), track_history=True
        )
        be_hist = [h[1] for h in hist]
        final_be = be_hist[-1]
        results.append(dict(
            aspect=aspect, kappa=kappa, res_size=res_size,
            noise_model=noise_model, seed=seed,
            effective_kappa=effective_kappa, method="spir",
            final_be=final_be,
            pass_fail=int(final_be <= PASS_TOL),
            iters_to_stable=iters_to_stable(be_hist, EPS),
        ))
        histories["spir"] = be_hist

        # --- QR reference ---
        # QR is backward stable by construction; pass_fail = 1 always.
        # final_be uses KW (same formula as all other methods).
        results.append(dict(
            aspect=aspect, kappa=kappa, res_size=res_size,
            noise_model=noise_model, seed=seed,
            effective_kappa=effective_kappa, method="qr_ref",
            final_be=kw_be(x_qr),
            pass_fail=1,
            iters_to_stable=0,
        ))
        histories["qr_ref"] = [kw_be(x_qr)]

    return results, histories


# ---------------------------------------------------------------------------
# Run, save, and checkpoint
# ---------------------------------------------------------------------------

def run_and_save(cfg, fossils_only=False):
    aspect, kappa, res_size, noise_model, seed = cfg
    try:
        results, histories = run_one_config(*cfg, fossils_only=fossils_only)
    except Exception as e:
        print(f"ERROR on config {cfg}: {e}")
        return

    # Save CSV (with file lock for parallel safety)
    # Only write rows for methods not already present (avoids duplicates on partial re-runs)
    lock = FileLock(LOCK_FILE)
    with lock:
        existing_methods = set()
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, "r") as f:
                for row in csv.DictReader(f):
                    if (float(row["aspect"]) == aspect and float(row["kappa"]) == kappa
                            and float(row["res_size"]) == res_size
                            and row["noise_model"] == noise_model
                            and int(row["seed"]) == seed):
                        existing_methods.add(row["method"])
        file_exists = os.path.exists(CSV_FILE)
        new_results = [r for r in results if r["method"] not in existing_methods]
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            for r in new_results:
                writer.writerow(r)

    # Save per-iteration histories as .npz
    key = f"aspect{aspect}_kappa{kappa:.0e}_res{res_size:.0e}_{noise_model}_seed{seed}"
    npz_path = os.path.join(HIST_DIR, f"{key}.npz")
    np.savez(npz_path, **{k: np.array(v) for k, v in histories.items()})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    fossils_only = "--fossils-only" in sys.argv

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HIST_DIR, exist_ok=True)

    all_configs = build_configs()

    if dry_run:
        # Sample: one slice through kappa at aspect=10, b_only, first seed
        configs = [(10, k, 1e-6, "b_only", 32439) for k in KAPPAS] + \
                  [(10, k, 1e-6, "a_and_b", 32439) for k in KAPPAS] + \
                  [(50, 1e12, 1e-6, "b_only", 32439),
                   (100, 1e12, 1e-6, "b_only", 32439)]
        print(f"Dry run: {len(configs)} configs")
    else:
        completed = get_completed(CSV_FILE)
        configs = [c for c in all_configs
                   if (float(c[0]), float(c[1]), float(c[2]), c[3], int(c[4]))
                   not in completed]
        print(f"Total: {len(all_configs)}  Completed: {len(all_configs)-len(configs)}  "
              f"Remaining: {len(configs)}")

    Parallel(n_jobs=4, verbose=10)(
        delayed(run_and_save)(cfg, fossils_only=fossils_only) for cfg in configs
    )

    print(f"\nDone. Results in {CSV_FILE}")
