# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Senior thesis at Princeton ORFE titled **"When Does Sketching Suffice? Backward Stability in Randomized Least Squares"** by Ethan Hsu (advisor: Prof. Elizaveta Rebrova, submission May 2026). The research question: under what conditions does sketch-and-solve alone achieve backward stability, versus requiring FOSSILS/SPIR iterative refinement?

## Running Experiments

```bash
# Full parameter sweep (parallelized, 840 configs x 5-10 seeds)
python sweep.py

# Re-run only FOSSILS method (skips other solvers for speed)
python sweep.py --fossils-only

# Dry run (20 configs only, for testing)
python sweep.py --dry-run

# Reproduce Figure 1 from Epperly et al.
python figure1_reproduction.py

# Plot per-iteration backward error trajectories from saved .npz files
python plot_be_trajectories.py
```

Use `python3` and confirm `numpy`, `scipy`, `joblib`, `filelock`, `matplotlib` are available. Install with `pip install --break-system-packages` if needed.

## Architecture

All algorithm implementations live in `fossils_lib.py`. The sweep script (`sweep.py`) imports from it and runs the full parameter space in parallel via `joblib`. Plotting scripts load pre-saved results from `results/`.

**`fossils_lib.py`** — core library:
- `generate_ls_problem`: constructs A (controlled kappa via log-spaced SVD), b (with orthogonal residual), and optionally perturbs both A and b for the `a_and_b` noise model
- `sparse_sign_embedding`: builds sparse sign sketching matrix (zeta=8 nonzeros/column, normalized by 1/sqrt(zeta))
- `backward_error_ls`: exact Walden-Karlson-Sun backward error (matches `backward_error_ls.m` from Epperly et al. repo)
- `backward_error_kw`: Karlson-Walden estimate (within sqrt(2) of true BE); uses precomputed SVD when provided
- `sketch_and_solve`: one-shot sketch solution via thin SVD of SA
- `sketch_and_precondition`: LSQR on preconditioned system `A R^{-1}` via `LinearOperator` (implicit matvecs, NOT explicit dense matrix); supports cold/warm start
- `fossils`: 2-step Polyak heavy ball refinement with column scaling (normalizes columns of A before sketching, matching Algorithm 4 of Epperly et al.)
- `spir`: 2-step CG refinement with column scaling (normalizes columns of A before sketching)

**`sweep.py`** — parameter sweep:
- Grid: aspect ratios {10,50,100,250}, kappa {1e3,...,1e14}, residual sizes {1e-2,1e-6,1e-10}, noise models {b_only, a_and_b}
- 5 seeds for kappa in {1e3,1e6,1e9}; 10 seeds for kappa in {1e12,1e14}
- Checkpointing: reads completed configs from CSV and skips them; file-locked CSV appends for parallel safety
- Output: `results/sweep_results.csv` (one row per config+method) and `results/be_histories/<key>.npz` (per-iteration BE arrays)
- n=50, d=600 (12n), PASS_TOL = sqrt(2)*machine_epsilon, iters_to_stable uses 5-consecutive-window below machine epsilon

## Key Constants and Parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| N | 50 | problem dimension n |
| D_SKETCH | 600 | sketch dimension d = 12n |
| PASS_TOL | sqrt(2)*eps | backward stability threshold (KW estimate is within sqrt(2) of true BE) |
| NOISE_LEVEL | 0.01 | relative perturbation for a_and_b model |
| N_ITER | 25 | total iterations for iterative methods |
| N_PHASE1/2 | 12/13 | FOSSILS/SPIR budget split |

## Critical Implementation Notes

- `sketch_and_precondition` uses `LinearOperator` for implicit `A R^{-1}` matvecs. **Never** materialize `A @ R^{-1}` as a dense matrix (inflates pass rates via numerical cancellation).
- QR `pass_fail` is hardcoded to 1 (KW formula degenerates for near-optimal solutions; QR is backward stable by construction).
- Both FOSSILS and SPIR apply column scaling before sketching (`scale = ||A[:,j]||`), matching Algorithm 4 / Section 4.2 of Epperly et al. The sweep has been re-run with column scaling for all methods.
- `iterative_sketching_momentum` and `sketch_and_precondition` precompute the full SVD of A once per config for KW tracking; the sketched SVD is only used internally.
- BLAS threads are pinned to 1 (`OMP_NUM_THREADS=1`) at the top of `sweep.py` before numpy/scipy import to avoid oversubscription in joblib parallel jobs.

## LaTeX / Thesis Writing

The thesis is written in LaTeX on Overleaf. If working on `.tex` files: unzip the Overleaf zip into a subfolder (e.g., `latex/`) and identify chapter files.

**Writing conventions (strictly enforced):**
- No em dashes. Use commas, semicolons, or restructure.
- Formal academic register throughout.
- For edits: provide exact old text → new text with enough surrounding context to locate unambiguously. Never use line numbers. Never delete comments or unrequested lines.
- Placement instructions must reference exact surrounding text (e.g., "insert after the sentence ending '...'").
- LaTeX output must be paste-ready into Overleaf with no modifications.

## Key Citation Claims

| Claim | Citation |
|-------|----------|
| Backward error formula is kappa-free (structural) | Epperly et al. 2025, Theorem 2.8 vs. Prop 2.9; Higham 2002, Thm 20.5 |
| KW estimate within sqrt(2) of true BE | Epperly et al. 2025, Fact 4.1 |
| SPIR/FOSSILS backward stable when kappa*u << 1 | Epperly et al. 2025, Theorem 6.1 |
| Forward stability → backward stability when kappa*||r|| small | Epperly et al. 2025, Corollary 2.10 |
| Column scaling improves conditioning | van der Sluis 1969; Higham 2002, Thm 7.5 |

## What NOT to Claim

- Do NOT claim backward error being kappa-independent is a "discovery." It is structural (the KW formula has no kappa parameter).
- Do NOT claim kappa has "no effect on backward stability" in general. The correct framing: in the tested parameter range, noise model is the dominant explanatory variable; kappa and aspect ratio show no visible effect on pass rates.
- DO distinguish: the formula being kappa-free (definitional) vs. pass rates being kappa-flat (empirical).
