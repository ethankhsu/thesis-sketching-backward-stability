# When Does Sketching Suffice? Backward Stability in Randomized Least Squares

Code and experimental results for Ethan Hsu's senior thesis at Princeton ORFE
(advisor: Prof. Elizaveta Rebrova, submission April 2026). The thesis studies
the conditions under which sketch-and-solve alone achieves backward stability
for overdetermined least squares, versus when iterative refinement
(FOSSILS, SPIR) is required.

## Overview

All solvers and the backward-error machinery live in `fossils_lib.py`:

- `generate_ls_problem` builds a least-squares instance `(A, b)` with
  controlled condition number and residual norm, optionally perturbing both
  `A` and `b` for the `a_and_b` noise model.
- `sparse_sign_embedding` constructs the sparse sign sketch
  (`zeta = 8` nonzeros per column, normalized by `1/sqrt(zeta)`).
- `backward_error_ls` computes the exact Walden-Karlson-Sun backward error.
- `backward_error_kw` computes the Karlson-Walden estimate, which is within
  a factor of `sqrt(2)` of the true backward error.
- `sketch_and_solve`, `sketch_and_precondition`, `fossils`, and `spir`
  implement the four randomized solvers compared in the thesis.
  FOSSILS and SPIR apply column scaling before sketching, matching
  Algorithm 4 / Section 4.2 of Epperly, Meier, and Nakatsukasa (2025).

`sketch_and_precondition` uses a `LinearOperator` for implicit `A R^{-1}`
matvecs; the preconditioned matrix is never materialized.

## Reproducing the experiments

```bash
# Full parameter sweep (840 configs x 5-10 seeds, parallelized with joblib)
python sweep.py

# Re-run only FOSSILS (skips other solvers for speed)
python sweep.py --fossils-only

# Dry run (20 configs only, for smoke testing)
python sweep.py --dry-run

# Reproduce Figure 1 from Epperly et al.
python figure1_reproduction.py

# Plot per-iteration backward error trajectories from saved .npz files
python plot_be_trajectories.py
```

Dependencies: `numpy`, `scipy`, `joblib`, `filelock`, `matplotlib`.
BLAS threads are pinned to 1 at the top of `sweep.py` to avoid
oversubscription under joblib parallelism.

### Parameter grid

| Parameter | Values |
|---|---|
| aspect ratio `m/n` | 10, 50, 100, 250 |
| condition number `kappa` | 1e3, 1e6, 1e9, 1e12, 1e14 |
| residual norm | 1e-2, 1e-6, 1e-10 |
| noise model | `b_only`, `a_and_b` |
| seeds | 5 (kappa in {1e3, 1e6, 1e9}), 10 (kappa in {1e12, 1e14}) |

Problem size: `n = 50`, sketch dimension `d = 12 n = 600`.
Backward-stability threshold: `sqrt(2) * eps_machine`.

## Repository layout

```
fossils_lib.py              core library (solvers, sketching, backward error)
sweep.py                    main parameter sweep with joblib + file-locked CSV
experiment_utils.py         shared helpers for sweep and follow-up scripts
figure1_reproduction.py     reproduces Figure 1 from Epperly et al.
plot_be_trajectories.py     per-iteration backward-error plots
sensitivity_sweep.py        sensitivity analysis over sketch parameters
real_data_experiments.py    experiments on real matrices
diagnostic_binary_split.py  diagnostic split-by-noise-model experiment
targeted_followups.py       targeted follow-up configurations
replot_*.py                 regenerate individual figures from saved CSVs
results/                    CSV outputs and per-iteration .npz histories
figures/                    generated plots (PDF and PNG)
```

Checkpointing: `sweep.py` reads completed configurations from
`results/sweep_results.csv` and skips them on re-runs. CSV appends are
file-locked for parallel safety.

## References

- Epperly, Meier, Nakatsukasa (2025). Fast randomized least-squares solvers
  can be just as accurate and stable as classical direct solvers.
- Higham (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM.
- van der Sluis (1969). Condition numbers and equilibration of matrices.
