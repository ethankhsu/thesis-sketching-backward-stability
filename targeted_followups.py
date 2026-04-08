"""
Targeted synthetic follow-up experiments.

This script runs only the two approved follow-up slices:
  1. Warm-start anomaly at kappa=1e3, b_only, res_size=1e-2, all 4 aspect ratios.
  2. FOSSILS anomaly at kappa=1e3, a_and_b, aspects 50/100/250, all 3 residual sizes.

Standalone sketch-and-solve is included as a direct baseline.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from collections import OrderedDict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import fossils_lib as fl
from experiment_utils import append_rows, evaluate_methods


RESULTS_CSV = "results/followup_results.csv"
SUMMARY_CSV = "results/followup_summary.csv"

N = 50
D_SKETCH = 12 * N
NOISE_LEVEL = 0.01
N_ITER = 25
N_PHASE = 12

SEEDS_20 = [
    32439,
    10001,
    20002,
    30003,
    40004,
    50005,
    60006,
    70007,
    80008,
    90009,
    110011,
    120012,
    130013,
    140014,
    150015,
    160016,
    170017,
    180018,
    190019,
    200020,
]

FIELDNAMES = [
    "study",
    "aspect",
    "kappa",
    "res_size",
    "noise_model",
    "seed",
    "effective_kappa",
    "method",
    "final_be",
    "pass_fail",
    "iters_to_stable",
]


def build_jobs():
    jobs = []

    for aspect in [10, 50, 100, 250]:
        for seed in SEEDS_20:
            jobs.append(
                OrderedDict(
                    study="warm_start_followup",
                    aspect=aspect,
                    kappa=1e3,
                    res_size=1e-2,
                    noise_model="b_only",
                    seed=seed,
                )
            )

    for aspect in [50, 100, 250]:
        for res_size in [1e-2, 1e-6, 1e-10]:
            for seed in SEEDS_20:
                jobs.append(
                    OrderedDict(
                        study="fossils_followup",
                        aspect=aspect,
                        kappa=1e3,
                        res_size=res_size,
                        noise_model="a_and_b",
                        seed=seed,
                    )
                )

    return jobs


def run_job(job):
    rng = np.random.default_rng(job["seed"])
    m = int(job["aspect"] * N)

    A, b, _, effective_kappa = fl.generate_ls_problem(
        m,
        N,
        job["kappa"],
        job["res_size"],
        noise_model=job["noise_model"],
        noise_level=NOISE_LEVEL,
        rng=rng,
    )

    extra_fields = dict(job)
    extra_fields["effective_kappa"] = float(effective_kappa)

    return evaluate_methods(
        A,
        b,
        seed=job["seed"],
        d_sketch=D_SKETCH,
        n_iter=N_ITER,
        n_phase=N_PHASE,
        include_sketch_solve=True,
        include_iter_sketch=True,
        extra_fields=extra_fields,
    )


if __name__ == "__main__":
    jobs = build_jobs()
    print(f"Running {len(jobs)} targeted follow-up configurations")

    all_rows = Parallel(n_jobs=2, verbose=5)(delayed(run_job)(job) for job in jobs)
    rows = [row for batch in all_rows for row in batch]

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
    append_rows(RESULTS_CSV, FIELDNAMES, rows)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["study", "noise_model", "aspect", "res_size", "method"], dropna=False)
        .agg(
            pass_rate=("pass_fail", "mean"),
            median_be=("final_be", "median"),
            n_runs=("pass_fail", "size"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved detailed rows to {RESULTS_CSV}")
    print(f"Saved summary rows to {SUMMARY_CSV}")

    print("\nWarm-start follow-up summary:")
    warm = summary[summary["study"] == "warm_start_followup"]
    print(
        warm.pivot_table(
            values="pass_rate",
            index=["aspect", "res_size"],
            columns="method",
        ).to_string()
    )

    print("\nFOSSILS follow-up summary:")
    fossils = summary[summary["study"] == "fossils_followup"]
    print(
        fossils.pivot_table(
            values="pass_rate",
            index=["aspect", "res_size"],
            columns="method",
        ).to_string()
    )
