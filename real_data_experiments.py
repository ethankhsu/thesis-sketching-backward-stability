"""
Real-data validation for the thesis follow-up.

Main path:
  - SuiteSparse well1033 / illc1033 as the sparse, heterogeneous-column test bed
  - LIBSVM cpusmall as the dense pilot
  - YearPredictionMSD is optional and only runs when explicitly requested

The script expects datasets to already exist on disk.
"""

import argparse
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiment_utils import (
    append_rows,
    apply_relative_gaussian_perturbation,
    construct_controlled_rhs,
    evaluate_methods,
    extract_tar_gz,
    find_matrix_market_file,
    load_matrix_market_matrix,
    load_svmlight_dense,
    matrix_metadata,
)


DATASETS = {
    "well1033": {
        "family": "suitesparse",
        "archive": Path("data/suitesparse/well1033.tar.gz"),
        "extract_dir": Path("data/suitesparse/well1033"),
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/HB/well1033.tar.gz",
    },
    "illc1033": {
        "family": "suitesparse",
        "archive": Path("data/suitesparse/illc1033.tar.gz"),
        "extract_dir": Path("data/suitesparse/illc1033"),
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/HB/illc1033.tar.gz",
    },
    "well1850": {
        "family": "suitesparse",
        "archive": Path("data/suitesparse/well1850.tar.gz"),
        "extract_dir": Path("data/suitesparse/well1850"),
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/HB/well1850.tar.gz",
    },
    "illc1850": {
        "family": "suitesparse",
        "archive": Path("data/suitesparse/illc1850.tar.gz"),
        "extract_dir": Path("data/suitesparse/illc1850"),
        "url": "https://suitesparse-collection-website.herokuapp.com/MM/HB/illc1850.tar.gz",
    },
    "cpusmall": {
        "family": "libsvm",
        "path": Path("data/libsvm/cpusmall"),
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall",
    },
    "YearPredictionMSD": {
        "family": "libsvm",
        "path": Path("data/libsvm/YearPredictionMSD"),
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD",
    },
}

FIELDNAMES = [
    "dataset_id",
    "dataset_family",
    "noise_model",
    "seed",
    "m",
    "n",
    "m_over_n",
    "density",
    "condition_proxy",
    "column_norm_ratio",
    "numerical_rank",
    "full_column_rank",
    "effective_kappa",
    "method",
    "final_be",
    "pass_fail",
    "iters_to_stable",
]

SUMMARY_CSV = "results/real_data_summary.csv"
DETAIL_CSV = "results/real_data_results.csv"
META_CSV = "results/real_data_matrix_metadata.csv"
FIG_PATH = "figures/real_data_pass_rates.pdf"

SEEDS = [32439, 10001, 20002, 30003, 40004]
NOISE_LEVEL = 0.01
RELATIVE_RESIDUAL = 1e-6
COLUMN_RATIO_GATE = 100.0

METHODS_FOR_FIG = [
    "sketch_solve",
    "sketch_pre_warm",
    "fossils",
    "spir",
    "qr_ref",
]

METHOD_LABELS = {
    "sketch_solve": "Sketch-and-solve",
    "sketch_pre_warm": "Sketch&Pre (warm)",
    "fossils": "FOSSILS",
    "spir": "SPIR",
    "qr_ref": "QR reference",
}


def load_dataset(dataset_id):
    spec = DATASETS[dataset_id]
    family = spec["family"]

    if family == "suitesparse":
        archive = spec["archive"]
        if not archive.exists():
            raise FileNotFoundError(
                f"Missing {archive}. Download from {spec['url']}"
            )
        extract_dir = extract_tar_gz(archive, spec["extract_dir"])
        matrix_path = find_matrix_market_file(extract_dir, stem_hint=dataset_id)
        A = load_matrix_market_matrix(matrix_path)
        b, _ = construct_controlled_rhs(A, relative_residual=RELATIVE_RESIDUAL, rng=np.random.default_rng(7))
    elif family == "libsvm":
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download from {spec['url']}"
            )
        A, b = load_svmlight_dense(path)
    else:
        raise ValueError(f"Unknown dataset family: {family}")

    if A.shape[0] <= A.shape[1]:
        raise ValueError(
            f"{dataset_id} is not overdetermined (shape {A.shape[0]} x {A.shape[1]})"
        )

    return np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64).reshape(-1), family


def run_one(dataset_id, noise_model, seed, A_base, b_base, base_meta):
    if noise_model == "b_only":
        A_run = A_base.copy()
        b_run = b_base.copy()
        effective_kappa = float(base_meta["condition_proxy"])
    elif noise_model == "a_and_b":
        A_run, b_run = apply_relative_gaussian_perturbation(
            A_base, b_base, noise_level=NOISE_LEVEL, rng=np.random.default_rng(seed)
        )
        effective_kappa = float(np.linalg.cond(A_run))
    else:
        raise ValueError(f"Unsupported noise model: {noise_model}")

    extra_fields = {
        "dataset_id": dataset_id,
        "dataset_family": DATASETS[dataset_id]["family"],
        "noise_model": noise_model,
        "seed": seed,
        "m": base_meta["m"],
        "n": base_meta["n"],
        "m_over_n": base_meta["m_over_n"],
        "density": base_meta["density"],
        "condition_proxy": base_meta["condition_proxy"],
        "column_norm_ratio": base_meta["column_norm_ratio"],
        "numerical_rank": base_meta["numerical_rank"],
        "full_column_rank": base_meta["full_column_rank"],
        "effective_kappa": effective_kappa,
    }

    return evaluate_methods(
        A_run,
        b_run,
        seed=seed,
        include_sketch_solve=True,
        include_iter_sketch=True,
        extra_fields=extra_fields,
    )


def save_metadata(metadata_rows):
    if os.path.exists(META_CSV):
        os.remove(META_CSV)
    append_rows(
        META_CSV,
        [
            "dataset_id",
            "dataset_family",
            "m",
            "n",
            "m_over_n",
            "density",
            "condition_proxy",
            "column_norm_ratio",
            "numerical_rank",
            "full_column_rank",
            "heterogeneity_gate_pass",
        ],
        metadata_rows,
    )


def make_figure(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    noise_order = [("b_only", r"$b$-only"), ("a_and_b", r"$A$-and-$b$")]
    dataset_order = list(summary["dataset_id"].drop_duplicates())
    x = np.arange(len(dataset_order))
    width = 0.14

    for ax, (noise_model, title) in zip(axes, noise_order):
        noise_df = summary[
            (summary["noise_model"] == noise_model)
            & (summary["method"].isin(METHODS_FOR_FIG))
        ].copy()

        for idx, method in enumerate(METHODS_FOR_FIG):
            method_df = noise_df[noise_df["method"] == method]
            rates = []
            for dataset_id in dataset_order:
                sub = method_df[method_df["dataset_id"] == dataset_id]
                rates.append(sub["pass_rate"].iloc[0] if not sub.empty else 0.0)

            ax.bar(
                x + (idx - 2) * width,
                rates,
                width=width,
                label=METHOD_LABELS[method],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_order)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_xlabel("Dataset")

    axes[0].set_ylabel("Backward stability pass rate")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Real-data validation: pass rates by dataset and noise model")
    plt.tight_layout()
    plt.savefig(FIG_PATH, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["well1033", "illc1033", "cpusmall"],
        help="Datasets to run. YearPredictionMSD is opt-in.",
    )
    args = parser.parse_args()

    metadata_rows = []
    jobs = []
    loaded = {}

    for dataset_id in args.datasets:
        A_base, b_base, family = load_dataset(dataset_id)
        meta = matrix_metadata(A_base)
        meta["dataset_id"] = dataset_id
        meta["dataset_family"] = family
        meta["heterogeneity_gate_pass"] = int(meta["column_norm_ratio"] >= COLUMN_RATIO_GATE)
        metadata_rows.append(meta)
        loaded[dataset_id] = (A_base, b_base, meta)

        if dataset_id in {"well1033", "illc1033"}:
            status = "PASS" if meta["heterogeneity_gate_pass"] else "WARN"
            print(
                f"{dataset_id}: column_norm_ratio={meta['column_norm_ratio']:.2f} "
                f"(gate {COLUMN_RATIO_GATE:.0f}) -> {status}"
            )

        for noise_model in ["b_only", "a_and_b"]:
            for seed in SEEDS:
                jobs.append((dataset_id, noise_model, seed))

    save_metadata(metadata_rows)
    print(f"Saved matrix metadata to {META_CSV}")

    all_rows = Parallel(n_jobs=1, verbose=5)(
        delayed(run_one)(
            dataset_id,
            noise_model,
            seed,
            loaded[dataset_id][0],
            loaded[dataset_id][1],
            loaded[dataset_id][2],
        )
        for dataset_id, noise_model, seed in jobs
    )
    rows = [row for batch in all_rows for row in batch]

    if os.path.exists(DETAIL_CSV):
        os.remove(DETAIL_CSV)
    append_rows(DETAIL_CSV, FIELDNAMES, rows)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(
            [
                "dataset_id",
                "dataset_family",
                "noise_model",
                "method",
                "m",
                "n",
                "m_over_n",
                "density",
                "condition_proxy",
                "column_norm_ratio",
            ],
            dropna=False,
        )
        .agg(
            pass_rate=("pass_fail", "mean"),
            median_be=("final_be", "median"),
            n_runs=("pass_fail", "size"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    make_figure(summary)

    print(f"Saved detailed rows to {DETAIL_CSV}")
    print(f"Saved summary rows to {SUMMARY_CSV}")
    print(f"Saved figure to {FIG_PATH}")

    report = summary[summary["method"].isin(METHODS_FOR_FIG)].copy()
    print(
        report.pivot_table(
            values="pass_rate",
            index=["dataset_id", "noise_model"],
            columns="method",
        ).to_string()
    )
