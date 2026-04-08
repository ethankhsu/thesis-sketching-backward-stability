"""
Shared utilities for thesis follow-up experiments.

This module adds:
  - common method evaluation, including standalone sketch-and-solve
  - real-data loading helpers for SuiteSparse / Matrix Market and LIBSVM files
  - matrix metadata summaries used in the real-data section
"""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.linalg import lstsq, svd

import fossils_lib as fl


EPS = np.finfo(np.float64).eps
PASS_TOL = np.sqrt(2.0) * EPS


def iters_to_stable(be_history, threshold, window=5):
    """
    Return the first iteration whose next `window` entries all stay below threshold.
    """
    n_hist = len(be_history)
    for i in range(n_hist):
        tail = be_history[i : i + window]
        if len(tail) == window and all(be <= threshold for be in tail):
            return i
    return None


def build_kw_context(A, b):
    """
    Precompute the full SVD data needed for consistent KW backward-error evaluation.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    _, sigma_full, Vt_full = svd(A, full_matrices=False)
    V_full = Vt_full.T
    A_norm_fro = np.linalg.norm(A, "fro")

    def kw_be(x):
        return fl.backward_error_kw(
            A,
            b,
            np.asarray(x, dtype=np.float64).reshape(-1),
            theta=np.inf,
            sigma_s=sigma_full,
            V_s=V_full,
        ) / A_norm_fro

    return {
        "sigma_full": sigma_full,
        "V_full": V_full,
        "A_norm_fro": A_norm_fro,
        "kw_be": kw_be,
    }


def matrix_metadata(A, sigma_full=None):
    """
    Compute matrix-level metadata for the real-data experiments.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    nnz = np.count_nonzero(A)
    col_norms = np.linalg.norm(A, axis=0)
    positive_norms = col_norms[col_norms > 0]
    if positive_norms.size == 0:
        col_norm_ratio = np.inf
    else:
        col_norm_ratio = float(np.max(positive_norms) / np.min(positive_norms))

    if sigma_full is None:
        sigma_full = svd(A, full_matrices=False, compute_uv=False)

    min_sigma = float(sigma_full[-1]) if sigma_full.size else 0.0
    if min_sigma <= 0.0:
        cond = np.inf
    else:
        cond = float(sigma_full[0] / min_sigma)

    rank = int(np.linalg.matrix_rank(A))

    return {
        "m": int(m),
        "n": int(n),
        "m_over_n": float(m / n),
        "density": float(nnz / A.size),
        "condition_proxy": cond,
        "column_norm_ratio": col_norm_ratio,
        "numerical_rank": rank,
        "full_column_rank": int(rank == n),
    }


def apply_relative_gaussian_perturbation(A, b, noise_level, rng=None):
    """
    Match the synthetic a_and_b perturbation model:
      ||dA||_F / ||A||_F = noise_level
      ||db||_2 / ||b||_2 = noise_level
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    dA = rng.standard_normal(A.shape)
    dA *= noise_level * np.linalg.norm(A, "fro") / max(np.linalg.norm(dA, "fro"), 1e-300)

    db = rng.standard_normal(b.shape[0])
    db *= noise_level * np.linalg.norm(b) / max(np.linalg.norm(db), 1e-300)

    return A + dA, b + db


def construct_controlled_rhs(A, relative_residual=1e-6, rng=None):
    """
    For a real full-column-rank matrix A, create b = A x_true + r with
    r orthogonal to col(A) and ||r||_2 = relative_residual * ||A x_true||_2.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    if m <= n:
        raise ValueError("construct_controlled_rhs requires an overdetermined matrix (m > n)")

    x_true = rng.standard_normal(n)
    Ax = A @ x_true

    Q, _ = np.linalg.qr(A, mode="complete")
    null_basis = Q[:, n:]
    if null_basis.shape[1] == 0:
        raise ValueError("No orthogonal complement available to construct a residual")

    r = null_basis @ rng.standard_normal(null_basis.shape[1])
    target_norm = relative_residual * max(np.linalg.norm(Ax), 1e-300)
    r *= target_norm / max(np.linalg.norm(r), 1e-300)
    b = Ax + r
    return b, x_true


def load_matrix_market_matrix(path):
    """
    Load a Matrix Market matrix as a dense float64 ndarray.
    """
    matrix = mmread(path)
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def find_matrix_market_file(root, stem_hint=None):
    """
    Find the primary matrix file inside an extracted SuiteSparse directory.
    """
    root = Path(root)
    mtx_files = sorted(root.rglob("*.mtx"))
    if not mtx_files:
        raise FileNotFoundError(f"No .mtx files found under {root}")

    if stem_hint is not None:
        preferred = [p for p in mtx_files if p.stem.lower() == stem_hint.lower()]
        if preferred:
            return preferred[0]

    # Fall back to the first file that does not look like an RHS.
    for path in mtx_files:
        lower = path.stem.lower()
        if "rhs" not in lower and "b" != lower:
            return path

    return mtx_files[0]


def extract_tar_gz(archive_path, dest_dir):
    """
    Extract a tar.gz archive if needed and return the extraction root.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    marker = dest_dir / ".extracted"
    if not marker.exists():
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        marker.write_text("ok", encoding="ascii")
    return dest_dir


def load_svmlight_dense(path, max_rows=None):
    """
    Load a LIBSVM / SVMLight file as a dense ndarray.

    Prefer scikit-learn when available. The manual fallback is suitable for
    smaller dense pilots such as cpusmall.
    """
    path = Path(path)

    try:
        from sklearn.datasets import load_svmlight_file  # type: ignore

        X, y = load_svmlight_file(path, n_features=None)
        if max_rows is not None:
            X = X[:max_rows]
            y = y[:max_rows]
        return np.asarray(X.toarray(), dtype=np.float64), np.asarray(y, dtype=np.float64)
    except Exception:
        pass

    row_idx = []
    col_idx = []
    data = []
    targets = []
    max_col = -1

    with path.open("r", encoding="ascii") as handle:
        for row, line in enumerate(handle):
            if max_rows is not None and row >= max_rows:
                break
            parts = line.strip().split()
            if not parts:
                continue
            targets.append(float(parts[0]))
            for token in parts[1:]:
                idx_str, value_str = token.split(":", 1)
                col = int(idx_str) - 1
                val = float(value_str)
                row_idx.append(len(targets) - 1)
                col_idx.append(col)
                data.append(val)
                if col > max_col:
                    max_col = col

    if max_col < 0:
        raise ValueError(f"No features parsed from {path}")

    X = sparse.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(targets), max_col + 1),
        dtype=np.float64,
    )
    return np.asarray(X.toarray(), dtype=np.float64), np.asarray(targets, dtype=np.float64)


def append_rows(csv_path, fieldnames, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_methods(
    A,
    b,
    seed,
    d_sketch=None,
    n_iter=25,
    n_phase=12,
    include_sketch_solve=True,
    include_iter_sketch=True,
    extra_fields=None,
):
    """
    Evaluate all requested methods on a fixed matrix/vector pair.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape

    if d_sketch is None:
        d_sketch = 12 * n

    kw_ctx = build_kw_context(A, b)
    kw_be = kw_ctx["kw_be"]

    extra_fields = dict(extra_fields or {})
    results = []

    def record(
        method,
        x=None,
        be=None,
        be_history=None,
        pass_fail_override=None,
        iters_to_stable_override=None,
    ):
        final_be = float(kw_be(x) if be is None else be)
        row = dict(extra_fields)
        row.update(
            {
                "method": method,
                "final_be": final_be,
                "pass_fail": (
                    int(final_be <= PASS_TOL)
                    if pass_fail_override is None
                    else int(pass_fail_override)
                ),
                "iters_to_stable": (
                    (
                        iters_to_stable(be_history, EPS)
                        if be_history is not None
                        else None
                    )
                    if iters_to_stable_override is None
                    else iters_to_stable_override
                ),
            }
        )
        results.append(row)

    if include_sketch_solve:
        try:
            x_s, _, _, _, _ = fl.sketch_and_solve(
                A, b, d=d_sketch, rng=np.random.default_rng(seed + 1)
            )
            record("sketch_solve", x=x_s, be_history=[kw_be(x_s)])
        except Exception:
            record("sketch_solve", be=np.inf, be_history=[np.inf])

    if include_iter_sketch:
        try:
            x_i, hist_i = fl.iterative_sketching_momentum(
                A,
                b,
                d=d_sketch,
                n_iter=n_iter,
                rng=np.random.default_rng(seed + 2),
                track_history=True,
            )
            be_hist = [entry[1] for entry in hist_i]
            record("iter_sketch_mom", x=x_i, be=be_hist[-1], be_history=be_hist)
        except Exception:
            record("iter_sketch_mom", be=np.inf, be_history=[np.inf])

        for cold_start, method_name, offset in [
            (True, "sketch_pre_cold", 3),
            (False, "sketch_pre_warm", 4),
        ]:
            try:
                x_sp, hist_sp = fl.sketch_and_precondition(
                    A,
                    b,
                    d=d_sketch,
                    n_iter=n_iter,
                    cold_start=cold_start,
                    rng=np.random.default_rng(seed + offset),
                    track_history=True,
                )
                be_hist = [entry[1] for entry in hist_sp]
                record(method_name, x=x_sp, be=be_hist[-1], be_history=be_hist)
            except Exception:
                record(method_name, be=np.inf, be_history=[np.inf])

    try:
        x_f, _, _ = fl.fossils(
            A, b, d=d_sketch, n_iter=n_phase, rng=np.random.default_rng(seed + 5)
        )
        record("fossils", x=x_f, be_history=[kw_be(x_f)])
    except Exception:
        record("fossils", be=np.inf, be_history=[np.inf])

    try:
        x_spir, hist_spir = fl.spir(
            A,
            b,
            d=d_sketch,
            n_iter=n_phase,
            rng=np.random.default_rng(seed + 6),
            track_history=True,
        )
        be_hist = [entry[1] for entry in hist_spir]
        record("spir", x=x_spir, be=be_hist[-1], be_history=be_hist)
    except Exception:
        record("spir", be=np.inf, be_history=[np.inf])

    x_qr, _, _, _ = lstsq(A, b)
    record("qr_ref", x=x_qr, be_history=[kw_be(x_qr)], pass_fail_override=1, iters_to_stable_override=0)

    return results
