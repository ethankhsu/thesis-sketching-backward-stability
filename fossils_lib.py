"""
Core library for FOSSILS thesis experiments.
Implements: sparse sign embeddings, sketch-and-solve, iterative sketching with momentum,
sketch-and-precondition (LSQR), FOSSILS, SPIR, backward error (Walden-Karlson-Sun-Higham).
"""

import numpy as np
from scipy.linalg import lstsq, solve_triangular, svd, qr
from scipy.sparse.linalg import lsqr as scipy_lsqr, LinearOperator


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def generate_ls_problem(m, n, kappa, residual_norm, rng=None,
                         noise_model="b_only", noise_level=0.0):
    """
    Generate a least-squares problem A in R^{m x n}, b in R^m with:
      - cond(A) = kappa
      - ||b - A x_true|| = residual_norm  (residual orthogonal to col(A))

    noise_model: "b_only" (default) or "a_and_b"
      - "b_only": A is clean, b already contains residual noise
      - "a_and_b": after constructing A, add a random perturbation
            delta_A with ||delta_A||_F / ||A||_F = noise_level
        and delta_b with ||delta_b|| / ||b|| = noise_level

    Returns A, b, x_true, effective_kappa.
    effective_kappa equals kappa for b_only, and cond(A + delta_A) for a_and_b.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random orthogonal matrices via QR
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # Log-spaced singular values from kappa down to 1
    sigma = np.logspace(np.log10(kappa), 0, n)
    A = U[:, :n] @ np.diag(sigma) @ V.T

    # True solution and residual orthogonal to col(A)
    x_true = rng.standard_normal(n)
    Ax = A @ x_true
    r_true = U[:, n:] @ rng.standard_normal(m - n)
    r_true = r_true * (residual_norm / np.linalg.norm(r_true))
    b = Ax + r_true

    effective_kappa = float(kappa)

    if noise_model == "a_and_b" and noise_level > 0.0:
        # Perturb A: ||delta_A||_F / ||A||_F = noise_level
        delta_A = rng.standard_normal((m, n))
        delta_A = delta_A * (noise_level * np.linalg.norm(A, 'fro')
                             / np.linalg.norm(delta_A, 'fro'))
        A = A + delta_A

        # Perturb b: ||delta_b|| / ||b|| = noise_level
        delta_b = rng.standard_normal(m)
        delta_b = delta_b * (noise_level * np.linalg.norm(b)
                             / np.linalg.norm(delta_b))
        b = b + delta_b

        # Log effective condition number (cheap at small n)
        effective_kappa = float(np.linalg.cond(A))

    return A, b, x_true, effective_kappa


# ---------------------------------------------------------------------------
# Sparse sign embedding
# ---------------------------------------------------------------------------

def sparse_sign_embedding(m, d, zeta=8, rng=None):
    """
    Build a d x m sparse sign embedding with zeta nonzeros per column.
    Returns a dense matrix S of shape (d, m).
    """
    if rng is None:
        rng = np.random.default_rng()

    S = np.zeros((d, m))
    for j in range(m):
        rows = rng.choice(d, size=zeta, replace=False)
        signs = rng.choice([-1, 1], size=zeta)
        S[rows, j] = signs
    S /= np.sqrt(zeta)
    return S


# ---------------------------------------------------------------------------
# Backward error (exact, matches backward_error_ls.m from Epperly et al. repo)
# ---------------------------------------------------------------------------

def backward_error_ls(A, b, xhat, theta=np.inf):
    """
    Exact backward error BE_theta(xhat) for least-squares.
    Matches backward_error_ls.m from the Epperly-Meier-Nakatsukasa repo exactly.

    For theta=inf:
      phi = ||r|| / ||xhat||
      be = min(phi, sigma_min([A, phi*(I - r*r'/||r||^2)]))
    """
    r = b - A @ xhat
    norm_x = np.linalg.norm(xhat)
    norm_r = np.linalg.norm(r)

    if norm_x < 1e-300:
        return 0.0

    if np.isinf(theta):
        mu = 1.0
    else:
        mu_val = theta**2 * norm_x**2
        mu = mu_val / (1.0 + mu_val)

    phi = np.sqrt(mu) * norm_r / norm_x

    # Build [A | phi*(I - r_hat r_hat^T)] and find its min singular value
    m = A.shape[0]
    r_hat = r / norm_r
    B = phi * (np.eye(m) - np.outer(r_hat, r_hat))
    M = np.hstack([A, B])
    sigma_min = np.linalg.svd(M, compute_uv=False)[-1]

    return min(phi, sigma_min)



def backward_error_kw(A, b, xhat, theta=np.inf, U_s=None, sigma_s=None, V_s=None):
    """
    Karlson-Walden estimate of backward error (within factor sqrt(2) of true).
    Uses sketched SVD if provided (U_s, sigma_s, V_s from sketch), else full SVD.
    O(mn) with precomputed SVD, O(mn^2) otherwise.
    """
    r = b - A @ xhat
    norm_x = np.linalg.norm(xhat)
    norm_r = np.linalg.norm(r)

    if U_s is None:
        _, sigma_s, Vt_s = svd(A, full_matrices=False)
        V_s = Vt_s.T

    if np.isinf(theta):
        if norm_x < 1e-300:
            return 0.0
        c = norm_r**2 / norm_x**2
        prefactor = 1.0 / norm_x
    else:
        denom = 1.0 + theta**2 * norm_x**2
        c = theta**2 * norm_r**2 / denom
        prefactor = theta / np.sqrt(denom)

    ATr = A.T @ r
    VTATr = V_s.T @ ATr
    inv_sqrt_diag = 1.0 / np.sqrt(sigma_s**2 + c)
    vec = inv_sqrt_diag * VTATr

    return prefactor * np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# Sketch-and-solve (one-shot)
# ---------------------------------------------------------------------------

def sketch_and_solve(A, b, d=None, rng=None):
    """
    Compute x = argmin ||SA x - Sb|| using thin SVD of SA.
    Returns x, and the sketched SVD components (U_s, sigma_s, V_s, S).
    """
    m, n = A.shape
    if d is None:
        d = 12 * n
    if rng is None:
        rng = np.random.default_rng()

    S = sparse_sign_embedding(m, d, rng=rng)
    SA = S @ A
    Sb = S @ b

    U_s, sigma_s, Vt_s = svd(SA, full_matrices=False)
    V_s = Vt_s.T

    x = V_s @ (np.diag(1.0 / sigma_s) @ (U_s.T @ Sb))
    return x, U_s, sigma_s, V_s, S


# ---------------------------------------------------------------------------
# Polyak heavy ball (FOSSILS inner solver)
# ---------------------------------------------------------------------------

def polyak_heavy_ball(A, b_rhs, V_s, sigma_s, alpha, beta, n_iter=100):
    """
    Solve (R^{-T} A^T A R^{-1}) y = c using Polyak heavy ball,
    where R^{-1} = V_s * diag(1/sigma_s) (preconditioner from sketch SVD).

    Returns delta_x = R^{-1} y (the solution correction).
    """
    # Preconditioner: P = V_s * diag(1/sigma_s), so P y = V_s * (y/sigma_s)
    inv_sigma = 1.0 / sigma_s

    def apply_P(y):
        return V_s @ (inv_sigma * y)

    def apply_Pt(z):
        return inv_sigma * (V_s.T @ z)

    def matvec(y):
        # (R^{-T} A^T A R^{-1}) y
        z = apply_P(y)
        Az = A @ z
        AtAz = A.T @ Az
        return apply_Pt(AtAz)

    c = apply_Pt(A.T @ b_rhs)

    # Polyak iteration
    dy = c.copy()
    dy_old = c.copy()

    for _ in range(n_iter):
        Mdy = matvec(dy)
        delta = alpha * (c - Mdy) + beta * (dy - dy_old)
        dy_old = dy.copy()
        dy = dy + delta

    return apply_P(dy)


# ---------------------------------------------------------------------------
# FOSSILS outer solver (single refinement step)
# ---------------------------------------------------------------------------

def fossils_outer_solver(A, b_rhs, V_s, sigma_s, eta, n_iter=100):
    """
    Single FOSSILS outer solve: finds correction delta to minimize
    ||b_rhs - A delta|| using preconditioned Polyak heavy ball.
    eta = distortion parameter, typically sqrt(n/d).
    """
    alpha = (1.0 - eta**2)**2
    beta = eta**2
    return polyak_heavy_ball(A, b_rhs, V_s, sigma_s, alpha, beta, n_iter)


# ---------------------------------------------------------------------------
# FOSSILS (full, 2-step refinement)
# ---------------------------------------------------------------------------

def fossils(A, b, d=None, n_iter=100, rng=None, track_history=False, x_true=None):
    """
    Full FOSSILS algorithm with 2-step iterative refinement (Algorithm 4, Epperly et al.).
    Applies column scaling before sketching.
    Returns x, and optionally history of (forward_error, backward_error) per step.
    """
    m, n = A.shape
    if d is None:
        d = 12 * n
    if rng is None:
        rng = np.random.default_rng()

    # Column scaling (Algorithm 4, step 1, Epperly et al.)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    A_sc = A / scale

    eta = np.sqrt(n / d)  # distortion estimate

    # Step 1: Sketch and compute SVD
    S = sparse_sign_embedding(m, d, rng=rng)
    SA = S @ A_sc
    Sb = S @ b

    U_s, sigma_s, Vt_s = svd(SA, full_matrices=False)
    V_s = Vt_s.T

    # Sketch-and-solve initialization
    x0 = V_s @ (np.diag(1.0 / sigma_s) @ (U_s.T @ Sb))

    history = []

    def record(x_sc, label):
        if track_history and x_true is not None:
            x = x_sc / scale  # undo scaling for error computation
            fe = np.linalg.norm(x - x_true) / max(np.linalg.norm(x_true), 1e-300)
            be = backward_error_kw(A, b, x, theta=np.inf, sigma_s=sigma_s, V_s=V_s)
            be_norm = be / np.linalg.norm(A, 'fro')
            history.append((label, fe, be_norm))

    record(x0, 'init')

    # First refinement
    r0 = b - A_sc @ x0
    dx1 = fossils_outer_solver(A_sc, r0, V_s, sigma_s, eta, n_iter)
    x1 = x0 + dx1
    record(x1, 'refine1')

    # Second refinement
    r1 = b - A_sc @ x1
    dx2 = fossils_outer_solver(A_sc, r1, V_s, sigma_s, eta, n_iter)
    x2 = x1 + dx2
    record(x2, 'refine2')

    # Undo column scaling (Algorithm 4, step 37)
    x2 = x2 / scale
    return x2, history, (U_s, sigma_s, V_s)


# ---------------------------------------------------------------------------
# Iterative sketching with momentum
# ---------------------------------------------------------------------------

def iterative_sketching_momentum(A, b, d=None, n_iter=25, rng=None,
                                  track_history=False, x_true=None):
    """
    Iterative sketching with optimal Polyak momentum.
    x_{i+1} = x_i + alpha R^{-1} R^{-T} A^T(b - A x_i) + beta(x_i - x_{i-1})
    alpha, beta chosen optimally given eta = sqrt(n/d).
    """
    m, n = A.shape
    if d is None:
        d = 4 * n
    if rng is None:
        rng = np.random.default_rng()

    S = sparse_sign_embedding(m, d, rng=rng)
    SA = S @ A

    # QR factorization for preconditioner
    _, R = qr(SA, mode='economic')

    eta = np.sqrt(n / d)
    alpha = (1.0 - eta**2)**2
    beta = eta**2

    history = []

    A_norm_fro = np.linalg.norm(A, 'fro')
    # KW estimate for per-iteration tracking (exact is O(m^2 n), too slow at m=4000)
    _, sigma_full, Vt_full = svd(A, full_matrices=False)
    V_full = Vt_full.T

    def record(x, i):
        if track_history:
            be = backward_error_kw(A, b, x, theta=np.inf,
                                   sigma_s=sigma_full, V_s=V_full) / A_norm_fro
            entry = [i, be]
            if x_true is not None:
                fe = np.linalg.norm(x - x_true) / max(np.linalg.norm(x_true), 1e-300)
                entry.append(fe)
            history.append(tuple(entry))

    x = np.zeros(n)
    x_old = np.zeros(n)

    record(x, 0)

    for i in range(1, n_iter + 1):
        r = b - A @ x
        # Preconditioned gradient: R^{-1} R^{-T} A^T r
        ATr = A.T @ r
        g = solve_triangular(R, solve_triangular(R, ATr, trans='T'), trans='N')
        x_new = x + alpha * g + beta * (x - x_old)
        x_old = x.copy()
        x = x_new
        record(x, i)

    return x, history


# ---------------------------------------------------------------------------
# Sketch-and-precondition (LSQR with warm start)
# ---------------------------------------------------------------------------

def sketch_and_precondition(A, b, d=None, n_iter=25, cold_start=False,
                             rng=None, track_history=False, x_true=None):
    """
    Sketch-and-precondition using LSQR on the preconditioned system A R^{-1}.
    Optionally uses sketch-and-solve warm start.
    """
    m, n = A.shape
    if d is None:
        d = 2 * n
    if rng is None:
        rng = np.random.default_rng()

    S = sparse_sign_embedding(m, d, rng=rng)
    SA = S @ A
    Sb = S @ b

    U_s, sigma_s, Vt_s = svd(SA, full_matrices=False)
    V_s = Vt_s.T
    R = np.diag(sigma_s) @ Vt_s  # R from sketch QR: SA = U_s * R

    # Preconditioned system: (A R^{-1}) y = b, x = R^{-1} y
    # We use the normal equations approach: solve AR^{-1} y = b via LSQR
    # Warm start: y0 = R x0 where x0 is sketch-and-solve
    if cold_start:
        x0 = np.zeros(n)
    else:
        x0 = V_s @ (np.diag(1.0 / sigma_s) @ (U_s.T @ Sb))

    history = []
    A_norm_fro = np.linalg.norm(A, 'fro')
    # KW estimate for per-iteration tracking (exact is O(m^2 n), too slow at m=4000)
    _, sigma_full, Vt_full = svd(A, full_matrices=False)
    V_full = Vt_full.T

    def record(x, i):
        if track_history:
            be = backward_error_kw(A, b, x, theta=np.inf,
                                   sigma_s=sigma_full, V_s=V_full) / A_norm_fro
            entry = [i, be]
            if x_true is not None:
                fe = np.linalg.norm(x - x_true) / max(np.linalg.norm(x_true), 1e-300)
                entry.append(fe)
            history.append(tuple(entry))

    record(x0, 0)

    # Implicit matvec operators for the preconditioned system AR^{-1}
    # R^{-1} = V_s diag(1/sigma_s), so:
    #   (AR^{-1}) y  = A @ (V_s @ (y / sigma_s))
    #   (AR^{-1})^T z = (V_s.T @ (A.T @ z)) / sigma_s
    inv_sigma = 1.0 / sigma_s

    def matvec(y):
        return A @ (V_s @ (inv_sigma * y))

    def rmatvec(z):
        return inv_sigma * (V_s.T @ (A.T @ z))

    AR_inv_op = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec)

    # Warm start in preconditioned space: y0 = R x0 = diag(sigma_s) V_s^T x0
    y0 = sigma_s * (V_s.T @ x0)

    # Single LSQR run with callback for per-iteration tracking
    x_i = x0.copy()
    if track_history:
        # Run one iteration at a time to collect history
        y_cur = y0.copy()
        for i in range(1, n_iter + 1):
            result = scipy_lsqr(AR_inv_op, b, x0=y_cur, iter_lim=1, atol=0, btol=0)
            y_cur = result[0]
            x_i = V_s @ (inv_sigma * y_cur)
            record(x_i, i)
    else:
        result = scipy_lsqr(AR_inv_op, b, x0=y0, iter_lim=n_iter, atol=0, btol=0)
        y_i = result[0]
        x_i = V_s @ (inv_sigma * y_i)

    return x_i, history


# ---------------------------------------------------------------------------
# SPIR (Sketch-and-Precondition with Iterative Refinement)
# Uses CG on the preconditioned normal equations, 2 refinement steps.
# Matches metasolver.m + spir.m from Epperly-Meier-Nakatsukasa repo.
# ---------------------------------------------------------------------------

def spir(A, b, d=None, n_iter=25, rng=None, track_history=False, x_true=None):
    """
    SPIR: two-step iterative refinement using CG on (R^{-T} A^T A R^{-1}) y = c.
    R^{-1} = V_s * diag(1/sigma_s) from sketch SVD.
    """
    m, n = A.shape
    if d is None:
        d = 12 * n
    if rng is None:
        rng = np.random.default_rng()

    # Column scaling (matches metasolver.m)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    A_sc = A / scale

    # Sketch and SVD
    S = sparse_sign_embedding(m, d, rng=rng)
    SA = S @ A_sc
    U_s, sigma_s, Vt_s = svd(SA, full_matrices=False)
    V_s = Vt_s.T

    # Sketch-and-solve initialization
    x = V_s @ ((U_s.T @ (S @ b)) / sigma_s)
    r = b - A_sc @ x

    A_norm_fro = np.linalg.norm(A, 'fro')
    # KW estimate for per-iteration tracking (exact is O(m^2 n), too slow at m=4000)
    _, sigma_full, Vt_full = svd(A, full_matrices=False)
    V_full = Vt_full.T

    history = []

    def record(xhat_sc, i):
        if track_history:
            xhat = xhat_sc / scale
            be = backward_error_kw(A, b, xhat, theta=np.inf,
                                   sigma_s=sigma_full, V_s=V_full) / A_norm_fro
            entry = [i, be]
            if x_true is not None:
                fe = np.linalg.norm(xhat - x_true) / max(np.linalg.norm(x_true), 1e-300)
                entry.append(fe)
            history.append(tuple(entry))

    record(x, 0)

    def apply_RAAR(dy):
        """(R^{-T} A^T A R^{-1}) dy"""
        z = V_s @ (dy / sigma_s)
        return (V_s.T @ (A_sc.T @ (A_sc @ z))) / sigma_s

    # Two refinement loops (matching metasolver.m loop structure)
    for loop in range(2):
        c = (V_s.T @ (A_sc.T @ r)) / sigma_s

        # CG on (R^{-T} A^T A R^{-1}) dy = c
        dy = c.copy()
        cg_r = c - apply_RAAR(dy)
        cg_p = cg_r.copy()
        rsq = cg_r @ cg_r

        for i in range(1, n_iter + 1):
            Ap = apply_RAAR(cg_p)
            alpha = rsq / (cg_p @ Ap)
            dy = dy + alpha * cg_p
            cg_r = cg_r - alpha * Ap
            new_rsq = cg_r @ cg_r
            beta = new_rsq / rsq
            rsq = new_rsq
            cg_p = cg_r + beta * cg_p

            x_cur = x + V_s @ (dy / sigma_s)
            record(x_cur, loop * n_iter + i)

        x = x + V_s @ (dy / sigma_s)
        r = b - A_sc @ x

    return x / scale, history
