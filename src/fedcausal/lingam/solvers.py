"""Regression solvers that turn a causal order into a weighted adjacency matrix.

Given a causal order (source to sink) and a covariance matrix, each variable's
coefficients are obtained by regressing it on the variables that precede it in
the order, restricted to the sufficient statistics already available: no raw
data is needed, only the covariance. Two estimators are provided: plain OLS
and an adaptive-Lasso variant that also shrinks small coefficients to zero.

References
----------
[1] Zou, H. "The Adaptive Lasso and Its Oracle Properties." Journal of the
    American Statistical Association, 101(476), 2006.
[2] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["estimate_b_from_cov", "estimate_b_lasso_from_cov"]


def estimate_b_from_cov(
    causal_order: Sequence[int],
    cov: npt.NDArray[np.float64],
    prune_thres: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Re-estimate adjacency matrix B by OLS from the original covariance.

    Parameters
    ----------
    causal_order : Sequence[int]
        Variable indices from source to sink.
    cov : numpy.ndarray
        Global covariance matrix, shape ``(p, p)``.
    prune_thres : float
        Smallest absolute value to keep in B.

    Returns
    -------
    numpy.ndarray
        Estimated DAG adjacency matrix, shape ``(p, p)``.

    References
    ----------
    [1] Zou, H. "The Adaptive Lasso and Its Oracle Properties." Journal of
        the American Statistical Association, 101(476), 2006.
    """
    p = cov.shape[0]
    b_est = np.zeros((p, p))
    for idx in range(1, len(causal_order)):
        child = causal_order[idx]
        parents = causal_order[:idx]
        sigma_pp = cov[np.ix_(parents, parents)]
        sigma_pc = cov[parents, child]
        try:
            beta = np.linalg.solve(sigma_pp, sigma_pc)
        except np.linalg.LinAlgError:
            # Handle near-singular covariance
            beta = np.linalg.lstsq(sigma_pp, sigma_pc, rcond=None)[0]
        b_est[child, parents] = beta

    if prune_thres > 0:
        b_est[np.abs(b_est) < prune_thres] = 0.0
    return b_est


def estimate_b_lasso_from_cov(
    causal_order: Sequence[int],
    cov: npt.NDArray[np.float64],
    n_samples: int = 1000,
    lmbda_type: str = "fixed",
    prune_thres: float = 0.05,
    gamma: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Re-estimate B by Adaptive Lasso from the global covariance.

    Parameters
    ----------
    causal_order : Sequence[int]
        Topological order.
    cov : numpy.ndarray
        Global covariance matrix, shape ``(p, p)``.
    n_samples : int
        Total number of samples (used for BIC).
    lmbda_type : {"fixed", "bic"}
        ``"fixed"`` for the 0.01 heuristic, ``"bic"`` for analytical selection.
    prune_thres : float
        Final hard-threshold for zeros.
    gamma : float
        Power parameter for adaptive weights.

    Returns
    -------
    numpy.ndarray
        Estimated DAG adjacency matrix, shape ``(p, p)``.

    Raises
    ------
    ValueError
        If `lmbda_type` is not ``"fixed"`` or ``"bic"``.

    References
    ----------
    [1] Zou, H. "The Adaptive Lasso and Its Oracle Properties." Journal of
        the American Statistical Association, 101(476), 2006.
    """
    p = cov.shape[0]
    b_est = np.zeros((p, p))
    for idx in range(1, len(causal_order)):
        child = causal_order[idx]
        parents = causal_order[:idx]
        sigma_pp = cov[np.ix_(parents, parents)]
        sigma_pc = cov[parents, child]
        sigma_cc = cov[child, child]

        # 1. Compute OLS for adaptive weights
        try:
            beta_ols = np.linalg.solve(sigma_pp, sigma_pc)
        except np.linalg.LinAlgError:
            beta_ols = np.linalg.lstsq(sigma_pp, sigma_pc, rcond=None)[0]

        weights = 1.0 / (np.abs(beta_ols) ** gamma + 1e-10)

        if lmbda_type == "fixed":
            # 2a. Fixed heuristic: proportional to average parent variance
            lmbda = float(0.01 * np.mean(np.diag(sigma_pp)))
            beta = _solve_lasso_cd(sigma_pp, sigma_pc, lmbda, weights)
        elif lmbda_type == "bic":
            # 2b. BIC selection: search over a grid
            lmbda_max = np.max(np.abs(sigma_pc)) if len(sigma_pc) > 0 else 0
            if lmbda_max < 1e-12:
                beta = np.zeros(len(parents))
            else:
                # 20 points from 0.1% of max to max
                lambdas = np.logspace(np.log10(lmbda_max * 1e-3), np.log10(lmbda_max), 20)
                best_bic = np.inf
                beta = np.zeros(len(parents))
                log_n = np.log(n_samples)

                for lam in lambdas:
                    curr_beta = _solve_lasso_cd(sigma_pp, sigma_pc, float(lam), weights)
                    # RSS/n = Var(y) - 2*beta*Sxy + beta*Sxx*beta
                    rss_n = (
                        sigma_cc
                        - 2 * curr_beta.dot(sigma_pc)
                        + curr_beta.dot(sigma_pp).dot(curr_beta)
                    )
                    rss_n = max(rss_n, 1e-12)
                    k = np.count_nonzero(curr_beta)
                    bic = n_samples * np.log(rss_n) + k * log_n
                    if bic < best_bic:
                        best_bic = bic
                        beta = curr_beta
        else:
            raise ValueError(f"Unknown lmbda_type: {lmbda_type!r}; expected 'fixed' or 'bic'.")

        b_est[child, parents] = beta

    b_est[np.abs(b_est) < prune_thres] = 0.0
    return b_est


def _solve_lasso_cd(
    cxx: npt.NDArray[np.float64],
    cxy: npt.NDArray[np.float64],
    lmbda: float,
    weights: npt.NDArray[np.float64],
    max_iter: int = 100,
    tol: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Coordinate descent Lasso optimizer using only covariance statistics.

    Solves ``min_beta 1/2 beta' Cxx beta - Cxy' beta + lambda |w * beta|``.

    Parameters
    ----------
    cxx : numpy.ndarray
        Parent-parent covariance block, shape ``(q, q)``.
    cxy : numpy.ndarray
        Parent-child covariance vector, shape ``(q,)``.
    lmbda : float
        Regularisation strength.
    weights : numpy.ndarray
        Adaptive per-coefficient weights, shape ``(q,)``.
    max_iter : int
        Maximum number of coordinate-descent sweeps.
    tol : float
        Convergence tolerance on the coefficient update norm.

    Returns
    -------
    numpy.ndarray
        Coefficient vector, shape ``(q,)``.
    """
    q = len(cxy)
    beta = np.zeros(q)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(q):
            # Partial residual: r_j = y'x_j - sum_{k!=j} x_k'x_j * beta_k
            r_j = cxy[j] - (cxx[j, :].dot(beta) - cxx[j, j] * beta[j])

            # Soft thresholding
            threshold = lmbda * weights[j]
            if r_j > threshold:
                beta[j] = (r_j - threshold) / cxx[j, j]
            elif r_j < -threshold:
                beta[j] = (r_j + threshold) / cxx[j, j]
            else:
                beta[j] = 0.0

        if np.linalg.norm(beta - beta_old) < tol:
            break
    return beta
