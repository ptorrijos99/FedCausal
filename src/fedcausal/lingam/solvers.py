"""Coefficient estimates for a known causal order, from the covariance alone.

Each variable is regressed on the ones before it in the order, using only the
aggregated covariance, so no data is needed. OLS, or an adaptive Lasso that
also shrinks small coefficients to zero.

References
----------
[1] Zou, H. "The Adaptive Lasso and Its Oracle Properties." Journal of the
    American Statistical Association, 101(476), 2006.
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
    """Estimate the coefficient matrix B by OLS.

    Parameters
    ----------
    causal_order : Sequence[int]
        Variable indices from source to sink.
    cov : numpy.ndarray
        Covariance matrix, shape ``(p, p)``.
    prune_thres : float
        Coefficients smaller than this in absolute value become zero.

    Returns
    -------
    numpy.ndarray
        ``B[child, parent]``, shape ``(p, p)``.
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
            # Singular covariance: fall back to least squares.
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
    """Estimate the coefficient matrix B by adaptive Lasso.

    Parameters
    ----------
    causal_order : Sequence[int]
        Variable indices from source to sink.
    cov : numpy.ndarray
        Covariance matrix, shape ``(p, p)``.
    n_samples : int
        Total number of samples. Only ``"bic"`` uses it.
    lmbda_type : {"fixed", "bic"}
        ``"fixed"`` sets the penalty to 0.01 times the mean variance of the
        parents; ``"bic"`` picks it from a grid of 20 by BIC.
    prune_thres : float
        Coefficients smaller than this in absolute value become zero.
    gamma : float
        Exponent of the adaptive weights.

    Returns
    -------
    numpy.ndarray
        ``B[child, parent]``, shape ``(p, p)``.

    Raises
    ------
    ValueError
        If `lmbda_type` is not ``"fixed"`` or ``"bic"``.
    """
    p = cov.shape[0]
    b_est = np.zeros((p, p))
    for idx in range(1, len(causal_order)):
        child = causal_order[idx]
        parents = causal_order[:idx]
        sigma_pp = cov[np.ix_(parents, parents)]
        sigma_pc = cov[parents, child]
        sigma_cc = cov[child, child]

        # OLS gives the adaptive weights.
        try:
            beta_ols = np.linalg.solve(sigma_pp, sigma_pc)
        except np.linalg.LinAlgError:
            beta_ols = np.linalg.lstsq(sigma_pp, sigma_pc, rcond=None)[0]

        weights = 1.0 / (np.abs(beta_ols) ** gamma + 1e-10)

        if lmbda_type == "fixed":
            lmbda = float(0.01 * np.mean(np.diag(sigma_pp)))
            beta = _solve_lasso_cd(sigma_pp, sigma_pc, lmbda, weights)
        elif lmbda_type == "bic":
            lmbda_max = np.max(np.abs(sigma_pc)) if len(sigma_pc) > 0 else 0
            if lmbda_max < 1e-12:
                beta = np.zeros(len(parents))
            else:
                lambdas = np.logspace(np.log10(lmbda_max * 1e-3), np.log10(lmbda_max), 20)
                best_bic = np.inf
                beta = np.zeros(len(parents))
                log_n = np.log(n_samples)

                for lam in lambdas:
                    curr_beta = _solve_lasso_cd(sigma_pp, sigma_pc, float(lam), weights)
                    # RSS / n from the covariance alone.
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
    """Solve the weighted Lasso by coordinate descent, from covariances alone.

    Minimises ``min_beta 1/2 beta' Cxx beta - Cxy' beta + lambda |w * beta|``.

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
            r_j = cxy[j] - (cxx[j, :].dot(beta) - cxx[j, j] * beta[j])

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
