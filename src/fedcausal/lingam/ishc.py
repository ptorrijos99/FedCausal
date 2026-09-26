"""FedISHC and FedHC (Chen et al.), third-order baselines for FedRCD.

``fedishc`` removes one source at a time, deflating the cumulants by the
clipped ratio ``c21[j, s] / c3[s]``, and recovers B from the accumulated mixing
matrix, ``B = I - A^-1``. ``fedhc`` sorts once by the third-order tau and
estimates B from the covariance. Both match the reference implementation
numerically; ``tests/lingam/test_reference_implementation.py`` checks it.

References
----------
[1] Chen, W., et al. "FedISHC: Federated Causal Discovery via Higher-Order
    Cumulants." AAAI / arXiv, 2025.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fedcausal.lingam.cumulants import Cumulants, _n_samples
from fedcausal.lingam.rcd import RCDResult
from fedcausal.lingam.solvers import estimate_b_from_cov, estimate_b_lasso_from_cov
from fedcausal.lingam.tau import tau_matrix_3rd

__all__ = ["fedhc", "fedishc"]

# The reference implementation's safeguards for the ratio deflation.
_EPS = 1e-6
_ALPHA_CLIP = 10.0


def fedishc(stats: Cumulants) -> RCDResult:
    """Find the causal order by FedISHC's ratio deflation, and B from the mixing matrix.

    Parameters
    ----------
    stats : Cumulants
        Aggregated statistics containing ``c3``, ``c12``, ``c21``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.
    """
    c3 = np.array(stats["c3"], dtype=float).copy()
    c12 = np.array(stats["c12"], dtype=float).copy()
    c21 = np.array(stats["c21"], dtype=float).copy()

    p = len(c3)
    active = list(range(p))
    causal_order: list[int] = []

    # Mixing matrix A, with X = A E.
    a_mat = np.eye(p)

    for _ in range(p - 1):
        tau = tau_matrix_3rd(c3[active], c12[np.ix_(active, active)], c21[np.ix_(active, active)])
        t_tau_scores = np.sum(tau, axis=1)

        min_idx_local = int(np.argmin(t_tau_scores))
        s = active[min_idx_local]
        causal_order.append(s)
        active.pop(min_idx_local)

        # Deflate by c3[s] as it was when s was picked, not as c3 changes below.
        c3_s = c3[s]

        for xj in active:
            if np.abs(c3_s) < _EPS:
                alpha = 0.0
            else:
                alpha = float(np.clip(c21[xj, s] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP))

            a_mat[xj, s] = alpha

            c3[xj] = c3[xj] - (alpha**3) * c3_s

            for xk in active:
                if np.abs(c3_s) > _EPS:
                    lam = float(np.clip(c21[xk, s] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP))
                else:
                    lam = 0.0

                c12[xj, xk] -= alpha * (lam**2) * c3_s
                c21[xj, xk] -= (alpha**2) * lam * c3_s

    if active:
        causal_order.append(active[0])

    try:
        a_inv = np.linalg.inv(a_mat)
        b = np.eye(p) - a_inv
    except np.linalg.LinAlgError:
        b = np.zeros((p, p))

    return RCDResult(tuple(causal_order), b)


def fedhc(
    stats: Cumulants,
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Sort the variables by their third-order tau row sums, then estimate B.

    On population statistics this equals the iterative loop (Proposition 2 of
    the FedISHC paper).

    Parameters
    ----------
    stats : Cumulants
        Aggregated statistics containing ``cov``, ``c3``, ``c12``, ``c21``.
    use_lasso : bool, optional
        If ``True``, estimate B with Adaptive Lasso; otherwise OLS.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.
    """
    c3 = np.array(stats["c3"], dtype=float)
    c12 = np.array(stats["c12"], dtype=float)
    c21 = np.array(stats["c21"], dtype=float)
    cov = np.array(stats["cov"], dtype=float)

    tau = tau_matrix_3rd(c3, c12, c21)
    t_tau_scores = np.sum(tau, axis=1)
    causal_order: list[int] = np.argsort(t_tau_scores).tolist()

    n_samples = _n_samples(stats)
    b: npt.NDArray[np.float64]
    if use_lasso:
        b = estimate_b_lasso_from_cov(causal_order, cov, n_samples=n_samples, lmbda_type=lmbda_type)
    else:
        b = estimate_b_from_cov(causal_order, cov)

    return RCDResult(tuple(causal_order), b)
