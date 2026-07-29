"""HC and HC-LiNGAM: centralised 4th-order-cumulant baselines.

Ports the two causal-ordering algorithms of Chen et al.'s ``HCLiNGAM`` class
(``method='hc'`` and ``method='hc-lingam'``), plus its OLS adjacency-matrix
estimator. Both work directly on a centralised data matrix -- there is no
federated aggregation step here, unlike the rest of this subpackage.

``hc_lingam``
    Algorithm 2 (regression-free). Computes the 4th-order tau matrix once
    over all variables and sorts by row sum. Cheaper, but assumes the global
    ordering induced by a single pass holds.

``hc``
    Algorithm 1 (iterative regression). Identifies one exogenous variable at
    a time from the tau matrix of the variables still active, regresses its
    linear effect out of the rest, and repeats -- similar in spirit to
    DirectLiNGAM but using the 4th-order tau criterion in place of HSIC.

References
----------
[1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fedcausal.lingam.rcd import RCDResult
from fedcausal.lingam.solvers import estimate_b_lasso_from_cov
from fedcausal.lingam.tau import tau_matrix_4th_from_data

__all__ = ["hc", "hc_lingam"]


def _fit_hc_lingam(x: npt.NDArray[np.float64]) -> list[int]:
    """Estimate the causal order via Algorithm 2 (regression-free global ordering).

    Sorts variables by total 4th-order tau score. The variable with the
    smallest total score is the most exogenous.

    Parameters
    ----------
    x : numpy.ndarray
        Centred data, shape ``(n, p)``.

    Returns
    -------
    list of int
        Estimated causal order, source first.
    """
    _n, p = x.shape
    if p < 2:
        return list(range(p))

    tau = tau_matrix_4th_from_data(x)
    t_tau_scores = np.sum(tau, axis=1)
    causal_order: list[int] = np.argsort(t_tau_scores).tolist()
    return causal_order


def _fit_hc(x: npt.NDArray[np.float64]) -> list[int]:
    """Estimate the causal order via Algorithm 1 (iterative regression).

    Identifies one exogenous variable at a time, regresses its effect out of
    the remaining variables, and repeats.

    Parameters
    ----------
    x : numpy.ndarray
        Centred data, shape ``(n, p)``.

    Returns
    -------
    list of int
        Estimated causal order, source first.
    """
    _n, p = x.shape
    k: list[int] = []
    x_curr = x.copy()
    active_indices = list(range(p))

    for _ in range(p - 1):
        n_curr = len(active_indices)

        # X_curr is already the reduced dataset (n, n_curr); active_indices
        # maps its local column index to the original global index.
        tau_local = tau_matrix_4th_from_data(x_curr)

        # Find root in the current set (smallest row sum).
        t_tau = np.sum(tau_local, axis=1)
        m_local_idx = int(np.argmin(t_tau))
        m_global_idx = active_indices[m_local_idx]

        k.append(m_global_idx)

        # Remove the effect of x_m from the rest by regression:
        # x_j := x_j - (cov(xj, xm) / var(xm)) * xm
        xm_data = x_curr[:, m_local_idx]
        var_xm = np.var(xm_data)

        new_x = []
        new_active_indices = []

        for i in range(n_curr):
            if i == m_local_idx:
                continue

            xi_data = x_curr[:, i]
            cov_im = np.cov(xi_data, xm_data, bias=True)[0, 1]
            coef = cov_im / var_xm
            resid = xi_data - coef * xm_data

            new_x.append(resid)
            new_active_indices.append(active_indices[i])

        x_curr = np.array(new_x).T
        active_indices = new_active_indices

    # Add the last remaining variable.
    k.append(active_indices[0])

    return k


def _estimate_adjacency_matrix(
    x: npt.NDArray[np.float64], order: list[int]
) -> npt.NDArray[np.float64]:
    """Estimate B given the causal order by OLS regression.

    For each variable, regress it on its predecessors in the order.

    Parameters
    ----------
    x : numpy.ndarray
        Centred data, shape ``(n, p)``.
    order : list of int
        Causal order, source first.

    Returns
    -------
    numpy.ndarray
        Estimated adjacency matrix, shape ``(p, p)``.
    """
    p = x.shape[1]
    b = np.zeros((p, p))

    for i_idx, target_var in enumerate(order):
        if i_idx == 0:
            continue

        predictors_indices = order[:i_idx]

        y = x[:, target_var]
        a_mat = x[:, predictors_indices]

        # OLS: coef = (A^T A)^-1 A^T y
        coefs, _, _, _ = np.linalg.lstsq(a_mat, y, rcond=None)

        for k_idx, split_idx in enumerate(predictors_indices):
            b[target_var, split_idx] = coefs[k_idx]

    return b


def _estimate_b(
    x: npt.NDArray[np.float64],
    order: list[int],
    *,
    use_lasso: bool,
    lmbda_type: str,
) -> npt.NDArray[np.float64]:
    """Dispatch to OLS (original) or Adaptive Lasso adjacency-matrix estimation.

    Parameters
    ----------
    x : numpy.ndarray
        Centred data, shape ``(n, p)``.
    order : list of int
        Causal order, source first.
    use_lasso : bool
        If ``True``, estimate B with Adaptive Lasso from the sample
        covariance instead of the original's per-target OLS regression.
    lmbda_type : str
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    numpy.ndarray
        Estimated adjacency matrix, shape ``(p, p)``.
    """
    if use_lasso:
        cov = np.cov(x, rowvar=False)
        return estimate_b_lasso_from_cov(
            order, np.atleast_2d(cov), n_samples=x.shape[0], lmbda_type=lmbda_type
        )
    return _estimate_adjacency_matrix(x, order)


def hc_lingam(
    x: npt.NDArray[np.float64],
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Recover the causal order and B via HC-LiNGAM (Algorithm 2).

    Ports ``HCLiNGAM(method='hc-lingam').fit(X)`` from the original
    implementation. Computes a single global 4th-order tau matrix and sorts
    by row sum, then estimates B by per-target OLS regression on the
    centred data -- exactly as the original does when `use_lasso` is
    ``False``.

    Parameters
    ----------
    x : numpy.ndarray
        Data, shape ``(n, p)``.
    use_lasso : bool, optional
        If ``True``, estimate B with Adaptive Lasso from the sample
        covariance instead of the original's per-target OLS regression. Not
        part of the original algorithm; added for interface parity with the
        other solvers in this subpackage.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Notes
    -----
    **Do not change this function's numerics** when ``use_lasso=False``.
    HC-LiNGAM is a baseline from Chen et al. (IEEE TNNLS, 2025); the causal
    order (global argsort of 4th-order tau row sums) and the default OLS
    coefficient estimate (``numpy.linalg.lstsq`` per target, on the centred
    data) are transcribed unchanged from the reference implementation.

    References
    ----------
    [1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct
        and Efficient Causal Discovery." IEEE TNNLS, 2025.
    """
    x_centred = np.asarray(x, dtype=float)
    x_centred = x_centred - np.mean(x_centred, axis=0)

    causal_order = _fit_hc_lingam(x_centred)
    b = _estimate_b(x_centred, causal_order, use_lasso=use_lasso, lmbda_type=lmbda_type)

    return RCDResult(tuple(causal_order), b)


def hc(
    x: npt.NDArray[np.float64],
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Recover the causal order and B via HC (Algorithm 1).

    Ports ``HCLiNGAM(method='hc').fit(X)`` from the original implementation.
    Iteratively identifies the currently most exogenous variable via the
    4th-order tau matrix of the active set, regresses its linear effect out
    of the rest (population covariance / population variance, i.e.
    ``bias=True``), and repeats. B is then estimated by per-target OLS
    regression on the centred data -- exactly as the original does when
    `use_lasso` is ``False``.

    Parameters
    ----------
    x : numpy.ndarray
        Data, shape ``(n, p)``.
    use_lasso : bool, optional
        If ``True``, estimate B with Adaptive Lasso from the sample
        covariance instead of the original's per-target OLS regression. Not
        part of the original algorithm; added for interface parity with the
        other solvers in this subpackage.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Notes
    -----
    **Do not change this function's numerics** when ``use_lasso=False``. HC
    is a baseline from Chen et al. (IEEE TNNLS, 2025); its iterative
    regression step deliberately uses the *biased* (population)
    covariance/variance estimators of the reference implementation, and its
    default OLS coefficient estimate (``numpy.linalg.lstsq`` per target, on
    the centred data) is transcribed unchanged.

    References
    ----------
    [1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct
        and Efficient Causal Discovery." IEEE TNNLS, 2025.
    """
    x_centred = np.asarray(x, dtype=float)
    x_centred = x_centred - np.mean(x_centred, axis=0)

    causal_order = _fit_hc(x_centred)
    b = _estimate_b(x_centred, causal_order, use_lasso=use_lasso, lmbda_type=lmbda_type)

    return RCDResult(tuple(causal_order), b)
