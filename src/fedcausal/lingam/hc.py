"""HC and HC-LiNGAM (Chen et al.), centralised fourth-order baselines.

Both work on the pooled data matrix. ``hc_lingam`` sorts once by the
fourth-order tau; ``hc`` removes one source at a time, regressing it out of the
rest. With the default OLS estimate of B, both match the reference
implementation numerically; ``tests/lingam/test_reference_implementation.py``
checks it.

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
    """Return the order of the variables by their fourth-order tau row sums.

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
    """Return the causal order found by removing one source at a time.

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

        # active_indices maps x_curr's columns to the original ones.
        tau_local = tau_matrix_4th_from_data(x_curr)

        t_tau = np.sum(tau_local, axis=1)
        m_local_idx = int(np.argmin(t_tau))
        m_global_idx = active_indices[m_local_idx]

        k.append(m_global_idx)

        # Regress x_m out of the rest, with population (biased) moments.
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

    k.append(active_indices[0])

    return k


def _estimate_adjacency_matrix(
    x: npt.NDArray[np.float64], order: list[int]
) -> npt.NDArray[np.float64]:
    """Estimate B by regressing each variable on the ones before it in `order`.

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
    """Estimate B by per-variable OLS on `x`, or by adaptive Lasso on its covariance.

    Parameters
    ----------
    x : numpy.ndarray
        Centred data, shape ``(n, p)``.
    order : list of int
        Causal order, source first.
    use_lasso : bool
        Use the adaptive Lasso.
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
    """Find the causal order with HC-LiNGAM (Algorithm 2), then estimate B.

    Parameters
    ----------
    x : numpy.ndarray
        Data, shape ``(n, p)``.
    use_lasso : bool, optional
        Estimate B by adaptive Lasso on the covariance instead of OLS.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.
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
    """Find the causal order with HC (Algorithm 1), then estimate B.

    Parameters
    ----------
    x : numpy.ndarray
        Data, shape ``(n, p)``.
    use_lasso : bool, optional
        Estimate B by adaptive Lasso on the covariance instead of OLS.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.
    """
    x_centred = np.asarray(x, dtype=float)
    x_centred = x_centred - np.mean(x_centred, axis=0)

    causal_order = _fit_hc(x_centred)
    b = _estimate_b(x_centred, causal_order, use_lasso=use_lasso, lmbda_type=lmbda_type)

    return RCDResult(tuple(causal_order), b)
