"""FedISHC: the published third-order-deflation baseline for federated LiNGAM.

Ports the ``ishc`` and ``hc`` functions from Chen et al.'s FedISHC reference
implementation. Both recover a causal order and a coefficient matrix from
aggregated 3rd-order cumulants alone -- no raw data is needed beyond the
initial statistics exchange -- but they differ in how (or whether) they
remove an identified source's influence before ranking the rest:

``fedishc``
    Full FedISHC. Iteratively identifies the most exogenous remaining
    variable, then algebraically deflates the remaining cumulants using the
    *alpha-ratio* (``c21[j, s] / c3[s]``, clipped). B is recovered from the
    accumulated mixing matrix, ``B = I - A^-1``.

``fedhc``
    The no-deflation ablation: sorts once by the initial tau^(3) row sums
    and stops there. Since it performs no deflation, it has no mixing matrix
    to invert; B is re-estimated externally from the aggregated covariance,
    exactly as :func:`fedcausal.lingam.rcd.fedrcd_noniterative` does for
    FedRCD.

References
----------
[1] Chen, W., et al. "FedISHC: Federated Causal Discovery via Higher-Order
    Cumulants." AAAI / arXiv, 2025.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fedcausal.lingam.cumulants import Cumulants
from fedcausal.lingam.rcd import RCDResult
from fedcausal.lingam.solvers import estimate_b_from_cov, estimate_b_lasso_from_cov
from fedcausal.lingam.tau import tau_matrix_3rd

__all__ = ["fedhc", "fedishc"]

# Numerical safeguards used by the alpha-ratio deflation in `fedishc`.
# Transcribed verbatim from the original FedISHC implementation -- see that
# function's Notes section for why these must not be adjusted.
_EPS = 1e-6
_ALPHA_CLIP = 10.0


def _n_samples(stats: Cumulants) -> int:
    """Look up the total sample count, raising if the aggregator omitted it.

    Parameters
    ----------
    stats : Cumulants
        Aggregated statistics from
        :func:`fedcausal.lingam.cumulants.aggregate_and_compute_cumulants`.

    Returns
    -------
    int
        The ``"n"`` entry.

    Raises
    ------
    KeyError
        If ``"n"`` is missing. A silent default would quietly mis-scale the
        Lasso penalty.
    """
    if "n" not in stats:
        raise KeyError("stats is missing 'n'; the aggregator always sets it")
    return int(stats["n"])


def fedishc(
    stats: Cumulants,
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Recover the causal order and B via FedISHC's iterative alpha deflation.

    Ports ``ishc`` from the original FedISHC implementation unchanged.
    Iteratively identifies the currently most exogenous variable via the
    3rd-order tau discriminator, then deflates the remaining cumulants using
    ``alpha = clip(c21[j, s] / c3[s], -10, 10)`` -- the ratio of a
    cross-cumulant to the source's marginal 3rd cumulant. B is recovered
    from the accumulated mixing matrix as ``B = I - A^-1``.

    Parameters
    ----------
    stats : Cumulants
        Aggregated statistics containing ``c3``, ``c12``, ``c21``.
    use_lasso : bool, optional
        Unused. Kept only so `fedishc` shares a call signature with the
        other three solvers in this subpackage; FedISHC's B always comes
        from inverting the accumulated mixing matrix, never from a
        covariance regression.
    lmbda_type : str, optional
        Unused, for the same reason as `use_lasso`.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Notes
    -----
    `use_lasso` and `lmbda_type` are accepted and silently ignored. This
    mirrors the original: ``ishc()`` in the reference implementation never
    took these parameters at all -- there is no OLS/Lasso B-estimation step
    to select between, since B always comes from ``I - A^-1``. They exist
    here purely so callers can iterate over all four solvers in this
    subpackage with one call signature.

    **Do not change this function's numerics.** FedISHC is a published
    baseline from another group's paper (Chen et al., 2025); the comparison
    this port exists for is only meaningful if `fedishc` behaves exactly as
    its authors implemented it. In particular, the alpha-ratio deflation
    used here (``c21[j, s] / c3[s]``, clipped to ``[-10, 10]``) is *not* a
    bug and is not to be "corrected" to the beta/OLS deflation that
    :func:`fedcausal.lingam.rcd.fedrcd` uses -- that difference is precisely
    what FedRCD's beta deflation claims to improve on. `_EPS` and
    `_ALPHA_CLIP` are transcribed from the original source and must not be
    retuned.

    References
    ----------
    [1] Chen, W., et al. "FedISHC: Federated Causal Discovery via
        Higher-Order Cumulants." AAAI / arXiv, 2025.
    """
    c3 = np.array(stats["c3"], dtype=float).copy()
    c12 = np.array(stats["c12"], dtype=float).copy()
    c21 = np.array(stats["c21"], dtype=float).copy()

    p = len(c3)
    active = list(range(p))
    causal_order: list[int] = []

    # Mixing matrix A (X = A * E). Initialised as identity.
    a_mat = np.eye(p)

    for _ in range(p - 1):
        # 1. Source identification on the currently active variables.
        tau = tau_matrix_3rd(c3[active], c12[np.ix_(active, active)], c21[np.ix_(active, active)])
        t_tau_scores = np.sum(tau, axis=1)

        min_idx_local = int(np.argmin(t_tau_scores))
        s = active[min_idx_local]
        causal_order.append(s)
        active.pop(min_idx_local)

        # 2. Algebraic deflation of remaining variables.
        # Freeze c3[s] before mutating c3: deflation uses the value at the
        # moment s was identified, not the running estimate.
        c3_s = c3[s]

        for xj in active:
            # alpha = effect of source s on x_j
            if np.abs(c3_s) < _EPS:
                alpha = 0.0
            else:
                alpha = float(np.clip(c21[xj, s] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP))

            a_mat[xj, s] = alpha

            # Update C3(x_j): C3'(x_j) = C3(x_j) - alpha^3 * C3(s)
            c3[xj] = c3[xj] - (alpha**3) * c3_s

            for xk in active:
                # lambda = effect of source s on x_k
                if np.abs(c3_s) > _EPS:
                    lam = float(np.clip(c21[xk, s] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP))
                else:
                    lam = 0.0

                # C1,2'(x_j, x_k) = C1,2(x_j, x_k) - alpha * lambda^2 * C3(s)
                c12[xj, xk] -= alpha * (lam**2) * c3_s
                # C2,1'(x_j, x_k) = C2,1(x_j, x_k) - alpha^2 * lambda * C3(s)
                c21[xj, xk] -= (alpha**2) * lam * c3_s

    # Last variable
    if active:
        causal_order.append(active[0])

    # Recover B from A
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
    """Recover the causal order via a single tau^(3) pass, no deflation.

    Ports the default (``iterative=False``) branch of ``hc`` from the
    original FedISHC implementation: the no-deflation variant of
    :func:`fedishc`. Sorts all variables by their 3rd-order tau row sums in
    one pass -- equivalent to the iterative argmin loop on population
    statistics, per the accompanying paper's Proposition 2. Because no
    cumulant is deflated, there is no mixing matrix to invert, so B is
    re-estimated from the aggregated covariance, exactly as
    :func:`fedcausal.lingam.rcd.fedrcd_noniterative` does for FedRCD.

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

    Notes
    -----
    The order-finding step (tau^(3) row sums, single pass) is transcribed
    unchanged from the original ``hc`` baseline; only the B estimate is new,
    since the original returns ``None`` for B and leaves its recovery to the
    caller.

    References
    ----------
    [1] Chen, W., et al. "FedISHC: Federated Causal Discovery via
        Higher-Order Cumulants." AAAI / arXiv, 2025.
    """
    c3 = np.array(stats["c3"], dtype=float)
    c12 = np.array(stats["c12"], dtype=float)
    c21 = np.array(stats["c21"], dtype=float)
    cov = np.array(stats["cov"], dtype=float)

    # Single-pass implementation: equivalent to the iterative form on
    # population statistics by Proposition 2 of the accompanying paper.
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
