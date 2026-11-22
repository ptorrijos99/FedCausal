"""The FedRCD solvers and the cumulant baselines they generalise.

Every solver finds a causal order with the tau discriminator, third or fourth
order, and then estimates B on that order from the covariance. They differ
only in what happens after a source is found:

``method="deflation"`` (FedRCD)
    Removes the source from the aggregated cumulants with the OLS coefficients
    ``cov[j, s] / var[s]`` and searches again. One round of communication.
``method="ratio"`` (FedISHC with ``order=3``)
    The same, with the cumulant-ratio coefficients ``Cum(s, .., s, j) / Cum(s, .., s)``.
``method="sort"`` (FedHC with ``order=3``, HC-LiNGAM with one site)
    No deflation: sorts once by the tau row sums.
:func:`fedrcd_exact` (HC with one site)
    The sites deflate their own data, one round per source, so no estimation
    error carries from one step to the next.

References
----------
[1] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
[2] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
[3] Chen, W., et al. "FedISHC: Federated Causal Discovery via Higher-Order
    Cumulants." AAAI / arXiv, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from fedcausal.graph import DAG
from fedcausal.lingam.cumulants import (
    Cumulants,
    _n_samples,
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
)
from fedcausal.lingam.solvers import estimate_b_from_cov, estimate_b_lasso_from_cov
from fedcausal.lingam.tau import tau_matrix_3rd, tau_matrix_4th

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["RCDResult", "exact_order", "fedrcd", "fedrcd_exact"]

Method = Literal["deflation", "ratio", "sort"]
_METHODS = ("deflation", "ratio", "sort")

# Below this, a source's own cumulant is treated as zero and the ratio as 0.
_RATIO_EPS = 1e-6
# Cumulant ratios are clipped to this magnitude, as in the FedISHC authors' code.
_RATIO_CLIP = 10.0


@dataclass(frozen=True)
class RCDResult:
    """A causal order and the coefficient matrix estimated for it.

    Attributes
    ----------
    causal_order : tuple of int
        Variable indices, source first. A topological order of the estimated
        structure.
    b : numpy.ndarray
        Shape ``(p, p)``. ``b[child, parent]`` is the linear coefficient of
        `parent` in `child`'s structural equation, and is zero whenever
        `parent` does not precede `child` in `causal_order`.
    """

    causal_order: tuple[int, ...]
    b: npt.NDArray[np.float64]

    def to_dag(self, threshold: float = 0.0) -> DAG:
        """Return the DAG of coefficients whose magnitude exceeds `threshold`.

        Parameters
        ----------
        threshold : float, optional
            Keep an edge only when ``abs(b[child, parent]) > threshold``. The
            default keeps every non-zero coefficient.

        Returns
        -------
        DAG
            Acyclic by construction: `b` is lower-triangular with respect to
            `causal_order`, so every edge runs from an earlier variable to a
            later one.
        """
        children, parents = np.nonzero(np.abs(self.b) > threshold)
        return DAG.from_edges(
            self.b.shape[0], list(zip(parents.tolist(), children.tolist(), strict=True))
        )


def _resolve_order(stats: Cumulants, order: int | None) -> int:
    if order is None:
        return 4 if "c4" in stats else 3
    if order not in (3, 4):
        raise ValueError(f"order must be 3 or 4, got {order!r}")
    if order == 4 and "c4" not in stats:
        raise ValueError("order=4 needs fourth-order statistics; the sites sent only third-order")
    return order


def _tau_row_sums(
    stats: Cumulants, order: int, idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    block = np.ix_(idx, idx)
    if order == 4:
        tau = tau_matrix_4th(
            stats["c4"][idx], stats["c13"][block], stats["c31"][block], stats["c22"][block]
        )
    else:
        tau = tau_matrix_3rd(stats["c3"][idx], stats["c12"][block], stats["c21"][block])
    sums: npt.NDArray[np.float64] = tau.sum(axis=1)
    return sums


def _coefficients(
    causal_order: Sequence[int],
    cov: npt.NDArray[np.float64],
    n_samples: int,
    use_lasso: bool,
    lmbda_type: str,
) -> npt.NDArray[np.float64]:
    if use_lasso:
        return estimate_b_lasso_from_cov(
            causal_order, cov, n_samples=n_samples, lmbda_type=lmbda_type
        )
    return estimate_b_from_cov(causal_order, cov)


def _ratio(num: npt.NDArray[np.float64], den: float) -> npt.NDArray[np.float64]:
    if abs(den) < _RATIO_EPS:
        return np.zeros_like(num)
    return np.clip(num / den, -_RATIO_CLIP, _RATIO_CLIP)


def _deflated_order(cur: Cumulants, order: int, method: Method) -> list[int]:
    """Remove one source at a time from the cumulants in `cur`, in place."""
    active = list(range(len(cur["cov"])))
    causal_order: list[int] = []
    while len(active) > 1:
        s = active.pop(int(np.argmin(_tau_row_sums(cur, order, np.asarray(active)))))
        causal_order.append(s)
        r = np.asarray(active)
        rr = np.ix_(r, r)
        var_s = max(float(cur["cov"][s, s]), 1e-15)
        if method == "deflation":
            b = cur["cov"][r, s] / var_s
        elif order == 4:
            b = _ratio(cur["c13"][r, s], float(cur["c4"][s]))
        else:
            b = _ratio(cur["c12"][r, s], float(cur["c3"][s]))
        b2, b3 = b * b, b * b * b

        # x_i <- x_i - b_i x_s for every remaining i. Ratio deflation assumes, as
        # FedISHC does, that x_i - b_i x_s is independent of the source, so a
        # marginal cumulant loses only its b_i^k part.
        ratio = method == "ratio"
        cur["cov"][rr] += (
            np.outer(b, b) * var_s - np.outer(b, cur["cov"][r, s]) - np.outer(cur["cov"][r, s], b)
        )
        if order == 3:
            c3_s = cur["c3"][s]
            if ratio:
                cur["c3"][r] -= b3 * c3_s
            else:
                cur["c3"][r] += -b3 * c3_s - 3 * b * cur["c21"][r, s] + 3 * b2 * cur["c12"][r, s]
            cur["c12"][rr] -= np.outer(b, b2) * c3_s
            cur["c21"][rr] -= np.outer(b2, b) * c3_s
            cur["c12"][r, r] = cur["c3"][r]
            cur["c21"][r, r] = cur["c3"][r]
        else:
            c4_s = cur["c4"][s]
            if ratio:
                cur["c4"][r] -= b2 * b2 * c4_s
            else:
                cur["c4"][r] += (
                    -4 * b * cur["c31"][r, s]
                    + 6 * b2 * cur["c22"][r, s]
                    - 4 * b3 * cur["c13"][r, s]
                    + b2 * b2 * c4_s
                )
            cur["c13"][rr] -= np.outer(b, b3) * c4_s
            cur["c31"][rr] -= np.outer(b3, b) * c4_s
            cur["c22"][rr] -= np.outer(b2, b2) * c4_s
            for key in ("c13", "c31", "c22"):
                cur[key][r, r] = cur["c4"][r]
    causal_order.extend(active)
    return causal_order


def fedrcd(
    stats: Cumulants,
    *,
    method: Method = "deflation",
    order: Literal[3, 4] | None = None,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Find the causal order from aggregated cumulants, then estimate B.

    Parameters
    ----------
    stats : Cumulants
        Output of :func:`~fedcausal.lingam.cumulants.aggregate_and_compute_cumulants`.
    method : {"deflation", "ratio", "sort"}, optional
        What follows each source; see the module docstring.
    order : {3, 4} or None, optional
        Which tau discriminator to use. Defaults to 4 when `stats` has
        fourth-order cumulants, else 3.
    use_lasso : bool, optional
        If ``True``, estimate B with Adaptive Lasso; otherwise OLS.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Raises
    ------
    ValueError
        If `method` or `order` is not one of the allowed values, or ``order=4``
        is asked of third-order statistics.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    k = _resolve_order(stats, order)
    cov = np.asarray(stats["cov"], dtype=np.float64)
    if method == "sort":
        causal_order = np.argsort(
            _tau_row_sums(stats, k, np.arange(len(cov))), kind="stable"
        ).tolist()
    else:
        keys = ("c3", "c12", "c21") if k == 3 else ("c4", "c13", "c31", "c22")
        cur = {key: np.array(stats[key], dtype=np.float64) for key in ("cov", *keys)}
        causal_order = _deflated_order(cur, k, method)
    b_est = _coefficients(causal_order, cov, _n_samples(stats), use_lasso, lmbda_type)
    return RCDResult(tuple(causal_order), b_est)


def exact_order(
    first: Cumulants,
    deflate: Callable[[int, npt.NDArray[np.float64]], Cumulants],
    order: Literal[3, 4] | None = None,
) -> list[int]:
    """Run the order search of :func:`fedrcd_exact` over any transport.

    Parameters
    ----------
    first : Cumulants
        The pooled statistics of the sites' original data.
    deflate : callable
        ``deflate(source, betas)`` has every site replace each remaining
        column ``j`` by ``x_j - betas[j] * x_source``, drop column `source`
        (both positions among the remaining columns), and returns the pooled
        statistics of the result.
    order : {3, 4} or None, optional
        Which tau discriminator to use; as in :func:`fedrcd`.

    Returns
    -------
    list of int
        The causal order, in the original column indices.
    """
    k = _resolve_order(first, order)
    active = list(range(len(first["cov"])))
    causal_order: list[int] = []
    stats = first
    while len(active) > 1:
        src = int(np.argmin(_tau_row_sums(stats, k, np.arange(len(active)))))
        causal_order.append(active.pop(src))
        cov = np.asarray(stats["cov"], dtype=np.float64)
        betas = cov[:, src] / max(float(cov[src, src]), 1e-15)
        if len(active) > 1:
            stats = deflate(src, betas)
    causal_order.extend(active)
    return causal_order


def fedrcd_exact(
    sites: Sequence[npt.ArrayLike],
    *,
    order: Literal[3, 4] = 4,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Find the causal order by deflating each site's data, one round per source.

    Each round the sites send the moment sums of their current data, the
    server picks the source by tau and the OLS coefficients on it, and every
    site deflates its data by them (``x_j -= beta_j * x_s``) and drops the
    source column. Runs the whole protocol in one process;
    :func:`fedcausal.flower.lingam.run_exact_server_loop` is the same over
    Flower. With a single site this is HC (Chen et al., 2025).

    Parameters
    ----------
    sites : sequence of array-like
        Each site's data, shape ``(n_k, p)``.
    order : {3, 4}, optional
        Which tau discriminator to use, and so which moment sums the sites send.
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
    ``p - 1`` rounds of O(p^2) statistics each.
    """
    local = [np.asarray(x, dtype=np.float64) for x in sites]

    def pooled() -> Cumulants:
        return aggregate_and_compute_cumulants([compute_local_raw_stats(x, order) for x in local])

    def deflate(src: int, betas: npt.NDArray[np.float64]) -> Cumulants:
        keep = np.arange(len(betas)) != src
        for k, x in enumerate(local):
            local[k] = x[:, keep] - np.outer(x[:, src], betas[keep])
        return pooled()

    first = pooled()
    causal_order = exact_order(first, deflate, order)
    b_est = _coefficients(
        causal_order, np.asarray(first["cov"]), int(first["n"]), use_lasso, lmbda_type
    )
    return RCDResult(tuple(causal_order), b_est)
