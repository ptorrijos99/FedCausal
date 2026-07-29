"""FedRCD: federated regression-directed cumulants.

Four variants of the same algorithm family:

``fedrcd_noniterative``
    One-shot: computes tau once on the original aggregated stats and sorts
    globally. Federated counterpart of HCLiNGAM. Requires only one
    communication round. Fast, but ignores residual structure.

``fedrcd``
    Iterative algebraic deflation: updates cumulant statistics in place using
    beta coefficients (OLS) after each source removal. One-shot -- no further
    communication after the initial stats exchange. Exact under the true
    LiNGAM model; accumulates numerical error under finite samples or model
    misspecification.

``fedrcd_exact``
    Multi-round exact deflation: ``p - 1`` rounds of communication. Each round
    the server broadcasts ``(source, beta)`` and clients deflate their local
    raw data, then recompute fresh sufficient statistics. No algebraic
    approximation -- exact federated counterpart of HC. Higher communication
    cost, but no error accumulation.

``ishc_beta``
    Improved ISHC baseline using beta-deflation (kept for comparison).

References
----------
[1] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
[2] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from fedcausal.graph import DAG
from fedcausal.lingam.cumulants import (
    Cumulants,
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
)
from fedcausal.lingam.solvers import estimate_b_from_cov, estimate_b_lasso_from_cov
from fedcausal.lingam.tau import tau_matrix_3rd, tau_matrix_4th

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["RCDResult", "fedrcd", "fedrcd_exact", "fedrcd_noniterative", "ishc_beta"]


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
        n_vars = self.b.shape[0]
        edges = [
            (parent, child)
            for child in range(n_vars)
            for parent in range(n_vars)
            if abs(float(self.b[child, parent])) > threshold
        ]
        return DAG.from_edges(n_vars, edges)


def _n_samples(stats: Cumulants) -> int:
    """Look up the total sample count, raising if the aggregator omitted it.

    Parameters
    ----------
    stats : Cumulants
        Aggregated statistics from :func:`aggregate_and_compute_cumulants`.

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


def fedrcd(
    stats: Cumulants,
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Identify the causal order iteratively, then re-estimate B by OLS/Lasso.

    Server-side solver. Updates cumulant statistics in place via beta (OLS)
    deflation after each source removal, then re-estimates B from the
    original (pre-deflation) covariance.

    Parameters
    ----------
    stats : Cumulants
        Aggregated statistics containing ``cov``, ``c3``, ``c12``, ``c21``,
        and optionally ``c4``, ``c13``, ``c31``, ``c22``.
    use_lasso : bool, optional
        If ``True``, estimate B with Adaptive Lasso; otherwise OLS.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.
    """
    cov = np.array(stats["cov"], dtype=float)
    cov_original = cov.copy()
    c3 = np.array(stats["c3"], dtype=float).copy()
    c12 = np.array(stats["c12"], dtype=float).copy()
    c21 = np.array(stats["c21"], dtype=float).copy()

    # Bound unconditionally so they are never possibly-undefined below, even
    # though they are only ever read when `has_4th` (equivalently `use_4th`,
    # which mirrors it) is True. The `if has_4th:` block overwrites these with
    # the real values; the shape here is irrelevant since it is never read.
    c4 = np.zeros(0, dtype=np.float64)
    c13 = np.zeros((0, 0), dtype=np.float64)
    c31 = np.zeros((0, 0), dtype=np.float64)
    c22 = np.zeros((0, 0), dtype=np.float64)

    has_4th = "c4" in stats
    if has_4th:
        c4 = np.array(stats["c4"], dtype=float).copy()
        c13 = np.array(stats["c13"], dtype=float).copy()
        c31 = np.array(stats["c31"], dtype=float).copy()
        c22 = np.array(stats["c22"], dtype=float).copy()

    p = len(c3)
    active = list(range(p))
    causal_order: list[int] = []

    for _iteration in range(p - 1):
        # ── 1. Source Identification ──────────────────────────────────
        # Use 4th-order tau when available (stronger discriminator).
        use_4th = has_4th

        if use_4th:
            tau = tau_matrix_4th(
                c4[active],
                c13[np.ix_(active, active)],
                c31[np.ix_(active, active)],
                c22[np.ix_(active, active)],
            )
        else:
            tau = tau_matrix_3rd(
                c3[active],
                c12[np.ix_(active, active)],
                c21[np.ix_(active, active)],
            )

        t_tau = np.sum(tau, axis=1)
        src_local = int(np.argmin(t_tau))
        s = active[src_local]
        causal_order.append(s)

        # ── 2. Compute β (OLS) — used for ALL updates ────────────────
        var_s = cov[s, s]
        if np.abs(var_s) < 1e-15:
            var_s = 1e-15

        betas = {}
        for j in active:
            if j == s:
                continue
            betas[j] = cov[j, s] / var_s

        # ── 3. Update all statistics using β ─────────────────────────
        remaining = [j for j in active if j != s]

        for i in remaining:
            bi = betas[i]

            # Covariance update
            for j in remaining:
                bj = betas[j]
                cov[i, j] -= bi * bj * var_s

            # 3rd-order cumulant updates (using β, not α)
            c3_s = c3[s]
            c3[i] = c3[i] - (bi**3) * c3_s - 3 * bi * c21[i, s] + 3 * (bi**2) * c12[i, s]

            for j in remaining:
                if i == j:
                    continue
                bj = betas[j]
                c12[i, j] -= bi * (bj**2) * c3_s
                c21[i, j] -= (bi**2) * bj * c3_s

            # 4th-order cumulant updates (using β)
            if has_4th:
                c4_s = c4[s]
                c4[i] = (
                    c4[i]
                    - 4 * bi * c31[i, s]
                    + 6 * (bi**2) * c22[i, s]
                    - 4 * (bi**3) * c13[i, s]
                    + (bi**4) * c4_s
                )

                for j in remaining:
                    if i == j:
                        continue
                    bj = betas[j]
                    c13[i, j] -= bi * (bj**3) * c4_s
                    c31[i, j] -= (bi**3) * bj * c4_s
                    c22[i, j] -= (bi**2) * (bj**2) * c4_s

        # Update diagonals
        for j in remaining:
            c12[j, j] = c3[j]
            c21[j, j] = c3[j]
            if has_4th:
                c13[j, j] = c4[j]
                c31[j, j] = c4[j]
                c22[j, j] = c4[j]

        active = remaining

    if active:
        causal_order.append(active[0])

    # ── 4. Re-estimate B from ORIGINAL covariance ─────────────────────
    n_samples = _n_samples(stats)
    if use_lasso:
        b = estimate_b_lasso_from_cov(
            causal_order, cov_original, n_samples=n_samples, lmbda_type=lmbda_type
        )
    else:
        b = estimate_b_from_cov(causal_order, cov_original)

    return RCDResult(tuple(causal_order), b)


def ishc_beta(
    stats: Cumulants,
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Identify the causal order via ISHC, deflating with beta (OLS).

    This is the improved variant that uses beta (OLS) for deflation instead
    of alpha (the cumulant ratio) that the original ISHC baseline uses.

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
    c3 = np.array(stats["c3"], dtype=float).copy()
    c12 = np.array(stats["c12"], dtype=float).copy()
    c21 = np.array(stats["c21"], dtype=float).copy()
    cov = np.array(stats["cov"], dtype=float)
    cov_original = cov.copy()

    p = len(c3)
    active = list(range(p))
    causal_order: list[int] = []

    for _ in range(p - 1):
        tau = tau_matrix_3rd(
            c3[active],
            c12[np.ix_(active, active)],
            c21[np.ix_(active, active)],
        )
        t_tau = np.sum(tau, axis=1)
        src_local = int(np.argmin(t_tau))
        s = active[src_local]
        causal_order.append(s)
        active.pop(src_local)

        # Use β (OLS) instead of α (c21/c3)
        var_s = cov[s, s]
        if np.abs(var_s) < 1e-15:
            var_s = 1e-15
        c3_s = c3[s]

        for j in active:
            bj = cov[j, s] / var_s

            # C3 update (full formula with β)
            c3[j] = c3[j] - (bj**3) * c3_s - 3 * bj * c21[j, s] + 3 * (bj**2) * c12[j, s]

            # Cross-cumulant updates
            for k in active:
                if k == j:
                    continue
                bk = cov[k, s] / var_s
                c12[j, k] -= bj * (bk**2) * c3_s
                c21[j, k] -= (bj**2) * bk * c3_s

            # Also update covariance for next iteration's β
            for k in active:
                cov[j, k] -= bj * (cov[k, s] / var_s) * var_s

        # Update diagonals
        for j in active:
            c12[j, j] = c3[j]
            c21[j, j] = c3[j]

    if active:
        causal_order.append(active[0])

    # Re-estimate B from original covariance
    n_samples = _n_samples(stats)
    if use_lasso:
        b = estimate_b_lasso_from_cov(
            causal_order, cov_original, n_samples=n_samples, lmbda_type=lmbda_type
        )
    else:
        b = estimate_b_from_cov(causal_order, cov_original)

    return RCDResult(tuple(causal_order), b)


def fedrcd_noniterative(
    stats: Cumulants,
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Sort by a single global tau pass, then re-estimate B by OLS/Lasso.

    Avoids iterative cumulant deflation entirely. Computes tau once from the
    aggregated stats and sorts by row sum (like HCLiNGAM), then estimates B
    from the original covariance.

    Parameters
    ----------
    stats : Cumulants
        Aggregated statistics containing ``cov`` and either (``c4``, ``c13``,
        ``c31``, ``c22``) or (``c3``, ``c12``, ``c21``).
    use_lasso : bool, optional
        If ``True``, estimate B with Adaptive Lasso; otherwise OLS.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.
    """
    cov = np.array(stats["cov"], dtype=float)
    has_4th = "c4" in stats

    if has_4th:
        c4 = np.array(stats["c4"], dtype=float)
        c13 = np.array(stats["c13"], dtype=float)
        c31 = np.array(stats["c31"], dtype=float)
        c22 = np.array(stats["c22"], dtype=float)
        tau = tau_matrix_4th(c4, c13, c31, c22)
    else:
        c3 = np.array(stats["c3"], dtype=float)
        c12 = np.array(stats["c12"], dtype=float)
        c21 = np.array(stats["c21"], dtype=float)
        tau = tau_matrix_3rd(c3, c12, c21)

    # Sort by total τ score (ascending = most exogenous first)
    t_tau = np.sum(tau, axis=1)
    causal_order = np.argsort(t_tau).tolist()

    # B re-estimation
    n_samples = _n_samples(stats)
    if use_lasso:
        b = estimate_b_lasso_from_cov(causal_order, cov, n_samples=n_samples, lmbda_type=lmbda_type)
    else:
        b = estimate_b_from_cov(causal_order, cov)

    return RCDResult(tuple(causal_order), b)


def fedrcd_exact(
    clients: Sequence[npt.NDArray[np.float64]],
    *,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Run multi-round exact federated deflation.

    Runs ``p - 1`` communication rounds. Each round:

    1. Clients compute fresh sufficient statistics from their current
       (deflated) local data and send them to the server.
    2. Server aggregates stats, identifies the source variable via tau4,
       and computes OLS coefficients beta from the current covariance.
    3. Server broadcasts ``(src_local_idx, beta)`` back to clients.
    4. Clients deflate: ``x_j := x_j - beta_j * x_s`` for all ``j != s``,
       then drop column ``s`` from their local data.

    Because statistics are recomputed from actual residuals at every round,
    no algebraic approximation is involved and numerical errors do not
    accumulate across steps. This is the exact federated counterpart of HC.

    Parameters
    ----------
    clients : Sequence[numpy.ndarray]
        Local datasets, each shape ``(n_k, p)``. Raw data is used only
        locally; only sufficient statistics (moment sums) leave each client.
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
    Communication cost: ``p - 1`` rounds x O(p^2) statistics per round =
    O(p^3) total.
    """
    p = clients[0].shape[1]
    causal_order: list[int] = []

    # Each client maintains a working copy of its (progressively deflated) data.
    # Only sufficient statistics cross the network — raw data stays local.
    x_local = [x.copy() for x in clients]

    # Round 0 also gives us the original global covariance for final B estimation.
    init_raw = [compute_local_raw_stats(x, order=4) for x in x_local]
    init_agg = aggregate_and_compute_cumulants(init_raw)
    cov_original = np.array(init_agg["cov"])
    n_total = int(init_agg["n"])

    # active_global[local_idx] → original variable index
    active_global: list[int] = list(range(p))

    # Reuse the stats already computed for round 0
    agg = init_agg

    for step in range(p - 1):
        n_active = len(active_global)

        # ── 1. Source identification via τ₄ ──────────────────────────
        c4 = np.array(agg["c4"])
        c13 = np.array(agg["c13"])
        c31 = np.array(agg["c31"])
        c22 = np.array(agg["c22"])

        tau = tau_matrix_4th(c4, c13, c31, c22)
        src_local = int(np.argmin(np.sum(tau, axis=1)))
        causal_order.append(active_global[src_local])

        # ── 2. Compute β from current covariance ─────────────────────
        cov_curr = np.array(agg["cov"])
        var_s = max(float(cov_curr[src_local, src_local]), 1e-15)
        # betas[j] = OLS coefficient for regressing x_j on x_s
        betas = cov_curr[:, src_local] / var_s  # shape (n_active,)

        # ── 3. Clients deflate locally and drop source column ─────────
        remaining_local = [j for j in range(n_active) if j != src_local]
        new_x_local = []
        for x in x_local:
            x_s = x[:, src_local]
            x_new = np.column_stack([x[:, j] - betas[j] * x_s for j in remaining_local])
            new_x_local.append(x_new)
        x_local = new_x_local
        active_global = [active_global[j] for j in remaining_local]

        # ── 4. Clients send fresh stats for the next round ────────────
        if step < p - 2:
            raw = [compute_local_raw_stats(x, order=4) for x in x_local]
            agg = aggregate_and_compute_cumulants(raw)

    if active_global:
        causal_order.append(active_global[0])

    # ── 5. Re-estimate B from ORIGINAL covariance ─────────────────────
    if use_lasso:
        b = estimate_b_lasso_from_cov(
            causal_order, cov_original, n_samples=n_total, lmbda_type=lmbda_type
        )
    else:
        b = estimate_b_from_cov(causal_order, cov_original)

    return RCDResult(tuple(causal_order), b)
