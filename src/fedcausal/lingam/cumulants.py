"""Cumulant estimators for causal discovery.

This module provides unbiased estimators for 3rd- and 4th-order cumulants and
cross-cumulants, designed for both local (single-machine) and federated
learning scenarios.

Architecture
------------
The module is organized in two layers:

1. **Raw moment computation** -- `compute_local_raw_stats`. Computes additive
   sufficient statistics (sums of powers / cross-products) from a local
   dataset. These are the *only* quantities that leave each client in a
   federated setting.
2. **Aggregation & cumulant estimation** -- `aggregate_and_compute_cumulants`.
   Aggregates raw moment sums from one or more clients, centers them using the
   global mean, and applies bias-corrected (k-statistic) formulas to obtain
   unbiased cumulant estimates up to 4th order.

A third layer of single-vector convenience wrappers (`unbiased_cumulant3`,
`unbiased_cumulant4`, `cross_cumulants3`, `cross_cumulants4`) presents the same
API as the original single-machine estimators, for callers that already hold
the raw vectors and have no need for the raw-stats pipeline.

References
----------
[1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
[2] Schefczik, N. and Hägele, M. "Ready-to-use unbiased estimators for
    multivariate cumulants including one that outperforms x^3."
    arXiv:1904.12154, 2019.
[3] Chen, W., et al. "FedISHC: Federated Causal Discovery via Higher-Order
    Cumulants." AAAI / arXiv, 2025.
[4] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Cumulants",
    "RawStats",
    "aggregate_and_compute_cumulants",
    "compute_local_raw_stats",
    "cross_cumulants3",
    "cross_cumulants4",
    "unbiased_cumulant3",
    "unbiased_cumulant4",
]

#: Raw, additive moment sums produced by :func:`compute_local_raw_stats`.
RawStats = dict[str, Any]

#: Unbiased cumulant estimates produced by :func:`aggregate_and_compute_cumulants`.
Cumulants = dict[str, Any]

_MIN_SAMPLES_3RD: int = 3
_MIN_SAMPLES_4TH: int = 4


def _n_samples(stats: Cumulants) -> int:
    """Look up the total sample count, raising if the aggregator omitted it.

    Shared by every solver in :mod:`fedcausal.lingam` that needs the sample
    count to scale its Lasso penalty (:mod:`fedcausal.lingam.rcd` and
    :mod:`fedcausal.lingam.ishc`); byte-identical between the two before being
    hoisted here.

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


# ═══════════════════════════════════════════════════════════════════════════
# §1  Raw Moment Computation (client-side, privacy-safe)
# ═══════════════════════════════════════════════════════════════════════════


def compute_local_raw_stats(x: npt.NDArray[np.float64], order: int = 4) -> RawStats:
    """Compute local raw moment sums suitable for federated aggregation.

    All returned quantities are *additive* across disjoint subsets of samples,
    which is the key property that enables secure aggregation without sharing
    raw data.

    Parameters
    ----------
    x : numpy.ndarray
        Local dataset, shape ``(n, p)``, with *n* samples and *p* variables.
    order : {3, 4}, default 4
        Maximum order of moments to compute.

    Returns
    -------
    dict
        Always present:

        - ``n`` : int -- sample count.
        - ``sum_x`` : ``(p,)`` -- Σ xᵢ.
        - ``sum_xx`` : ``(p, p)`` -- Σ xᵢ xⱼ (includes diagonal).
        - ``sum_x3`` : ``(p,)`` -- Σ xᵢ³.
        - ``sum_x1y2`` : ``(p, p)`` -- Σ xᵢ xⱼ² (``i != j``; diagonal is zero).
        - ``sum_x2y1`` : ``(p, p)`` -- Σ xᵢ² xⱼ (``i != j``; diagonal is zero).

        When ``order >= 4``, also:

        - ``sum_x4`` : ``(p,)`` -- Σ xᵢ⁴.
        - ``sum_x1y3`` : ``(p, p)`` -- Σ xᵢ xⱼ³.
        - ``sum_x3y1`` : ``(p, p)`` -- Σ xᵢ³ xⱼ.
        - ``sum_x2y2`` : ``(p, p)`` -- Σ xᵢ² xⱼ².
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape

    # --- 1st / 2nd order ---
    sum_x = np.sum(x, axis=0)  # (p,)
    sum_xx = x.T @ x  # (p, p)

    # --- Pre-compute element-wise powers ---
    x2 = x**2

    # --- 3rd order ---
    sum_x3 = np.sum(x2 * x, axis=0)  # (p,)  — xᵢ³ = xᵢ² · xᵢ
    sum_x1y2 = np.zeros((p, p))
    sum_x2y1 = np.zeros((p, p))

    for i in range(p):
        xi, xi2 = x[:, i], x2[:, i]
        for j in range(p):
            if i == j:
                continue
            xj2 = x2[:, j]
            sum_x1y2[i, j] = xi @ xj2  # Σ xᵢ xⱼ²
            sum_x2y1[i, j] = xi2 @ x[:, j]  # Σ xᵢ² xⱼ

    stats: RawStats = {
        "n": n,
        "sum_x": sum_x,
        "sum_xx": sum_xx,
        "sum_x3": sum_x3,
        "sum_x1y2": sum_x1y2,
        "sum_x2y1": sum_x2y1,
    }

    if order < 4:
        return stats

    # --- 4th order ---
    x3 = x2 * x
    sum_x4 = np.sum(x2**2, axis=0)  # (p,)
    sum_x1y3 = np.zeros((p, p))
    sum_x3y1 = np.zeros((p, p))
    sum_x2y2 = np.zeros((p, p))

    for i in range(p):
        xi, xi2, xi3 = x[:, i], x2[:, i], x3[:, i]
        for j in range(p):
            if i == j:
                continue
            xj, xj2, xj3 = x[:, j], x2[:, j], x3[:, j]
            sum_x1y3[i, j] = xi @ xj3  # Σ xᵢ xⱼ³
            sum_x3y1[i, j] = xi3 @ xj  # Σ xᵢ³ xⱼ
            sum_x2y2[i, j] = xi2 @ xj2  # Σ xᵢ² xⱼ²

    stats.update(
        {
            "sum_x4": sum_x4,
            "sum_x1y3": sum_x1y3,
            "sum_x3y1": sum_x3y1,
            "sum_x2y2": sum_x2y2,
        }
    )

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# §2  Aggregation & Global Cumulant Estimation (server-side)
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_and_compute_cumulants(client_stats: Sequence[RawStats]) -> Cumulants:
    """Aggregate raw moment sums from multiple clients into global cumulants.

    Parameters
    ----------
    client_stats : Sequence[dict]
        Each element is the output of :func:`compute_local_raw_stats` from one
        client. All clients must share the same variable set (i.e. horizontal
        FL).

    Returns
    -------
    dict
        - ``n`` : int -- total sample count.
        - ``cov`` : ``(p, p)`` -- sample covariance matrix.
        - ``c3`` : ``(p,)`` -- C₃(xᵢ).
        - ``c12`` : ``(p, p)`` -- C₁,₂(xᵢ, xⱼ) = Cum(xᵢ, xⱼ, xⱼ).
        - ``c21`` : ``(p, p)`` -- C₂,₁(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xⱼ).

        When 4th-order stats are available, also:

        - ``c4`` : ``(p,)`` -- C₄(xᵢ).
        - ``c13`` : ``(p, p)`` -- C₁,₃(xᵢ, xⱼ) = Cum(xᵢ, xⱼ, xⱼ, xⱼ).
        - ``c31`` : ``(p, p)`` -- C₃,₁(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xᵢ, xⱼ).
        - ``c22`` : ``(p, p)`` -- C₂,₂(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xⱼ, xⱼ).

        An empty dict when `client_stats` is empty.

    Notes
    -----
    Bias correction follows the k-statistic formulas of [2] in the module's
    References:

    - 3rd order: k₃ = N / ((N-1)(N-2)) · M₃.
    - 4th order: k₄ = [N(N+1) M₄ - 3(N-1) M₂²] / [(N-1)(N-2)(N-3)].

    where Mₖ denotes the sum of k-th centered powers.
    """
    if not client_stats:
        return {}

    p = len(client_stats[0]["sum_x"])
    has_4th = "sum_x4" in client_stats[0]

    # ── Step 1: Sum all raw statistics across clients ──────────────────
    totals = _aggregate_raw_stats(client_stats, p, has_4th)

    n = totals["n"]
    mu = totals["sum_x"] / n

    # ── Step 2: Sample covariance ──────────────────────────────────────
    cov = (totals["sum_xx"] - n * np.outer(mu, mu)) / max(n - 1, 1)

    # ── Step 3: 3rd-order cumulants ───────────────────────────────────
    c3, c12, c21 = _compute_3rd_order(totals, mu, n, p)

    result: Cumulants = {
        "n": n,
        "cov": cov,
        "c3": c3,
        "c12": c12,
        "c21": c21,
    }

    if not has_4th:
        return result

    # ── Step 4: 4th-order cumulants ───────────────────────────────────
    c4, c13, c31, c22 = _compute_4th_order(totals, mu, n, p)

    result.update({"c4": c4, "c13": c13, "c31": c31, "c22": c22})
    return result


# ── Private helpers for aggregation ───────────────────────────────────────


def _aggregate_raw_stats(stats_list: Sequence[RawStats], p: int, has_4th: bool) -> RawStats:
    """Sum raw moment arrays across all clients."""
    totals: RawStats = {
        "n": 0,
        "sum_x": np.zeros(p),
        "sum_xx": np.zeros((p, p)),
        "sum_x3": np.zeros(p),
        "sum_x1y2": np.zeros((p, p)),
        "sum_x2y1": np.zeros((p, p)),
    }

    if has_4th:
        totals.update(
            {
                "sum_x4": np.zeros(p),
                "sum_x1y3": np.zeros((p, p)),
                "sum_x3y1": np.zeros((p, p)),
                "sum_x2y2": np.zeros((p, p)),
            }
        )

    for s in stats_list:
        totals["n"] += s["n"]
        for key in totals:
            if key == "n":
                continue
            if key in s:
                totals[key] = totals[key] + s[key]

    return totals


def _compute_3rd_order(
    totals: RawStats, mu: npt.NDArray[np.float64], n: int, p: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute unbiased 3rd-order marginal and cross-cumulants.

    Centering identity used: Σ(x - μ)³ = S₃ - 3μ S₂ + 2N μ³.
    """
    if n < _MIN_SAMPLES_3RD:
        warnings.warn(
            f"Total sample size N={n} < {_MIN_SAMPLES_3RD}; 3rd-order cumulants are unreliable.",
            stacklevel=2,
        )
        return np.zeros(p), np.zeros((p, p)), np.zeros((p, p))

    factor3 = n / ((n - 1) * (n - 2))

    s_xx = totals["sum_xx"]
    s_x3 = totals["sum_x3"]
    s_x1y2 = totals["sum_x1y2"]
    s_x2y1 = totals["sum_x2y1"]

    # ── Marginal C₃(xᵢ) ──
    #   Σ(xᵢ − μᵢ)³ = S₃[i] − 3μᵢ·S₂[i] + 2N·μᵢ³
    sum_cen_x3 = s_x3 - 3 * mu * np.diag(s_xx) + 2 * n * (mu**3)
    c3 = factor3 * sum_cen_x3

    # ── Cross-cumulants C₁,₂ and C₂,₁ ──
    c12 = np.zeros((p, p))
    c21 = np.zeros((p, p))

    for i in range(p):
        c12[i, i] = c3[i]
        c21[i, i] = c3[i]
        for j in range(p):
            if i == j:
                continue

            mi, mj = mu[i], mu[j]

            # Σ(xᵢ−μᵢ)(xⱼ−μⱼ)²
            #   = S_x1y2[i,j] − 2μⱼ·S_xx[i,j] − μᵢ·S_xx[j,j] + 2N·μᵢ·μⱼ²
            m_12 = s_x1y2[i, j] - 2 * mj * s_xx[i, j] - mi * s_xx[j, j] + 2 * n * mi * mj**2
            c12[i, j] = factor3 * m_12

            # Σ(xᵢ−μᵢ)²(xⱼ−μⱼ)  — symmetric swap of roles
            #   = S_x2y1[i,j] − 2μᵢ·S_xx[i,j] − μⱼ·S_xx[i,i] + 2N·μⱼ·μᵢ²
            m_21 = s_x2y1[i, j] - 2 * mi * s_xx[i, j] - mj * s_xx[i, i] + 2 * n * mj * mi**2
            c21[i, j] = factor3 * m_21

    return c3, c12, c21


def _compute_4th_order(
    totals: RawStats, mu: npt.NDArray[np.float64], n: int, p: int
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute unbiased 4th-order marginal and cross-cumulants.

    k-statistic formula: k₄ = [N(N+1) M₄ - 3(N-1) M₂²] / [(N-1)(N-2)(N-3)].

    Centering identities used::

        Σ(x−μ)⁴ = S₄ − 4μ S₃ + 6μ² S₂ − 3Nμ⁴
        Σ(x−μₓ)(y−μᵧ)³ = S_{x¹y³} − μₓ S_{y³} − 3μᵧ S_{x¹y²}
                          + 3μₓμᵧ S_{y²} + 3μᵧ² S_{xy} − 3Nμₓμᵧ³
    """
    if n < _MIN_SAMPLES_4TH:
        warnings.warn(
            f"Total sample size N={n} < {_MIN_SAMPLES_4TH}; 4th-order cumulants are unreliable.",
            stacklevel=2,
        )
        z = np.zeros(p)
        zz = np.zeros((p, p))
        return z, zz.copy(), zz.copy(), zz.copy()

    d4 = (n - 1) * (n - 2) * (n - 3)
    alpha4 = n * (n + 1)
    beta4 = n - 1

    s_xx = totals["sum_xx"]
    s_x3 = totals["sum_x3"]
    s_x4 = totals["sum_x4"]
    s_x1y2 = totals["sum_x1y2"]
    s_x2y1 = totals["sum_x2y1"]
    s_x1y3 = totals["sum_x1y3"]
    s_x3y1 = totals["sum_x3y1"]
    s_x2y2 = totals["sum_x2y2"]

    # ── Centered 2nd-moment sums (reused in bias correction) ──
    #   M₂[i] = Σ(xᵢ − μᵢ)² = S_xx[i,i] − N μᵢ²
    m2_diag = np.diag(s_xx) - n * mu**2

    # ── Marginal C₄(xᵢ) ──
    #   M₄ = S₄ − 4μ S₃ + 6μ² S₂ − 3Nμ⁴
    m4_vec = s_x4 - 4 * mu * s_x3 + 6 * mu**2 * np.diag(s_xx) - 3 * n * mu**4
    c4 = (alpha4 * m4_vec - 3 * beta4 * m2_diag**2) / d4

    # ── Cross-cumulants C₁,₃ / C₃,₁ / C₂,₂ ──
    c13 = np.zeros((p, p))
    c31 = np.zeros((p, p))
    c22 = np.zeros((p, p))

    for i in range(p):
        # Diagonal: all cross-cumulants collapse to the marginal
        c13[i, i] = c4[i]
        c31[i, i] = c4[i]
        c22[i, i] = c4[i]

        for j in range(p):
            if i == j:
                continue

            mi, mj = mu[i], mu[j]

            # --- M₁₃ = Σ(xᵢ − μᵢ)(xⱼ − μⱼ)³ ---
            #   = S_{x¹y³} − μᵢ·S_{y³} − 3μⱼ·S_{x¹y²}
            #     + 3μᵢμⱼ·S_{y²} + 3μⱼ²·S_{xy} − 3N·μᵢ·μⱼ³
            m13 = (
                s_x1y3[i, j]
                - mi * s_x3[j]
                - 3 * mj * s_x1y2[i, j]
                + 3 * mi * mj * s_xx[j, j]
                + 3 * mj**2 * s_xx[i, j]
                - 3 * n * mi * mj**3  # ← μⱼ³ (not μⱼ²)
            )

            # --- M₃₁ = Σ(xᵢ − μᵢ)³(xⱼ − μⱼ)  [swap i ↔ j roles in M₁₃] ---
            #   = S_{x³y¹} − μⱼ·S_{x³} − 3μᵢ·S_{x²y¹}
            #     + 3μⱼμᵢ·S_{x²} + 3μᵢ²·S_{xy} − 3N·μⱼ·μᵢ³
            m31 = (
                s_x3y1[i, j]
                - mj * s_x3[i]
                - 3 * mi * s_x2y1[i, j]
                + 3 * mj * mi * s_xx[i, i]
                + 3 * mi**2 * s_xx[i, j]
                - 3 * n * mj * mi**3  # ← μᵢ³ (not μᵢ²)
            )

            # --- M₂₂ = Σ(xᵢ − μᵢ)²(xⱼ − μⱼ)² ---
            #   = S_{x²y²} − 2μⱼ·S_{x²y¹} − 2μᵢ·S_{x¹y²}
            #     + μⱼ²·S_{x²} + μᵢ²·S_{y²} + 4μᵢμⱼ·S_{xy} − 3N·μᵢ²μⱼ²
            m22 = (
                s_x2y2[i, j]
                - 2 * mj * s_x2y1[i, j]
                - 2 * mi * s_x1y2[i, j]
                + mj**2 * s_xx[i, i]
                + mi**2 * s_xx[j, j]
                + 4 * mi * mj * s_xx[i, j]
                - 3 * n * mi**2 * mj**2
            )

            # --- Centered 2nd-moment cross-sum ---
            #   M₁₁ = Σ(xᵢ − μᵢ)(xⱼ − μⱼ) = S_xx[i,j] − N·μᵢμⱼ
            m11 = s_xx[i, j] - n * mi * mj
            m20 = m2_diag[i]
            m02 = m2_diag[j]

            # --- Unbiased k-statistics ---
            c13[i, j] = (alpha4 * m13 - 3 * beta4 * m11 * m02) / d4
            c31[i, j] = (alpha4 * m31 - 3 * beta4 * m11 * m20) / d4
            c22[i, j] = (alpha4 * m22 - beta4 * (m20 * m02 + 2 * m11**2)) / d4

    return c4, c13, c31, c22


# ═══════════════════════════════════════════════════════════════════════════
# §3  Convenience Wrappers (single-vector API)
# ═══════════════════════════════════════════════════════════════════════════

# ── Single-vector estimators ──────────────────────────────────────────────


def unbiased_cumulant3(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | None = None,
    z: npt.NDArray[np.float64] | None = None,
) -> float:
    """Return the unbiased 3rd-order cumulant estimator.

    - ``unbiased_cumulant3(x)`` returns C₃(x) = Cum(x, x, x).
    - ``unbiased_cumulant3(x, y, z)`` returns Cum(x, y, z).

    Parameters
    ----------
    x : numpy.ndarray
        First vector, shape ``(n,)``.
    y : numpy.ndarray, optional
        Second vector, shape ``(n,)``. Defaults to `x`.
    z : numpy.ndarray, optional
        Third vector, shape ``(n,)``. Defaults to `y`.

    Returns
    -------
    float
        The k-statistic k₃ = N / ((N-1)(N-2)) · Σ(xᵢ - μₓ)(yᵢ - μᵧ)(zᵢ - μ_z).

    Raises
    ------
    ValueError
        If fewer than `_MIN_SAMPLES_3RD` samples are given.
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        y = z = x
    else:
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float) if z is not None else y

    n = len(x)
    if n < _MIN_SAMPLES_3RD:
        raise ValueError(f"Need at least {_MIN_SAMPLES_3RD} samples; got {n}.")

    xc = x - x.mean()
    yc = y - y.mean()
    zc = z - z.mean()

    factor = n / ((n - 1) * (n - 2))
    return float(factor * np.sum(xc * yc * zc))


def unbiased_cumulant4(x: npt.NDArray[np.float64]) -> float:
    """Return the unbiased 4th-order marginal cumulant.

    This is the excess-kurtosis numerator.

    Parameters
    ----------
    x : numpy.ndarray
        Vector, shape ``(n,)``.

    Returns
    -------
    float
        k₄ = [N(N+1) M₄ - 3(N-1) M₂²] / [(N-1)(N-2)(N-3)], where M₄ = Σ(xᵢ -
        x̄)⁴ and M₂ = Σ(xᵢ - x̄)².

    Raises
    ------
    ValueError
        If fewer than `_MIN_SAMPLES_4TH` samples are given.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < _MIN_SAMPLES_4TH:
        raise ValueError(f"Need at least {_MIN_SAMPLES_4TH} samples; got {n}.")

    xc = x - x.mean()
    m2 = np.sum(xc**2)
    m4 = np.sum(xc**4)

    d = (n - 1) * (n - 2) * (n - 3)
    return float((n * (n + 1) * m4 - 3 * (n - 1) * m2**2) / d)


# ── Pairwise cross-cumulant estimators ────────────────────────────────────


def cross_cumulants3(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return the unbiased 3rd-order cross-cumulants between `x` and `y`.

    Parameters
    ----------
    x : numpy.ndarray
        First vector, shape ``(n,)``.
    y : numpy.ndarray
        Second vector, shape ``(n,)``.

    Returns
    -------
    tuple of float
        ``(c12, c21)`` where c12 = Cum(x, y, y) and c21 = Cum(x, x, y).

    Raises
    ------
    ValueError
        If fewer than `_MIN_SAMPLES_3RD` samples are given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < _MIN_SAMPLES_3RD:
        raise ValueError(f"Need at least {_MIN_SAMPLES_3RD} samples; got {n}.")

    xc = x - x.mean()
    yc = y - y.mean()
    factor = n / ((n - 1) * (n - 2))

    c12 = float(factor * np.sum(xc * yc**2))
    c21 = float(factor * np.sum(xc**2 * yc))
    return c12, c21


def cross_cumulants4(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    """Return the unbiased 4th-order cross-cumulants between `x` and `y`.

    Parameters
    ----------
    x : numpy.ndarray
        First vector, shape ``(n,)``.
    y : numpy.ndarray
        Second vector, shape ``(n,)``.

    Returns
    -------
    tuple of float
        ``(c13, c31, c22)`` where c13 = Cum(x, y, y, y), c31 = Cum(x, x, x, y),
        and c22 = Cum(x, x, y, y).

    Raises
    ------
    ValueError
        If fewer than `_MIN_SAMPLES_4TH` samples are given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < _MIN_SAMPLES_4TH:
        raise ValueError(f"Need at least {_MIN_SAMPLES_4TH} samples; got {n}.")

    xc = x - x.mean()
    yc = y - y.mean()

    s20 = np.sum(xc**2)
    s02 = np.sum(yc**2)
    s11 = np.sum(xc * yc)

    s13 = np.sum(xc * yc**3)
    s31 = np.sum(xc**3 * yc)
    s22 = np.sum(xc**2 * yc**2)

    d = (n - 1) * (n - 2) * (n - 3)
    alpha = n * (n + 1)
    beta = n - 1

    k13 = (alpha * s13 - 3 * beta * s11 * s02) / d
    k31 = (alpha * s31 - 3 * beta * s11 * s20) / d
    k22 = (alpha * s22 - beta * (s20 * s02 + 2 * s11**2)) / d

    return float(k13), float(k31), float(k22)
