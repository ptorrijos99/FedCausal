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

The solvers are causalbase's (:mod:`causalbase.lingam`, where
:class:`causalbase.lingam.RCD` runs them on one table); :func:`fedrcd` runs
them on the cumulants the sites' sums add up to, and :func:`fedrcd_exact`
deflates each site's data in turn.

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

from typing import TYPE_CHECKING, Literal

from causalbase.lingam.solvers import (
    LambdaSelection,
    Pruning,
    RCDResult,
    _exact_from_blocks,
    exact_order,
    rcd,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from fedcausal.lingam.cumulants import Cumulants

__all__ = ["LambdaSelection", "Pruning", "RCDResult", "exact_order", "fedrcd", "fedrcd_exact"]


def fedrcd(
    stats: Cumulants,
    *,
    method: Literal["deflation", "ratio", "sort"] = "deflation",
    order: Literal[3, 4] | None = None,
    pruning: Pruning = "adaptive_lasso",
    lambda_selection: LambdaSelection = "bic",
) -> RCDResult:
    """Find the causal order from aggregated cumulants, then estimate the coefficients.

    The server's whole computation in the one-round methods: it needs only
    the statistics, never a row of data. The same as
    :func:`causalbase.lingam.rcd`.

    Parameters
    ----------
    stats : Cumulants
        Output of :func:`~fedcausal.lingam.cumulants_from_sums`.
    method : {"deflation", "ratio", "sort"}, optional
        What follows each source; see the module docstring.
    order : {3, 4} or None, optional
        Which tau discriminator to use. Defaults to 4 when `stats` has
        fourth-order cumulants, else 3.
    pruning : {"adaptive_lasso", None}, default="adaptive_lasso"
        How the coefficients are estimated on the order; ``None`` is OLS,
        which keeps every predecessor as a parent.
    lambda_selection : {"bic", "fixed"}, default="bic"
        Penalty selection for the adaptive lasso.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Raises
    ------
    ValueError
        If `method`, `order`, `pruning` or `lambda_selection` is not one of the
        allowed values, or ``order=4`` is asked of third-order statistics.
    """
    return rcd(
        stats, method=method, order=order, pruning=pruning, lambda_selection=lambda_selection
    )


def fedrcd_exact(
    sites: Sequence[npt.ArrayLike],
    *,
    order: Literal[3, 4] = 4,
    pruning: Pruning = "adaptive_lasso",
    lambda_selection: LambdaSelection = "bic",
) -> RCDResult:
    """Find the causal order by deflating each site's data, one round per source.

    Each round the sites send the moment sums of their current data, the
    server picks the source by tau and the OLS coefficients on it, and every
    site deflates its data by them (``x_j -= beta_j * x_s``) and drops the
    source column. Runs the whole protocol in one process;
    :func:`fedcausal.flower.lingam.run_server_loop` with ``method="exact"`` is the same over
    Flower. With a single site this is HC (Chen et al., 2025), which is
    :func:`causalbase.lingam.rcd_exact`.

    Parameters
    ----------
    sites : sequence of array-like
        Each site's data, shape ``(n_k, p)``.
    order : {3, 4}, optional
        Which tau discriminator to use, and so which moment sums the sites send.
    pruning, lambda_selection
        As in :func:`fedrcd`.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Notes
    -----
    ``p - 1`` rounds of O(p^2) statistics each.
    """
    return _exact_from_blocks(sites, order, pruning, lambda_selection)
