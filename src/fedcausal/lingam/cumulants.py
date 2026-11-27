"""Unbiased third- and fourth-order cumulants, computable across sites.

Each site reduces its rows to moment sums about its own mean
(:func:`moment_sums`); the server merges them exactly into sums about the
pooled mean and applies the k-statistic bias corrections
(:func:`cumulants_from_sums`). The result equals what the pooled data would
give. Both functions are causalbase's (:mod:`causalbase.lingam`), which uses
them on blocks of one table.

References
----------
[1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
[2] Schefczik, N. and Hägele, M. "Ready-to-use unbiased estimators for
    multivariate cumulants including one that outperforms x^3."
    arXiv:1904.12154, 2019.
[3] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
"""

from __future__ import annotations

from causalbase.lingam.cumulants import Cumulants, MomentSums, cumulants_from_sums, moment_sums

__all__ = [
    "Cumulants",
    "MomentSums",
    "cumulants_from_sums",
    "moment_sums",
]
