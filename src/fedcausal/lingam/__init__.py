"""Federated causal discovery over continuous data, from higher-order cumulants.

Each site sends moment sums; the server turns them into cumulants and a tau
discriminator recovers the causal order. Every solver is a pure function of
those statistics and needs only numpy and scipy. The Flower transport is
:mod:`fedcausal.flower.lingam`.

References
----------
[1] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
[2] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
"""

from __future__ import annotations

from fedcausal.lingam.cumulants import (
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
)
from fedcausal.lingam.hc import hc, hc_lingam
from fedcausal.lingam.ishc import fedhc, fedishc
from fedcausal.lingam.rcd import (
    RCDResult,
    fedrcd,
    fedrcd_exact,
    fedrcd_noniterative,
    ishc_beta,
)

__all__ = [
    "RCDResult",
    "aggregate_and_compute_cumulants",
    "compute_local_raw_stats",
    "fedhc",
    "fedishc",
    "fedrcd",
    "fedrcd_exact",
    "fedrcd_noniterative",
    "hc",
    "hc_lingam",
    "ishc_beta",
]
