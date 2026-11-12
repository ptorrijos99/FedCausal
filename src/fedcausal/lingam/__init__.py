"""Federated causal discovery over continuous data, from higher-order cumulants.

Each site sends moment sums; the server turns them into cumulants and a tau
discriminator recovers the causal order. Every solver is a pure function of
those statistics and needs only numpy. The Flower transport is
:mod:`fedcausal.flower.lingam`.

The published baselines are settings of the same solvers:

- FedRCD: ``fedrcd(stats)``, one round.
- FedRCD-Exact: ``fedrcd_exact(sites)``, ``p - 1`` rounds.
- FedISHC: ``fedrcd(stats, method="ratio", order=3)``, one round.
- FedHC: ``fedrcd(stats, method="sort", order=3)``, one round.
- HC-LiNGAM: ``fedrcd(stats, method="sort")`` over a single site.
- HC: ``fedrcd_exact([x])``.

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

from fedcausal.lingam.cumulants import (
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
)
from fedcausal.lingam.estimator import FedRCD
from fedcausal.lingam.rcd import RCDResult, fedrcd, fedrcd_exact

__all__ = [
    "FedRCD",
    "RCDResult",
    "aggregate_and_compute_cumulants",
    "compute_local_raw_stats",
    "fedrcd",
    "fedrcd_exact",
]
