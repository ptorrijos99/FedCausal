"""Federated causal discovery over continuous data, from higher-order cumulants.

Each site sends moment sums; the server turns them into cumulants and a tau
discriminator recovers the causal order. Every solver is a pure function of
those statistics, and the solvers are causalbase's (:mod:`causalbase.lingam`),
whose :class:`~causalbase.lingam.RCD` runs them on data held in one place.
The Flower transport is :mod:`fedcausal.flower.lingam`.

The published baselines are settings of the same solvers:

- FedRCD: ``fedrcd(stats)``, one round.
- FedRCD-Exact: ``fedrcd_exact(sites)``, ``p - 1`` rounds.
- FedISHC: ``fedrcd(stats, method="ratio", order=3)``, one round.
- FedHC: ``fedrcd(stats, method="sort", order=3)``, one round.
- HC-LiNGAM: ``fedrcd(stats, method="sort")`` over a single site, which is
  ``causalbase.lingam.RCD(method="sort")``.
- HC: ``fedrcd_exact([x])``, which is ``causalbase.lingam.RCD(method="exact")``.

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

from fedcausal.lingam.cumulants import Cumulants, MomentSums, cumulants_from_sums, moment_sums
from fedcausal.lingam.estimator import FedRCD
from fedcausal.lingam.rcd import LambdaSelection, Pruning, RCDResult, fedrcd, fedrcd_exact

__all__ = [
    "Cumulants",
    "FedRCD",
    "LambdaSelection",
    "MomentSums",
    "Pruning",
    "RCDResult",
    "cumulants_from_sums",
    "fedrcd",
    "fedrcd_exact",
    "moment_sums",
]
