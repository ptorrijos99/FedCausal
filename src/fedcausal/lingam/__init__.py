"""Federated causal discovery over continuous data via higher-order cumulants.

This subpackage implements the FedRCD family and its baselines. Where the
Bayesian-network side of ``fedcausal`` learns by fusing *models*, these methods
aggregate *sufficient statistics*: each client sends raw moment sums, the server
turns them into unbiased cumulant estimates, and a causal order falls out of the
tau discriminator. Raw data never leaves a client in either design.

The Flower transport lives in :mod:`fedcausal.lingam.flower` and is imported
only on demand, so this package works without ``flwr`` installed.

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
    "fedrcd",
    "fedrcd_exact",
    "fedrcd_noniterative",
    "ishc_beta",
]
