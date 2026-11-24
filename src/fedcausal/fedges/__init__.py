"""FedGES: federated structure learning over discrete data.

Torrijos, Gámez & Puerta, "FedGES: A Federated Learning Approach for Bayesian
Network Structure Learning", *Machine Learning* 115, 9 (2026).

:class:`FedGES` runs a whole federation in one machine, optionally across
processes. The pieces it is built from, :class:`Site` and :func:`run_rounds`,
are what the Flower transport in :mod:`fedcausal.flower` runs across machines.
"""

from fedcausal.fedges.convergence import (
    AnyOf,
    Convergence,
    ModelEquality,
    ScoreImprovement,
    default_convergence,
)
from fedcausal.fedges.estimator import FedGES
from fedcausal.fedges.model import BNModel
from fedcausal.fedges.rounds import SitePool, run_rounds
from fedcausal.fedges.site import Refinement, Search, Site

__all__ = [
    "AnyOf",
    "BNModel",
    "Convergence",
    "FedGES",
    "ModelEquality",
    "Refinement",
    "ScoreImprovement",
    "Search",
    "Site",
    "SitePool",
    "default_convergence",
    "run_rounds",
]
