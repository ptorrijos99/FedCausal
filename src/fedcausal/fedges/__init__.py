"""FedGES: federated Bayesian network structure learning over discrete data.

Every client learns a local structure with GES, the server fuses their DAGs
with a structural :class:`~fedcausal.fusion.FusionOperator` and broadcasts the
result, and convergence criteria stop the rounds. :func:`run_fedges` runs the
whole thing in one process; :mod:`fedcausal.flower.fedges` runs the same loop
over a real federation.

Torrijos, Gámez & Puerta, "FedGES: A Federated Learning Approach for Bayesian
Network Structure Learning", *Machine Learning* 115, 9 (2026).
"""

from fedcausal.fedges.client import Client
from fedcausal.fedges.convergence import (
    Convergence,
    IterationEquality,
    ModelEquality,
    MultipleConvergence,
    NoneConvergence,
    ScoreImprovement,
)
from fedcausal.fedges.defaults import (
    default_client_fusion,
    default_convergence,
    default_server_fusion,
)
from fedcausal.fedges.learner import GESLearner, LocalLearner, Refinement
from fedcausal.fedges.model import BNModel
from fedcausal.fedges.observer import (
    MetricsObserver,
    ModelCollector,
    RoundObserver,
    RoundStats,
)
from fedcausal.fedges.server import Server, run_fedges

__all__ = [
    "BNModel",
    "Client",
    "Server",
    "RoundStats",
    "RoundObserver",
    "MetricsObserver",
    "ModelCollector",
    "run_fedges",
    "GESLearner",
    "LocalLearner",
    "Refinement",
    "Convergence",
    "NoneConvergence",
    "ScoreImprovement",
    "ModelEquality",
    "IterationEquality",
    "MultipleConvergence",
    "default_client_fusion",
    "default_server_fusion",
    "default_convergence",
]
