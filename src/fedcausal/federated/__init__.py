"""In-process (single-process) federated structure-learning engine.

Clients learn locally and a coordinator fuses their structures:
clients learn locally with GES, the server fuses their DAGs with a structural
:class:`~fedcausal.fusion.FusionOperator`, and convergence criteria stop the
rounds. See :func:`run_fedges` for the one-call entry point.
"""

from fedcausal.federated.client import Client
from fedcausal.federated.convergence import (
    Convergence,
    IterationEquality,
    ModelEquality,
    MultipleConvergence,
    NoneConvergence,
    ScoreImprovement,
)
from fedcausal.federated.defaults import (
    default_client_fusion,
    default_convergence,
    default_server_fusion,
)
from fedcausal.federated.learner import GESLearner, LocalLearner, Refinement
from fedcausal.federated.model import BNModel
from fedcausal.federated.observer import (
    MetricsObserver,
    ModelCollector,
    RoundObserver,
    RoundStats,
)
from fedcausal.federated.server import Server, run_fedges

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
