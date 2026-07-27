"""Default federation wiring, shared by every transport.

The in-process engine and the Flower app must agree on what "the defaults" are,
or the two stop being comparable. They both build their defaults here rather
than each spelling them out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.federated.convergence import (
    ModelEquality,
    MultipleConvergence,
    ScoreImprovement,
)
from fedcausal.fusion import Consensus, ConsensusUnion

if TYPE_CHECKING:
    from fedcausal.federated.convergence import Convergence
    from fedcausal.fusion import FusionOperator

__all__ = ["default_client_fusion", "default_server_fusion", "default_convergence"]


def default_client_fusion() -> FusionOperator:
    """Return the default client-side fusion: the full consensus union."""
    return ConsensusUnion()


def default_server_fusion() -> FusionOperator:
    """Return the default server-side fusion: 50% cross-client agreement."""
    return Consensus(50)


def default_convergence() -> Convergence:
    """Return the default stopping rule: no score improvement, or a repeated model."""
    return MultipleConvergence([ScoreImprovement(), ModelEquality()])
