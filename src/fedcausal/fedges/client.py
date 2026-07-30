"""A single federated participant.

Holds local data, a local learner, and a client-side fusion operator. Each round
it rebuilds its local model (warm-started from the previous one) and, after the
server broadcasts the global model, fuses local + global and optionally refines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.exceptions import FedCausalError
from fedcausal.fedges.model import BNModel

if TYPE_CHECKING:
    from fedcausal.data import DiscreteDataset
    from fedcausal.fedges.learner import LocalLearner
    from fedcausal.fusion import FusionOperator

__all__ = ["Client"]


class Client:
    """A single federated participant.

    Parameters
    ----------
    local_fusion : FusionOperator
        Fuses the client's local model with the broadcast global model.
    learner : LocalLearner
        Builds and refines the local model.
    data : DiscreteDataset
        The client's local data partition.
    initial_model : BNModel or None, optional
        Optional starting local model (default ``None`` -> GES from empty).

    Attributes
    ----------
    id : int
        Client index, assigned by the server (``-1`` until then).
    iteration : int
        Number of times :meth:`build_local_model` has run.
    local_model : BNModel or None
        The client's current local model.
    """

    def __init__(
        self,
        local_fusion: FusionOperator,
        learner: LocalLearner,
        data: DiscreteDataset,
        initial_model: BNModel | None = None,
    ) -> None:
        self.local_fusion = local_fusion
        self.learner = learner
        self.data = data
        self.local_model = initial_model
        self.id = -1
        self.iteration = 0

    def build_local_model(self) -> BNModel:
        """Rebuild the local model, warm-started from the current one, and return it."""
        self.iteration += 1
        self.local_model = self.learner.build_local_model(self.local_model, self.data)
        return self.local_model

    def fuse(self, global_model: BNModel) -> BNModel:
        """Fuse local + global, then refine, updating and returning the local model.

        Raises
        ------
        FedCausalError
            If called before :meth:`build_local_model`.
        """
        old = self.local_model
        if old is None:
            raise FedCausalError("fuse() called before build_local_model()")
        fused_dag = self.local_fusion.fuse([old.dag, global_model.dag])
        fused = BNModel(fused_dag, self.learner.score_dag(fused_dag, self.data))
        self.local_model = self.learner.refine_local_model(old, fused, self.data)
        return self.local_model
