"""The FedGES server loop, and :func:`run_fedges`, which runs a whole federation.

Each round every site rebuilds its local model, the server checks convergence,
fuses the local models, and sends the result back for each site to fuse in.
The server holds no data: scores come from the sites, and metrics against a
true network go through an optional observer. The Flower transport runs the
same sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.data import contiguous_split
from fedcausal.exceptions import ConfigurationError, ConvergedBeforeFusionError
from fedcausal.fedges.client import Client
from fedcausal.fedges.defaults import (
    default_client_fusion,
    default_convergence,
    default_server_fusion,
)
from fedcausal.fedges.learner import GESLearner, Refinement
from fedcausal.fedges.model import BNModel
from fedcausal.fedges.observer import MetricsObserver

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.data import DiscreteDataset
    from fedcausal.fedges.convergence import Convergence
    from fedcausal.fedges.observer import RoundObserver, RoundStats
    from fedcausal.fusion import FusionOperator
    from fedcausal.graph import DAG

__all__ = ["Server", "run_fedges"]


class Server:
    """The federated coordinator.

    Parameters
    ----------
    global_fusion : FusionOperator
        Server-side fusion of the clients' local models.
    convergence : Convergence
        Round-loop stopping criterion.
    clients : Sequence[Client]
        Participating clients; assigned ids ``0..n-1`` in order.
    n_iterations : int, optional
        Maximum number of rounds. Default 5.
    initial_global : BNModel or None, optional
        Global model to start from.

    Attributes
    ----------
    global_model : BNModel or None
        The fused global model after :meth:`run`.

    Raises
    ------
    ConfigurationError
        If `clients` is empty or `n_iterations` is negative.
    """

    def __init__(
        self,
        global_fusion: FusionOperator,
        convergence: Convergence,
        clients: Sequence[Client],
        n_iterations: int = 5,
        initial_global: BNModel | None = None,
    ) -> None:
        if not clients:
            raise ConfigurationError("a federation needs at least one client")
        if n_iterations < 0:
            raise ConfigurationError(f"n_iterations must be non-negative, got {n_iterations}")
        self.global_fusion = global_fusion
        self.convergence = convergence
        self.clients = list(clients)
        self.n_iterations = n_iterations
        self.global_model = initial_global
        for i, client in enumerate(self.clients):
            client.id = i

    def run(self, observer: RoundObserver | None = None) -> BNModel | None:
        """Run the round loop, emitting events to `observer`; return the global model.

        Returns ``None`` if the loop stopped before any fusion, which happens
        only with a zero iteration budget or a convergence criterion that fires
        on its first sighting.
        """
        for iteration in range(1, self.n_iterations + 1):
            local_models = []
            for client in self.clients:
                local_models.append(client.build_local_model())
                if observer is not None:
                    observer.on_round("build", iteration, client.id, local_models[-1])

            if self.convergence.check_convergence(local_models):
                break

            fused_dag = self.global_fusion.fuse([m.dag for m in local_models])
            self.global_model = BNModel(fused_dag)  # no data here, so no score
            if observer is not None:
                observer.on_round("server", iteration, -1, self.global_model)

            for client in self.clients:
                client.fuse(self.global_model)
        return self.global_model


def run_fedges(
    data: DiscreteDataset,
    n_clients: int,
    n_iterations: int = 5,
    client_fusion: FusionOperator | None = None,
    server_fusion: FusionOperator | None = None,
    convergence: Convergence | None = None,
    refinement: Refinement = "None",
    truth: DAG | None = None,
) -> tuple[BNModel, list[RoundStats]]:
    """Run FedGES in one process on `data`, split into contiguous blocks, one per site.

    Parameters
    ----------
    data : DiscreteDataset
        Full dataset to partition across clients.
    n_clients : int
        Number of clients.
    n_iterations : int, optional
        Maximum number of rounds. Default 5.
    client_fusion, server_fusion : FusionOperator or None, optional
        Fusion operators. Default ``ConsensusUnion()`` and ``Consensus(50)``.
    convergence : Convergence or None, optional
        Stopping criterion. Default ``ScoreImprovement or ModelEquality``.
    refinement : {"None", "FES", "BES", "GES"}, optional
        Client refinement stage passed to :class:`GESLearner`.
    truth : DAG or None, optional
        True DAG. When given, the returned rows carry SHD and SMHD.

    Returns
    -------
    tuple[BNModel, list[RoundStats]]
        The final global model and one stats row per reported model.

    Raises
    ------
    ValueError
        If `n_clients` is not positive, `n_iterations` is negative, or
        `refinement` is unknown.
    ConvergedBeforeFusionError
        If the loop stopped before producing any global model.
    """
    parts = contiguous_split(data, n_clients)
    fusion = client_fusion or default_client_fusion()
    # One learner per client, so each keeps the score cache for its own data.
    clients = [Client(fusion, GESLearner(refinement=refinement), p) for p in parts]
    server = Server(
        server_fusion or default_server_fusion(),
        convergence or default_convergence(),
        clients,
        n_iterations=n_iterations,
    )
    observer = MetricsObserver(truth)
    global_model = server.run(observer)
    if global_model is None:
        reason = (
            "n_iterations is 0"
            if n_iterations == 0
            else "the convergence criterion fired on the first round"
        )
        raise ConvergedBeforeFusionError(f"no global model was fused: {reason}")
    return global_model, observer.rows
