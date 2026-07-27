"""The federated server loop, and a one-call runner for the whole federation.

Each round every client rebuilds its local model, the server checks convergence,
fuses the local models into a global model, and broadcasts it back for each
client to fuse locally.

The loop is transport-agnostic and holds no client data: the model score is a
client output (carried on :class:`~fedcausal.federated.model.BNModel`), and
structural metrics against a ground-truth network are emitted through an optional
:class:`~fedcausal.federated.observer.RoundObserver` rather than computed inline.
This is what lets the Flower ``ServerApp`` reuse the exact same sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.data import contiguous_split
from fedcausal.exceptions import ConfigurationError, ConvergedBeforeFusionError
from fedcausal.federated.client import Client
from fedcausal.federated.defaults import (
    default_client_fusion,
    default_convergence,
    default_server_fusion,
)
from fedcausal.federated.learner import GESLearner, Refinement
from fedcausal.federated.model import BNModel
from fedcausal.federated.observer import MetricsObserver

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.data import DiscreteDataset
    from fedcausal.federated.convergence import Convergence
    from fedcausal.federated.observer import RoundObserver, RoundStats
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
        Maximum number of rounds (default 5).
    initial_global : BNModel or None, optional
        Optional starting global model.

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
            self.global_model = BNModel(fused_dag)  # server holds no data -> no score
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
    """Wire and run FedGES end-to-end on a single dataset.

    Splits `data` into `n_clients` contiguous partitions
    (:func:`~fedcausal.data.contiguous_split`) and runs the default wiring
    unless overridden; see :mod:`fedcausal.federated.defaults`.

    Parameters
    ----------
    data : DiscreteDataset
        Full dataset to partition across clients.
    n_clients : int
        Number of clients.
    n_iterations : int, optional
        Maximum number of rounds (default 5).
    client_fusion, server_fusion : FusionOperator or None, optional
        Fusion operators; default to ``ConsensusUnion()`` / ``Consensus(50)``.
    convergence : Convergence or None, optional
        Stopping criterion; defaults to ``ScoreImprovement or ModelEquality``.
    refinement : {"None", "FES", "BES", "GES"}, optional
        Client refinement stage passed to :class:`GESLearner`.
    truth : DAG or None, optional
        Ground-truth DAG; when given, the returned rows carry SHD/SMHD.

    Returns
    -------
    tuple[BNModel, list[RoundStats]]
        The final global model and the per-round stats collected by an
        internal :class:`~fedcausal.federated.observer.MetricsObserver`.

    Raises
    ------
    ConfigurationError
        If `n_clients` or `n_iterations` is invalid, or `refinement` is not a
        recognised stage.
    ConvergedBeforeFusionError
        If the loop stopped before producing any global model.
    """
    parts = contiguous_split(data, n_clients)
    fusion = client_fusion or default_client_fusion()
    # One learner per client: each keeps its own memoised score for its own
    # partition, which a shared instance would thrash on every client switch.
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
        raise ConvergedBeforeFusionError(
            f"the loop stopped after {n_iterations} iteration(s) without fusing a "
            "global model; the convergence criterion fired on the first round"
        )
    return global_model, observer.rows
