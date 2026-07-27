"""Drive a FedGES federation in-process through Flower's simulation engine.

Convenient for notebooks and tests: it partitions one dataset, spins up that
many supernodes, runs the loop, and hands back the final global model.

Flower's simulation engine runs on Ray, which is an extra install and is not
available for every Python version and platform. The deployment path in
:mod:`fedcausal.flower.app` needs none of that, so this module is a
convenience, not the main entry point -- see ``docs/flower-deployment.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.exceptions import ConfigurationError
from fedcausal.federated.defaults import (
    default_client_fusion,
    default_convergence,
    default_server_fusion,
)
from fedcausal.federated.observer import ModelCollector
from fedcausal.flower.app import build_client_app, build_server_app
from fedcausal.flower.data import InMemoryPartitions

if TYPE_CHECKING:
    from fedcausal.data import DiscreteDataset
    from fedcausal.federated.convergence import Convergence
    from fedcausal.federated.learner import Refinement
    from fedcausal.federated.model import BNModel
    from fedcausal.federated.observer import RoundObserver
    from fedcausal.fusion import FusionOperator

__all__ = ["run_flower_fedges"]


def run_flower_fedges(
    data: DiscreteDataset,
    n_clients: int,
    n_iterations: int = 5,
    client_fusion: FusionOperator | None = None,
    server_fusion: FusionOperator | None = None,
    convergence: Convergence | None = None,
    refinement: Refinement = "None",
    observer: RoundObserver | None = None,
) -> BNModel | None:
    """Run FedGES over a Flower simulation and return the final global model.

    Mirrors :func:`fedcausal.federated.run_fedges` over the Flower transport:
    splits `data` into `n_clients` contiguous partitions and drives the same
    round loop. Both read their defaults from
    :mod:`fedcausal.federated.defaults`, so the two transports cannot drift.

    Parameters
    ----------
    data : DiscreteDataset
        Full dataset to partition across clients.
    n_clients : int
        Number of clients and supernodes.
    n_iterations : int, optional
        Maximum number of rounds (default 5).
    client_fusion, server_fusion : FusionOperator or None, optional
        Fusion operators; default to ``ConsensusUnion()`` / ``Consensus(50)``.
    convergence : Convergence or None, optional
        Stopping criterion; defaults to ``ScoreImprovement or ModelEquality``.
    refinement : {"None", "FES", "BES", "GES"}, optional
        Client refinement stage.
    observer : RoundObserver or None, optional
        Receives build and server events from the server side.

    Returns
    -------
    BNModel or None
        The final global model, or `None` if the run converged before any
        fusion.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed.
    """
    try:
        from flwr.simulation import run_simulation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            "Flower's simulation engine is not installed. Install it with "
            "`pip install 'fedcausal[flower-simulation]'`, or deploy against a "
            "real SuperLink instead; see docs/flower-deployment.md."
        ) from exc

    from fedcausal.data import contiguous_split

    parts = contiguous_split(data, n_clients)
    collector = ModelCollector()
    fanout = _Fanout([collector] + ([observer] if observer is not None else []))

    client_app = build_client_app(
        InMemoryPartitions(parts),
        client_fusion or default_client_fusion(),
        refinement,
    )
    server_app = build_server_app(
        server_fusion or default_server_fusion(),
        convergence or default_convergence(),
        n_clients,
        data.n_vars,
        n_iterations,
        fanout,
    )
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=n_clients)
    return collector.globals[-1] if collector.globals else None


class _Fanout:
    """Dispatch each round event to several observers."""

    def __init__(self, observers: list[RoundObserver]) -> None:
        self._observers = list(observers)

    def on_round(self, stage: str, iteration: int, client_id: int, model: BNModel) -> None:
        """Forward one round event to every registered observer."""
        for obs in self._observers:
            obs.on_round(stage, iteration, client_id, model)
