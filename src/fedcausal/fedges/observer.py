"""Round observation for the federated loop, decoupled from transport and truth.

The loop core emits one event per built local model and per server fusion; an
optional observer turns those into whatever the caller needs. ``MetricsObserver``
is the study's observer: it records structural metrics against a ground-truth
DAG. A real federation attaches no observer (or a logging one) and never needs
the truth network -- SHD/SMHD are graph-vs-graph, and the model's BDeu score is
carried by the client that computed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fedcausal.metrics import shd, smhd

if TYPE_CHECKING:
    from fedcausal.fedges.model import BNModel
    from fedcausal.graph import DAG

__all__ = ["RoundObserver", "RoundStats", "MetricsObserver", "ModelCollector"]


@runtime_checkable
class RoundObserver(Protocol):
    """Receives one callback per built local model and per server fusion."""

    def on_round(self, stage: str, iteration: int, client_id: int, model: BNModel) -> None:
        """Handle a round event (`stage` is ``"build"`` or ``"server"``)."""
        ...


@dataclass
class RoundStats:
    """A structural stats row for one round event.

    Attributes
    ----------
    iteration : int
        1-based round number.
    stage : str
        ``"build"`` (a client's freshly built model) or ``"server"`` (the
        fused global model).
    client_id : int
        Client index, or ``-1`` for the server row.
    edges : int
        Number of directed edges in the model.
    score : float or None
        The model's BDeu score on the owning client's data; ``None`` for the
        server row (the server holds no data).
    shd : int or None
        SHD vs. the ground-truth DAG, or ``None`` if no truth was given.
    smhd : int or None
        SMHD vs. the ground-truth DAG, or ``None`` if no truth was given.
    """

    iteration: int
    stage: str
    client_id: int
    edges: int
    score: float | None
    shd: int | None
    smhd: int | None


class MetricsObserver:
    """Collects :class:`RoundStats` rows, scoring structure against `truth`.

    Parameters
    ----------
    truth : DAG or None, optional
        Ground-truth DAG for SHD/SMHD (default ``None`` -> those stay ``None``).
    """

    def __init__(self, truth: DAG | None = None) -> None:
        self.truth = truth
        self.rows: list[RoundStats] = []

    def on_round(self, stage: str, iteration: int, client_id: int, model: BNModel) -> None:
        """Append a stats row for one round event."""
        s = shd(self.truth, model.dag) if self.truth is not None else None
        sm = smhd(self.truth, model.dag) if self.truth is not None else None
        score = None if stage == "server" else model.score
        self.rows.append(RoundStats(iteration, stage, client_id, model.dag.num_edges, score, s, sm))


class ModelCollector:
    """Collects the full per-round models, transport-agnostically.

    Used by the transport-equivalence test to compare the sequence of global
    models an in-process run and a Flower run produce. ``globals`` holds one
    :class:`~fedcausal.fedges.model.BNModel` per server fusion (in round
    order); ``builds`` holds every built local model.
    """

    def __init__(self) -> None:
        self.builds: list[BNModel] = []
        self.globals: list[BNModel] = []

    def on_round(self, stage: str, iteration: int, client_id: int, model: BNModel) -> None:
        """Store the model, routing server fusions to ``globals``."""
        if stage == "server":
            self.globals.append(model)
        else:
            self.builds.append(model)
