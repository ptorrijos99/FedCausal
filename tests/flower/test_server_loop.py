"""The server loop's failure handling, driven over a stub grid.

Flower's simulation engine needs Ray, which is not installable everywhere, so
these drive :func:`run_server_loop` against a fake ``Grid`` that routes messages
straight to the client handler in-process. That also makes the failure cases --
a client erroring out, clients disagreeing on their variables -- easy to set up,
which they are not through a real simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from flwr.common import Context, Error, Message, RecordDict  # noqa: E402

from fedcausal.data import DiscreteDataset, contiguous_split  # noqa: E402
from fedcausal.exceptions import (  # noqa: E402
    ClientFailureError,
    ConfigurationError,
    SchemaMismatchError,
)
from fedcausal.federated.defaults import default_convergence, default_server_fusion  # noqa: E402
from fedcausal.federated.observer import ModelCollector  # noqa: E402
from fedcausal.flower.app import handle_train_message, run_server_loop  # noqa: E402
from fedcausal.flower.data import InMemoryPartitions  # noqa: E402
from fedcausal.fusion import ConsensusUnion  # noqa: E402


def _ds(n_vars: int = 3, n_rows: int = 1500, seed: int = 5) -> DiscreteDataset:
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, n_rows)]
    for _ in range(n_vars - 1):
        previous = cols[-1]
        cols.append(np.where(rng.random(n_rows) < 0.9, previous, 1 - previous))
    values = np.stack(cols, axis=1).astype(np.int32)
    names = [f"V{i}" for i in range(n_vars)]
    cards = np.array([2] * n_vars, dtype=np.int32)
    return DiscreteDataset(values, cards, names, [["0", "1"]] * n_vars)


class _StubGrid:
    """Route each message to an in-process client handler.

    Parameters
    ----------
    loaders : list
        One partition loader per supernode.
    fail_nodes : set[int]
        Node indices whose replies should come back carrying an error.
    """

    def __init__(self, loaders: list[object], fail_nodes: set[int] | None = None) -> None:
        self._loaders = loaders
        self._fail = fail_nodes or set()
        self._contexts = [
            Context(
                run_id=1,
                node_id=i,
                node_config={"partition-id": i},
                state=RecordDict(),
                run_config={},
            )
            for i in range(len(loaders))
        ]

    def get_node_ids(self) -> list[int]:
        return list(range(len(self._loaders)))

    def send_and_receive(self, messages: list[Message]) -> list[Message]:
        replies = []
        for node_id, msg in enumerate(messages):
            if node_id in self._fail:
                replies.append(Message(Error(code=1, reason="client exploded"), reply_to=msg))
                continue
            replies.append(
                handle_train_message(
                    msg, self._contexts[node_id], self._loaders[node_id], ConsensusUnion()
                )
            )
        return replies


def _single(data: DiscreteDataset):  # type: ignore[no-untyped-def]
    """Return a loader always serving `data`, whatever the partition id."""
    return lambda context: data


def _server_context() -> Context:
    return Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config={})


def _run(grid: _StubGrid, n_clients: int, num_nodes: int, **kwargs: object):  # type: ignore[no-untyped-def]
    return run_server_loop(
        grid,  # type: ignore[arg-type]
        _server_context(),
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=n_clients,
        num_nodes=num_nodes,
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_healthy_federation_produces_a_global_model() -> None:
    parts = contiguous_split(_ds(), 3)
    grid = _StubGrid([InMemoryPartitions(parts) for _ in range(3)])
    collector = ModelCollector()

    final = _run(grid, 3, 3, n_iterations=4, observer=collector)

    assert final is not None
    assert collector.globals
    assert final == collector.globals[-1]


def test_the_final_model_lands_on_the_server_state() -> None:
    parts = contiguous_split(_ds(), 3)
    grid = _StubGrid([InMemoryPartitions(parts) for _ in range(3)])
    context = _server_context()

    run_server_loop(
        grid,  # type: ignore[arg-type]
        context,
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=3,
        num_nodes=3,
        n_iterations=4,
    )

    assert "global_dag" in context.state.array_records


def test_a_failed_client_stops_the_run() -> None:
    """A dropout must not quietly shrink the ensemble the fusion sees."""
    parts = contiguous_split(_ds(), 3)
    grid = _StubGrid([InMemoryPartitions(parts) for _ in range(3)], fail_nodes={1})

    with pytest.raises(ClientFailureError, match="needs 3"):
        _run(grid, 3, 3, n_iterations=2)


def test_min_clients_tolerates_a_dropout_when_asked() -> None:
    parts = contiguous_split(_ds(), 3)
    grid = _StubGrid([InMemoryPartitions(parts) for _ in range(3)], fail_nodes={1})

    final = _run(grid, 3, 3, n_iterations=2, min_clients=2)

    assert final is not None


def test_disagreeing_schemas_are_rejected() -> None:
    """Edges travel as bare indices, so mismatched variables are a wrong answer."""
    three, four = _ds(n_vars=3), _ds(n_vars=4)
    grid = _StubGrid([_single(three), _single(four)])

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        _run(grid, 2, 3, n_iterations=2)


def test_matching_schemas_are_accepted() -> None:
    data = _ds(n_vars=3)
    grid = _StubGrid([_single(data), _single(data)])

    assert _run(grid, 2, 3, n_iterations=2) is not None


def test_min_clients_is_validated() -> None:
    parts = contiguous_split(_ds(), 2)
    grid = _StubGrid([InMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="min_clients"):
        _run(grid, 2, 3, min_clients=0)
