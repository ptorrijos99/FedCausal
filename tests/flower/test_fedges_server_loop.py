"""The FedGES server loop's failure handling, driven over a stub grid.

The failure cases -- a client erroring out, clients disagreeing on their
variables -- are easy to set up in-process and awkward through a real
simulation, which is what `StubGrid` is for.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flwr")

from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.exceptions import (  # noqa: E402
    ClientFailureError,
    ConfigurationError,
    SchemaMismatchError,
)
from fedcausal.fedges.defaults import default_convergence, default_server_fusion  # noqa: E402
from fedcausal.fedges.observer import ModelCollector  # noqa: E402
from fedcausal.flower.fedges import handle_train_message, run_server_loop  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402
from fedcausal.fusion import ConsensusUnion  # noqa: E402

from ._stubs import StubGrid, discrete_chain, server_context, single  # noqa: E402


def _grid(loaders: list[object], fail_nodes: set[int] | None = None) -> StubGrid:
    """A `StubGrid` bound to the FedGES client handler."""
    return StubGrid(
        lambda msg, context, loader: handle_train_message(msg, context, loader, ConsensusUnion()),
        loaders,
        fail_nodes,
    )


def _run(grid: StubGrid, n_clients: int, num_nodes: int, **kwargs: object):  # type: ignore[no-untyped-def]
    return run_server_loop(
        grid,  # type: ignore[arg-type]
        server_context(),
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=n_clients,
        num_nodes=num_nodes,
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_healthy_federation_produces_a_global_model() -> None:
    parts = contiguous_split(discrete_chain(), 3)
    grid = _grid([InMemoryPartitions(parts) for _ in range(3)])
    collector = ModelCollector()

    final = _run(grid, 3, 3, n_iterations=4, observer=collector)

    assert final is not None
    assert collector.globals
    assert final == collector.globals[-1]


def test_the_final_model_lands_on_the_server_state() -> None:
    parts = contiguous_split(discrete_chain(), 3)
    grid = _grid([InMemoryPartitions(parts) for _ in range(3)])
    context = server_context()

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
    parts = contiguous_split(discrete_chain(), 3)
    grid = _grid([InMemoryPartitions(parts) for _ in range(3)], fail_nodes={1})

    with pytest.raises(ClientFailureError, match="needs 3"):
        _run(grid, 3, 3, n_iterations=2)


def test_min_clients_tolerates_a_dropout_when_asked() -> None:
    parts = contiguous_split(discrete_chain(), 3)
    grid = _grid([InMemoryPartitions(parts) for _ in range(3)], fail_nodes={1})

    final = _run(grid, 3, 3, n_iterations=2, min_clients=2)

    assert final is not None


def test_disagreeing_schemas_are_rejected() -> None:
    """Edges travel as bare indices, so mismatched variables are a wrong answer."""
    three, four = discrete_chain(n_vars=3), discrete_chain(n_vars=4)
    grid = _grid([single(three), single(four)])

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        _run(grid, 2, 3, n_iterations=2)


def test_matching_schemas_are_accepted() -> None:
    data = discrete_chain(n_vars=3)
    grid = _grid([single(data), single(data)])

    assert _run(grid, 2, 3, n_iterations=2) is not None


def test_min_clients_is_validated() -> None:
    parts = contiguous_split(discrete_chain(), 2)
    grid = _grid([InMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="min_clients"):
        _run(grid, 2, 3, min_clients=0)
