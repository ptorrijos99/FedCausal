"""The one-shot FedRCD solvers over Flower."""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal.data import ContinuousDataset

pytest.importorskip("flwr")


from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.exceptions import (  # noqa: E402
    ClientFailureError,
    ConfigurationError,
    SchemaMismatchError,
)
from fedcausal.flower.lingam import (  # noqa: E402
    handle_train_message,
    run_server_loop,
)
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402

from ._stubs import StubGrid, continuous_chain  # noqa: E402


@pytest.mark.parametrize(("method", "order"), [("deflation", 4), ("ratio", 3), ("sort", 3)])
def test_flower_matches_the_in_process_solver_exactly(method: str, order: int) -> None:
    """The transport does not change the answer."""
    from fedcausal.lingam import fedrcd
    from fedcausal.lingam.cumulants import (
        cumulants_from_sums,
        moment_sums,
    )

    data = continuous_chain(1)
    parts_arr = np.array_split(data.values, 3, axis=0)
    expected = fedrcd(
        cumulants_from_sums([moment_sums(p, order) for p in parts_arr]),
        method=method,  # type: ignore[arg-type]
        order=order,  # type: ignore[arg-type]
    )

    parts = contiguous_split(data, 3)
    grid = StubGrid(handle_train_message, [InMemoryPartitions(parts) for _ in range(3)])
    actual = run_server_loop(grid, n_clients=3, method=method, order=order)  # type: ignore[arg-type]

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.coef, expected.coef, rtol=1e-12, atol=1e-12)


def test_disagreeing_schemas_are_rejected() -> None:
    """Statistics travel as aggregated arrays, so mismatched variables are a wrong answer."""
    three = continuous_chain(0, n_vars=3)
    four = continuous_chain(0, n_vars=4)
    grid = StubGrid(handle_train_message, [lambda _ctx: three, lambda _ctx: four])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_server_loop(grid, n_clients=2)  # type: ignore[arg-type]


def test_same_names_different_order_is_rejected() -> None:
    """The same variables in a different order must be rejected.

    The arrays have the same shapes, so nothing else would fail: the server would
    add up statistics of different variables and return a meaningless order.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(500, 3))
    forward = ContinuousDataset(values, ["a", "b", "c"])
    swapped = ContinuousDataset(values[:, [1, 0, 2]], ["b", "a", "c"])
    grid = StubGrid(handle_train_message, [lambda _ctx: forward, lambda _ctx: swapped])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_server_loop(grid, n_clients=2)  # type: ignore[arg-type]


def test_a_failed_client_stops_the_run() -> None:
    """A dropout must not quietly shrink the ensemble the solver sees."""
    data = continuous_chain(0)
    parts = contiguous_split(data, 3)
    grid = StubGrid(
        handle_train_message,
        [InMemoryPartitions(parts) for _ in range(3)],
        fail_nodes={1},
    )

    with pytest.raises(ClientFailureError, match="needs 3"):
        run_server_loop(grid, n_clients=3)  # type: ignore[arg-type]


def test_min_clients_tolerates_a_dropout_when_asked() -> None:
    data = continuous_chain(0)
    parts = contiguous_split(data, 3)
    grid = StubGrid(
        handle_train_message,
        [InMemoryPartitions(parts) for _ in range(3)],
        fail_nodes={1},
    )

    result = run_server_loop(grid, n_clients=3, min_clients=2)  # type: ignore[arg-type]

    assert result is not None


def test_min_clients_is_validated() -> None:
    data = continuous_chain(0)
    parts = contiguous_split(data, 2)
    grid = StubGrid(handle_train_message, [InMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="min_clients"):
        run_server_loop(grid, n_clients=2, min_clients=0)  # type: ignore[arg-type]


def test_unknown_method_is_an_error() -> None:
    data = continuous_chain(0)
    parts = contiguous_split(data, 2)
    grid = StubGrid(handle_train_message, [InMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="unknown method"):
        run_server_loop(grid, n_clients=2, method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="order must be 3 or 4"):
        run_server_loop(grid, n_clients=2, order=5)  # type: ignore[arg-type]


@pytest.mark.slow
def test_run_flower_rcd_matches_the_in_process_solver_via_simulation() -> None:
    """The same equivalence claim, once more through a genuine Ray simulation."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower.lingam import run_flower_rcd
    from fedcausal.lingam import fedrcd
    from fedcausal.lingam.cumulants import (
        cumulants_from_sums,
        moment_sums,
    )

    data = continuous_chain(1)
    parts_arr = np.array_split(data.values, 3, axis=0)
    expected = fedrcd(cumulants_from_sums([moment_sums(p, order=4) for p in parts_arr]))
    actual = run_flower_rcd(contiguous_split(data, 3))

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.coef, expected.coef, rtol=1e-12, atol=1e-12)
