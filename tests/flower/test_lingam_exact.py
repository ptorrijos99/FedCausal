"""FedRCD-Exact over Flower gives the same result as the in-process solver."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.exceptions import ClientFailureError, SchemaMismatchError  # noqa: E402
from fedcausal.flower.lingam import (  # noqa: E402
    handle_train_message,
    run_server_loop,
)
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402

from ._stubs import StubGrid, continuous_chain, permuted_continuous_chain  # noqa: E402


@pytest.mark.parametrize(
    "make_dataset",
    [
        lambda: continuous_chain(1),
        lambda: permuted_continuous_chain(1, perm=[3, 2, 1, 0]),
    ],
    ids=["identity-order", "permuted-order"],
)
def test_exact_transport_matches_the_in_process_solver(make_dataset) -> None:
    """The transport does not change the answer, and takes ``p - 1`` round trips.

    The permuted case makes the source a different column at every round.
    """
    from fedcausal.lingam import fedrcd_exact

    data = make_dataset()
    expected = fedrcd_exact(list(np.array_split(data.values, 3, axis=0)))

    parts = contiguous_split(data, 3)
    grid = StubGrid(handle_train_message, [InMemoryPartitions(parts) for _ in range(3)])
    actual = run_server_loop(grid, n_clients=3, method="exact")  # type: ignore[arg-type]

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.coef, expected.coef, rtol=1e-10, atol=1e-10)
    assert grid.call_count == data.n_vars - 1


def test_disagreeing_schemas_are_rejected_at_round_zero() -> None:
    """Round 0 still carries named variables, so a mismatch is caught before any deflation."""
    three = continuous_chain(0, n_vars=3)
    four = continuous_chain(0, n_vars=4)
    grid = StubGrid(handle_train_message, [lambda _ctx: three, lambda _ctx: four])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_server_loop(grid, n_clients=2, method="exact")  # type: ignore[arg-type]


def test_a_failed_client_stops_the_run() -> None:
    """A dropout must not quietly shrink the ensemble the solver sees, in any round."""
    data = continuous_chain(0)
    parts = contiguous_split(data, 3)
    grid = StubGrid(
        handle_train_message,
        [InMemoryPartitions(parts) for _ in range(3)],
        fail_nodes={1},
    )

    with pytest.raises(ClientFailureError, match="needs 3"):
        run_server_loop(grid, n_clients=3, method="exact")  # type: ignore[arg-type]


@pytest.mark.slow
def test_run_flower_rcd_exact_matches_the_in_process_solver_via_simulation() -> None:
    """The same equivalence claim, once more through a genuine Ray simulation."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower.lingam import run_flower_rcd
    from fedcausal.lingam import fedrcd_exact

    data = continuous_chain(1)
    expected = fedrcd_exact(list(np.array_split(data.values, 3, axis=0)))
    actual = run_flower_rcd(contiguous_split(data, 3), method="exact")

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.coef, expected.coef, rtol=1e-10, atol=1e-10)
