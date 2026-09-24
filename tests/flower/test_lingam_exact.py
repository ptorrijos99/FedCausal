"""Flower transport for the multi-round FedRCD-Exact protocol.

The equivalence claim, that the transport does not change the answer, is
driven over the in-process `StubGrid`, so it runs everywhere. `StubGrid` keeps
one `Context` per node across rounds, which is what this stateful protocol
needs: each client's deflated working matrix lives in `context.state` between
rounds. Only the check through a real simulation needs Ray.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.exceptions import ClientFailureError, SchemaMismatchError  # noqa: E402
from fedcausal.flower.lingam import (  # noqa: E402
    handle_deflate_message,
    run_exact_server_loop,
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
    """The transport must not change the answer. This is the equivalence claim.

    Drives `run_exact_server_loop` over the in-process `StubGrid`, so this
    runs everywhere and is never skipped: it exercises the real
    `handle_deflate_message` handler -- including its `context.state`
    persistence across rounds -- and the real `stats_to_record` /
    `record_to_stats` serialisation, and depends on nothing but `flwr` itself.

    Parametrised over two fixtures deliberately. `continuous_chain` numbers its
    variables in causal order, so `src_local` is 0 at every single round on it
    -- a client that dropped a hard-coded column instead of the server-named
    `src_local` would still pass against it. `permuted_continuous_chain` breaks
    that coincidence and is the case that actually exercises `src_local`
    varying round to round; the identity case is kept too since it still
    covers the simple path.
    """
    from fedcausal.lingam import fedrcd_exact

    data = make_dataset()
    expected = fedrcd_exact(list(np.array_split(data.values, 3, axis=0)))

    parts = contiguous_split(data, 3)
    grid = StubGrid(handle_deflate_message, [InMemoryPartitions(parts) for _ in range(3)])
    actual = run_exact_server_loop(grid, n_clients=3)  # type: ignore[arg-type]

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-10, atol=1e-10)
    # p - 1 rounds of communication -- a round loop that runs one round too many
    # or too few (e.g. an off-by-one in a broadcast guard) may not change the
    # output at all, so this is the only thing that can catch it.
    assert grid.call_count == data.n_vars - 1


def test_disagreeing_schemas_are_rejected_at_round_zero() -> None:
    """Round 0 still carries named variables, so a mismatch is caught before any deflation."""
    three = continuous_chain(0, n_vars=3)
    four = continuous_chain(0, n_vars=4)
    grid = StubGrid(handle_deflate_message, [lambda _ctx: three, lambda _ctx: four])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_exact_server_loop(grid, n_clients=2)  # type: ignore[arg-type]


def test_a_failed_client_stops_the_run() -> None:
    """A dropout must not quietly shrink the ensemble the solver sees, in any round."""
    data = continuous_chain(0)
    parts = contiguous_split(data, 3)
    grid = StubGrid(
        handle_deflate_message,
        [InMemoryPartitions(parts) for _ in range(3)],
        fail_nodes={1},
    )

    with pytest.raises(ClientFailureError, match="needs 3"):
        run_exact_server_loop(grid, n_clients=3)  # type: ignore[arg-type]


@pytest.mark.slow
def test_run_flower_rcd_exact_matches_the_in_process_solver_via_simulation() -> None:
    """The same equivalence claim, once more through a genuine Ray simulation."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower.lingam import run_flower_rcd_exact
    from fedcausal.lingam import fedrcd_exact

    data = continuous_chain(1)
    expected = fedrcd_exact(list(np.array_split(data.values, 3, axis=0)))
    actual = run_flower_rcd_exact(data, n_clients=3)

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-10, atol=1e-10)
