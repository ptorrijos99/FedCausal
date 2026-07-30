"""Flower transport for the one-shot FedRCD family.

Follows the split `tests/flower/test_transport_equivalence.py` established for
FedGES: the load-bearing equivalence claim (the transport must not change the
answer) is driven over an in-process stub grid, so it runs everywhere and is
never skipped -- it never touches Flower's Ray-based simulation engine. Only
the tests that call `run_flower_rcd` itself, which spins up real supernodes via
`flwr.simulation.run_simulation`, need Ray and are marked `slow` / gated with
`pytest.importorskip("ray", ...)`, exactly as the FedGES suite gates its own
Ray path.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal.data import ContinuousDataset

pytest.importorskip("flwr")

from flwr.common import Context, Message, MessageType, RecordDict  # noqa: E402

from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.exceptions import (  # noqa: E402
    ClientFailureError,
    ConfigurationError,
    SchemaMismatchError,
)
from fedcausal.flower.lingam import (  # noqa: E402
    build_client_app,
    handle_stats_message,
    run_server_loop,
)
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402
from fedcausal.flower.serde import names_fingerprint  # noqa: E402

from ._stubs import StubGrid, continuous_chain  # noqa: E402


def test_handle_stats_message_replies_with_moments_and_fingerprint() -> None:
    """One round-trip: the reply carries stats, row count, and a schema fingerprint."""
    data = continuous_chain(0, n=500, n_vars=3)
    context = Context(
        run_id=1, node_id=1, node_config={"partition-id": 0}, state=RecordDict(), run_config={}
    )
    msg = Message(
        content=RecordDict({}), dst_node_id=1, message_type=MessageType.TRAIN, group_id="0"
    )

    reply = handle_stats_message(msg, context, InMemoryPartitions([data]))

    control = reply.content["control"]
    assert control["n"] == 500
    assert control["n_vars"] == 3
    assert control["schema"] == names_fingerprint(data.names)


def test_run_server_loop_recovers_the_chain() -> None:
    data = continuous_chain(0)
    parts = contiguous_split(data, 3)
    grid = StubGrid(handle_stats_message, [InMemoryPartitions(parts) for _ in range(3)])

    result = run_server_loop(grid, n_clients=3, solver="fedrcd")  # type: ignore[arg-type]

    assert result.causal_order == (0, 1, 2, 3)


def test_flower_matches_the_in_process_solver_exactly() -> None:
    """The transport must not change the answer. This is the equivalence claim.

    Drives `run_server_loop` over the in-process `StubGrid`, so this runs
    everywhere and is never skipped: it exercises the real `handle_stats_message`
    handler and the real `stats_to_record` / `record_to_stats` serialisation, and
    depends on nothing but `flwr` itself.
    """
    from fedcausal.lingam import fedrcd
    from fedcausal.lingam.cumulants import (
        aggregate_and_compute_cumulants,
        compute_local_raw_stats,
    )

    data = continuous_chain(1)
    parts_arr = np.array_split(data.values, 3, axis=0)
    expected = fedrcd(
        aggregate_and_compute_cumulants([compute_local_raw_stats(p, order=4) for p in parts_arr])
    )

    parts = contiguous_split(data, 3)
    grid = StubGrid(handle_stats_message, [InMemoryPartitions(parts) for _ in range(3)])
    actual = run_server_loop(grid, n_clients=3, solver="fedrcd")  # type: ignore[arg-type]

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-12, atol=1e-12)


def test_disagreeing_schemas_are_rejected() -> None:
    """Statistics travel as aggregated arrays, so mismatched variables are a wrong answer."""
    three = continuous_chain(0, n_vars=3)
    four = continuous_chain(0, n_vars=4)
    grid = StubGrid(handle_stats_message, [lambda _ctx: three, lambda _ctx: four])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_server_loop(grid, n_clients=2)  # type: ignore[arg-type]


def test_same_names_different_order_is_rejected() -> None:
    """The case `_check_schemas` exists for -- and the one the other test above misses.

    A count mismatch (3 vs 4 variables, see `test_disagreeing_schemas_are_rejected`) is
    caught incidentally by numpy's broadcasting rules inside
    `aggregate_and_compute_cumulants` even with no schema check at all: the moment-sum
    arrays have mismatched shapes and something raises regardless.

    Same names, same count, different order is different: every array has an identical
    shape on both clients, so nothing raises on its own. Without `_check_schemas`, the
    server would silently aggregate cumulants across columns that mean different things
    on different clients, and hand back a confident, meaningless causal order. This is
    the only scenario where removing the check would corrupt the result instead of
    raising -- so it is the one that actually needs a test.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(500, 3))
    forward = ContinuousDataset.from_arrays(values, ["a", "b", "c"])
    swapped = ContinuousDataset.from_arrays(values[:, [1, 0, 2]], ["b", "a", "c"])
    grid = StubGrid(handle_stats_message, [lambda _ctx: forward, lambda _ctx: swapped])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_server_loop(grid, n_clients=2)  # type: ignore[arg-type]


def test_a_failed_client_stops_the_run() -> None:
    """A dropout must not quietly shrink the ensemble the solver sees."""
    data = continuous_chain(0)
    parts = contiguous_split(data, 3)
    grid = StubGrid(
        handle_stats_message,
        [InMemoryPartitions(parts) for _ in range(3)],
        fail_nodes={1},
    )

    with pytest.raises(ClientFailureError, match="needs 3"):
        run_server_loop(grid, n_clients=3)  # type: ignore[arg-type]


def test_min_clients_tolerates_a_dropout_when_asked() -> None:
    data = continuous_chain(0)
    parts = contiguous_split(data, 3)
    grid = StubGrid(
        handle_stats_message,
        [InMemoryPartitions(parts) for _ in range(3)],
        fail_nodes={1},
    )

    result = run_server_loop(grid, n_clients=3, min_clients=2)  # type: ignore[arg-type]

    assert result is not None


def test_min_clients_is_validated() -> None:
    data = continuous_chain(0)
    parts = contiguous_split(data, 2)
    grid = StubGrid(handle_stats_message, [InMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="min_clients"):
        run_server_loop(grid, n_clients=2, min_clients=0)  # type: ignore[arg-type]


def test_unknown_solver_is_an_error() -> None:
    data = continuous_chain(0)
    parts = contiguous_split(data, 2)
    grid = StubGrid(handle_stats_message, [InMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="unknown solver"):
        run_server_loop(grid, n_clients=2, solver="bogus")  # type: ignore[arg-type]


def test_build_client_app_returns_a_client_app() -> None:
    """A thin registration wrapper; not exercised through a real supernode here."""
    from flwr.client import ClientApp

    data = continuous_chain(0)
    app = build_client_app(InMemoryPartitions([data]))

    assert isinstance(app, ClientApp)


@pytest.mark.slow
def test_run_flower_rcd_recovers_the_chain() -> None:
    """The Ray-backed public entry point, real supernodes and all."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower.lingam import run_flower_rcd

    result = run_flower_rcd(continuous_chain(0), n_clients=3)
    assert result.causal_order == (0, 1, 2, 3)


@pytest.mark.slow
def test_run_flower_rcd_matches_the_in_process_solver_via_simulation() -> None:
    """The same equivalence claim, once more through a genuine Ray simulation."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower.lingam import run_flower_rcd
    from fedcausal.lingam import fedrcd
    from fedcausal.lingam.cumulants import (
        aggregate_and_compute_cumulants,
        compute_local_raw_stats,
    )

    data = continuous_chain(1)
    parts_arr = np.array_split(data.values, 3, axis=0)
    expected = fedrcd(
        aggregate_and_compute_cumulants([compute_local_raw_stats(p, order=4) for p in parts_arr])
    )
    actual = run_flower_rcd(data, n_clients=3)

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-12, atol=1e-12)
