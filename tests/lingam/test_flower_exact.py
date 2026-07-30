"""Flower transport for the multi-round FedRCD-Exact protocol.

Follows the split `tests/lingam/test_flower_oneshot.py` established for the
one-shot family: the load-bearing equivalence claim (the transport must not
change the answer) is driven over an in-process stub grid, so it runs
everywhere and is never skipped -- it never touches Flower's Ray-based
simulation engine. Only the tests that call `run_flower_rcd_exact` itself,
which spins up real supernodes via `flwr.simulation.run_simulation`, need Ray
and are marked `slow` / gated with `pytest.importorskip("ray", ...)`, exactly
as the one-shot suite gates its own Ray path.

The `_StubGrid` used below persists a `Context` per node across
`send_and_receive` calls (see `_flower_stub.py`), which is exactly what this
stateful, multi-round protocol needs: each client's progressively deflated
working matrix lives in that persisted `context.state` between rounds.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.exceptions import ClientFailureError, SchemaMismatchError  # noqa: E402
from fedcausal.lingam.flower import (  # noqa: E402
    build_exact_client_app,
    handle_deflate_message,
    run_exact_server_loop,
)

from ._flower_stub import (  # noqa: E402
    _chain_dataset,
    _ContinuousInMemoryPartitions,
    _permuted_chain_dataset,
    _StubGrid,
)


def test_run_exact_server_loop_recovers_the_chain() -> None:
    data = _chain_dataset(0)
    parts = contiguous_split(data, 3)
    grid = _StubGrid(
        handle_deflate_message, [_ContinuousInMemoryPartitions(parts) for _ in range(3)]
    )

    result = run_exact_server_loop(grid, n_clients=3)  # type: ignore[arg-type]

    assert result.causal_order == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "make_dataset",
    [
        lambda: _chain_dataset(1),
        lambda: _permuted_chain_dataset(1, perm=[3, 2, 1, 0]),
    ],
    ids=["identity-order", "permuted-order"],
)
def test_exact_transport_matches_the_in_process_solver(make_dataset) -> None:
    """The transport must not change the answer. This is the equivalence claim.

    Drives `run_exact_server_loop` over the in-process `_StubGrid`, so this
    runs everywhere and is never skipped: it exercises the real
    `handle_deflate_message` handler -- including its `context.state`
    persistence across rounds -- and the real `stats_to_record` /
    `record_to_stats` serialisation, and depends on nothing but `flwr` itself.

    Parametrised over two fixtures deliberately. `_chain_dataset` numbers its
    variables in causal order, so `src_local` is 0 at every single round on it
    -- a client that dropped a hard-coded column instead of the server-named
    `src_local` would still pass against it. `_permuted_chain_dataset` breaks
    that coincidence and is the case that actually exercises `src_local`
    varying round to round; the identity case is kept too since it still
    covers the simple path.
    """
    from fedcausal.lingam import fedrcd_exact

    data = make_dataset()
    expected = fedrcd_exact(list(np.array_split(data.values, 3, axis=0)))

    parts = contiguous_split(data, 3)
    grid = _StubGrid(
        handle_deflate_message, [_ContinuousInMemoryPartitions(parts) for _ in range(3)]
    )
    actual = run_exact_server_loop(grid, n_clients=3)  # type: ignore[arg-type]

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-10, atol=1e-10)
    # p - 1 rounds of communication -- a round loop that runs one round too many
    # or too few (e.g. an off-by-one in a broadcast guard) may not change the
    # output at all, so this is the only thing that can catch it.
    assert grid.call_count == data.n_vars - 1


def test_client_state_shrinks_by_one_column_per_round():
    """Each round drops the identified source and deflates the survivors by it.

    Checks the actual residual values, not just the shrunk width: `src_local`
    is deliberately not 0 and `betas` is deliberately non-zero, so a client
    that dropped the wrong column, or dropped the right column but skipped the
    `x_j -= beta_j * x_s` deflation, would produce a matrix of the right shape
    but the wrong contents -- which a shape-only assertion cannot see.
    """
    from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict

    from fedcausal.lingam.flower import _WORKING_STATE_KEY

    data = _chain_dataset(2, n=500)
    context = Context(run_id=0, node_id=0, node_config={}, state=RecordDict(), run_config={})

    def loader(_ctx):
        return data

    msg = Message(
        RecordDict({"control": ConfigRecord({"round": 0})}), dst_node_id=0, message_type="train"
    )
    handle_deflate_message(msg, context, loader)
    first = context.state[_WORKING_STATE_KEY].to_numpy_ndarrays()[0]
    assert first.shape[1] == 4
    np.testing.assert_allclose(first, data.values)

    src_local = 1
    betas = np.array([0.3, -0.5, 0.2, 0.1])
    follow_up = Message(
        RecordDict(
            {
                "control": ConfigRecord({"round": 1, "src_local": src_local}),
                "betas": ArrayRecord([betas]),
            }
        ),
        dst_node_id=0,
        message_type="train",
    )
    handle_deflate_message(follow_up, context, loader)
    second = context.state[_WORKING_STATE_KEY].to_numpy_ndarrays()[0]

    remaining = [j for j in range(4) if j != src_local]
    x_s = data.values[:, src_local]
    expected = np.column_stack([data.values[:, j] - betas[j] * x_s for j in remaining])

    assert second.shape[1] == 3
    np.testing.assert_allclose(second, expected)


def test_disagreeing_schemas_are_rejected_at_round_zero() -> None:
    """Round 0 still carries named variables, so a mismatch is caught before any deflation."""
    three = _chain_dataset(0, n_vars=3)
    four = _chain_dataset(0, n_vars=4)
    grid = _StubGrid(handle_deflate_message, [lambda _ctx: three, lambda _ctx: four])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_exact_server_loop(grid, n_clients=2)  # type: ignore[arg-type]


def test_a_failed_client_stops_the_run() -> None:
    """A dropout must not quietly shrink the ensemble the solver sees, in any round."""
    data = _chain_dataset(0)
    parts = contiguous_split(data, 3)
    grid = _StubGrid(
        handle_deflate_message,
        [_ContinuousInMemoryPartitions(parts) for _ in range(3)],
        fail_nodes={1},
    )

    with pytest.raises(ClientFailureError, match="needs 3"):
        run_exact_server_loop(grid, n_clients=3)  # type: ignore[arg-type]


def test_build_exact_client_app_returns_a_client_app() -> None:
    """A thin registration wrapper; not exercised through a real supernode here."""
    from flwr.client import ClientApp

    data = _chain_dataset(0)
    app = build_exact_client_app(_ContinuousInMemoryPartitions([data]))

    assert isinstance(app, ClientApp)


@pytest.mark.slow
def test_exact_transport_recovers_the_chain() -> None:
    """The Ray-backed public entry point, real supernodes and all."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.lingam.flower import run_flower_rcd_exact

    assert run_flower_rcd_exact(_chain_dataset(0), n_clients=3).causal_order == (0, 1, 2, 3)


@pytest.mark.slow
def test_run_flower_rcd_exact_matches_the_in_process_solver_via_simulation() -> None:
    """The same equivalence claim, once more through a genuine Ray simulation."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.lingam import fedrcd_exact
    from fedcausal.lingam.flower import run_flower_rcd_exact

    data = _chain_dataset(1)
    expected = fedrcd_exact(list(np.array_split(data.values, 3, axis=0)))
    actual = run_flower_rcd_exact(data, n_clients=3)

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-10, atol=1e-10)


def test_lingam_package_imports_without_flwr(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fedcausal.lingam` must not drag in flwr; only its flower module may.

    Duplicated from `test_flower_oneshot.py` deliberately: this file must
    stand on its own as a regression guard even if the one-shot suite's copy
    is ever removed, and monkeypatching `sys.modules` is cheap enough that
    sharing it would not be worth the indirection.
    """
    import sys

    monkeypatch.setitem(sys.modules, "flwr", None)
    for name in [m for m in sys.modules if m.startswith("fedcausal.lingam")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    import fedcausal.lingam  # noqa: F401
