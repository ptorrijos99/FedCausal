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

from flwr.common import Context, Error, Message, MessageType, RecordDict  # noqa: E402

from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.exceptions import (  # noqa: E402
    ClientFailureError,
    ConfigurationError,
    SchemaMismatchError,
)
from fedcausal.flower.serde import names_fingerprint  # noqa: E402
from fedcausal.lingam.flower import (  # noqa: E402
    build_rcd_client_app,
    handle_stats_message,
    run_rcd_server_loop,
)


def _chain_dataset(seed: int, n: int = 6000, n_vars: int = 4) -> ContinuousDataset:
    rng = np.random.default_rng(seed)
    e = rng.laplace(0.0, 1.0, size=(n, n_vars))
    x = np.zeros((n, n_vars))
    x[:, 0] = e[:, 0]
    for j in range(1, n_vars):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return ContinuousDataset.from_arrays(x, [f"x{j}" for j in range(n_vars)])


class _ContinuousInMemoryPartitions:
    """Serve one of several in-memory `ContinuousDataset`s, indexed by ``partition-id``.

    The continuous, test-local analogue of `fedcausal.flower.data.InMemoryPartitions`.
    """

    def __init__(self, partitions: list[ContinuousDataset]) -> None:
        self._partitions = partitions

    def __call__(self, context: Context) -> ContinuousDataset:
        return self._partitions[int(context.node_config["partition-id"])]


class _StubGrid:
    """Route each message to an in-process RCD client handler.

    A LiNGAM-specific analogue of `tests/flower/test_server_loop.py`'s `_StubGrid`:
    that one calls `fedcausal.flower.app.handle_train_message` with a hardcoded
    `ConsensusUnion` fusion operator, neither of which applies to the one-shot
    stats protocol here, so this is a separate (structurally identical) stub
    rather than a shared import -- reusing it would mean changing what it does.

    Parameters
    ----------
    loaders : list
        One partition loader per supernode.
    fail_nodes : set[int], optional
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
                handle_stats_message(msg, self._contexts[node_id], self._loaders[node_id])  # type: ignore[arg-type]
            )
        return replies


def test_handle_stats_message_replies_with_moments_and_fingerprint() -> None:
    """One round-trip: the reply carries stats, row count, and a schema fingerprint."""
    data = _chain_dataset(0, n=500, n_vars=3)
    context = Context(
        run_id=1, node_id=1, node_config={"partition-id": 0}, state=RecordDict(), run_config={}
    )
    msg = Message(
        content=RecordDict({}), dst_node_id=1, message_type=MessageType.TRAIN, group_id="0"
    )

    reply = handle_stats_message(msg, context, _ContinuousInMemoryPartitions([data]))

    control = reply.content["control"]
    assert control["n"] == 500
    assert control["n_vars"] == 3
    assert control["schema"] == names_fingerprint(data.names)


def test_run_rcd_server_loop_recovers_the_chain() -> None:
    data = _chain_dataset(0)
    parts = contiguous_split(data, 3)
    grid = _StubGrid([_ContinuousInMemoryPartitions(parts) for _ in range(3)])

    result = run_rcd_server_loop(grid, n_clients=3, solver="fedrcd")  # type: ignore[arg-type]

    assert result.causal_order == (0, 1, 2, 3)


def test_flower_matches_the_in_process_solver_exactly() -> None:
    """The transport must not change the answer. This is the equivalence claim.

    Drives `run_rcd_server_loop` over the in-process `_StubGrid`, so this runs
    everywhere and is never skipped: it exercises the real `handle_stats_message`
    handler and the real `stats_to_record` / `record_to_stats` serialisation, and
    depends on nothing but `flwr` itself.
    """
    from fedcausal.lingam import fedrcd
    from fedcausal.lingam.cumulants import (
        aggregate_and_compute_cumulants,
        compute_local_raw_stats,
    )

    data = _chain_dataset(1)
    parts_arr = np.array_split(data.values, 3, axis=0)
    expected = fedrcd(
        aggregate_and_compute_cumulants([compute_local_raw_stats(p, order=4) for p in parts_arr])
    )

    parts = contiguous_split(data, 3)
    grid = _StubGrid([_ContinuousInMemoryPartitions(parts) for _ in range(3)])
    actual = run_rcd_server_loop(grid, n_clients=3, solver="fedrcd")  # type: ignore[arg-type]

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-12, atol=1e-12)


def test_disagreeing_schemas_are_rejected() -> None:
    """Statistics travel as aggregated arrays, so mismatched variables are a wrong answer."""
    three = _chain_dataset(0, n_vars=3)
    four = _chain_dataset(0, n_vars=4)
    grid = _StubGrid([lambda _ctx: three, lambda _ctx: four])  # type: ignore[list-item]

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        run_rcd_server_loop(grid, n_clients=2)  # type: ignore[arg-type]


def test_a_failed_client_stops_the_run() -> None:
    """A dropout must not quietly shrink the ensemble the solver sees."""
    data = _chain_dataset(0)
    parts = contiguous_split(data, 3)
    grid = _StubGrid([_ContinuousInMemoryPartitions(parts) for _ in range(3)], fail_nodes={1})

    with pytest.raises(ClientFailureError, match="needs 3"):
        run_rcd_server_loop(grid, n_clients=3)  # type: ignore[arg-type]


def test_min_clients_tolerates_a_dropout_when_asked() -> None:
    data = _chain_dataset(0)
    parts = contiguous_split(data, 3)
    grid = _StubGrid([_ContinuousInMemoryPartitions(parts) for _ in range(3)], fail_nodes={1})

    result = run_rcd_server_loop(grid, n_clients=3, min_clients=2)  # type: ignore[arg-type]

    assert result is not None


def test_min_clients_is_validated() -> None:
    data = _chain_dataset(0)
    parts = contiguous_split(data, 2)
    grid = _StubGrid([_ContinuousInMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="min_clients"):
        run_rcd_server_loop(grid, n_clients=2, min_clients=0)  # type: ignore[arg-type]


def test_unknown_solver_is_an_error() -> None:
    data = _chain_dataset(0)
    parts = contiguous_split(data, 2)
    grid = _StubGrid([_ContinuousInMemoryPartitions(parts) for _ in range(2)])

    with pytest.raises(ConfigurationError, match="unknown solver"):
        run_rcd_server_loop(grid, n_clients=2, solver="bogus")  # type: ignore[arg-type]


def test_build_rcd_client_app_returns_a_client_app() -> None:
    """A thin registration wrapper; not exercised through a real supernode here."""
    from flwr.client import ClientApp

    data = _chain_dataset(0)
    app = build_rcd_client_app(_ContinuousInMemoryPartitions([data]))

    assert isinstance(app, ClientApp)


@pytest.mark.slow
def test_run_flower_rcd_recovers_the_chain() -> None:
    """The Ray-backed public entry point, real supernodes and all."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.lingam.flower import run_flower_rcd

    result = run_flower_rcd(_chain_dataset(0), n_clients=3)
    assert result.causal_order == (0, 1, 2, 3)


@pytest.mark.slow
def test_run_flower_rcd_matches_the_in_process_solver_via_simulation() -> None:
    """The same equivalence claim, once more through a genuine Ray simulation."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.lingam import fedrcd
    from fedcausal.lingam.cumulants import (
        aggregate_and_compute_cumulants,
        compute_local_raw_stats,
    )
    from fedcausal.lingam.flower import run_flower_rcd

    data = _chain_dataset(1)
    parts_arr = np.array_split(data.values, 3, axis=0)
    expected = fedrcd(
        aggregate_and_compute_cumulants([compute_local_raw_stats(p, order=4) for p in parts_arr])
    )
    actual = run_flower_rcd(data, n_clients=3)

    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.b, expected.b, rtol=1e-12, atol=1e-12)


def test_lingam_package_imports_without_flwr(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fedcausal.lingam` must not drag in flwr; only its flower module may."""
    import sys

    monkeypatch.setitem(sys.modules, "flwr", None)
    for name in [m for m in sys.modules if m.startswith("fedcausal.lingam")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    import fedcausal.lingam  # noqa: F401
