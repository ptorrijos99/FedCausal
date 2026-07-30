"""Shared Flower-transport test doubles for the FedRCD suites.

`_chain_dataset`, `_ContinuousInMemoryPartitions`, and `_StubGrid` are used by
both the one-shot (`test_flower_oneshot.py`) and multi-round
(`test_flower_exact.py`) FedRCD transport suites. `_StubGrid` is parameterised
by the message handler it routes each round to, since the two protocols use
different handlers -- `handle_stats_message` for the one-shot family,
`handle_deflate_message` for the stateful multi-round one -- but everything
else about it (per-node `Context` created once and reused across
`send_and_receive` calls, so client-side state persists between rounds
exactly as it would over a real transport; failure injection) is identical
between the two, which is why it lives here instead of being duplicated.

Not a test module itself (no `test_` prefix), so pytest does not collect it.
Importing it requires `flwr`, exactly like the two suites that import it; both
call `pytest.importorskip("flwr")` before doing so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flwr.common import Context, Error, Message, RecordDict

from fedcausal.data import ContinuousDataset

if TYPE_CHECKING:
    from collections.abc import Callable


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
    """Route each message to an in-process RCD handler, keeping per-node state across rounds.

    A LiNGAM-specific analogue of `tests/flower/test_server_loop.py`'s `_StubGrid`:
    that one calls `fedcausal.flower.app.handle_train_message` with a hardcoded
    `ConsensusUnion` fusion operator, neither of which applies to FedRCD, so this is
    a separate (structurally identical) stub rather than a shared import -- reusing
    it would mean changing what it does.

    Parameters
    ----------
    handler : Callable
        The per-message handler to route to: `handle_stats_message` for the
        one-shot protocol, `handle_deflate_message` for the stateful
        multi-round one.
    loaders : list
        One partition loader per supernode.
    fail_nodes : set[int], optional
        Node indices whose replies should come back carrying an error.
    """

    def __init__(
        self,
        handler: Callable[[Message, Context, object], Message],
        loaders: list[object],
        fail_nodes: set[int] | None = None,
    ) -> None:
        self._handler = handler
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
                self._handler(msg, self._contexts[node_id], self._loaders[node_id])  # type: ignore[arg-type]
            )
        return replies
