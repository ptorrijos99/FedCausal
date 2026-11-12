"""In-process doubles for the transport tests, so they run without Ray.

`StubGrid` stands in for a Flower ``Grid``: it hands each message straight to
the real client handler and keeps one `Context` per node across rounds, as a
real transport would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flwr.common import Context, Error, Message, RecordDict

from fedcausal.data import ContinuousDataset, DiscreteDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class StubGrid:
    """Route each message to an in-process client handler.

    Parameters
    ----------
    handler : Callable
        The client handler, called as ``handler(msg, context, loader)``.
    loaders : Sequence
        One partition loader per supernode.
    fail_nodes : set[int], optional
        Nodes whose replies carry an error.

    Attributes
    ----------
    call_count : int
        How many round trips the protocol made.
    """

    def __init__(
        self,
        handler: Callable[[Message, Context, object], Message],
        loaders: Sequence[object],
        fail_nodes: set[int] | None = None,
    ) -> None:
        self._handler = handler
        self._loaders = list(loaders)
        self._fail = fail_nodes or set()
        self._contexts = [node_context(i) for i in range(len(self._loaders))]
        self.call_count = 0

    def get_node_ids(self) -> list[int]:
        return list(range(len(self._loaders)))

    def send_and_receive(self, messages: list[Message]) -> list[Message]:
        self.call_count += 1
        replies = []
        for node_id, msg in enumerate(messages):
            if node_id in self._fail:
                replies.append(Message(Error(code=1, reason="client exploded"), reply_to=msg))
                continue
            replies.append(self._handler(msg, self._contexts[node_id], self._loaders[node_id]))
        return replies


def node_context(partition_id: int = 0, **node_config: object) -> Context:
    """Return a supernode context carrying `partition_id` in its node config."""
    config = {"partition-id": partition_id, **node_config}
    return Context(run_id=1, node_id=1, node_config=config, state=RecordDict(), run_config={})


def server_context() -> Context:
    """Return a bare server-side context."""
    return Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config={})


def single(data: object) -> Callable[[Context], object]:
    """Return a loader always serving `data`, whatever the partition id."""
    return lambda context: data


def discrete_chain(n_vars: int = 3, n_rows: int = 1500, seed: int = 5) -> DiscreteDataset:
    """Binary chain V0 -> V1 -> ... with strong dependence between neighbours."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, n_rows)]
    for _ in range(n_vars - 1):
        previous = cols[-1]
        cols.append(np.where(rng.random(n_rows) < 0.9, previous, 1 - previous))
    values = np.stack(cols, axis=1).astype(np.int32)
    cards = np.array([2] * n_vars, dtype=np.int32)
    return DiscreteDataset(values, cards, [f"V{i}" for i in range(n_vars)], [["0", "1"]] * n_vars)


def continuous_chain(seed: int, n: int = 6000, n_vars: int = 4) -> ContinuousDataset:
    """Linear non-Gaussian chain x0 -> x1 -> ... with Laplace noise."""
    rng = np.random.default_rng(seed)
    e = rng.laplace(0.0, 1.0, size=(n, n_vars))
    x = np.zeros((n, n_vars))
    x[:, 0] = e[:, 0]
    for j in range(1, n_vars):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return ContinuousDataset(x, [f"x{j}" for j in range(n_vars)])


def permuted_continuous_chain(seed: int, perm: Sequence[int], n: int = 6000) -> ContinuousDataset:
    """The chain of `continuous_chain`, with its columns permuted.

    In causal order the source is always column 0, so a client that drops the
    wrong column would go unnoticed; permuting makes it matter.
    """
    n_vars = len(perm)
    base = continuous_chain(seed, n=n, n_vars=n_vars)
    return ContinuousDataset(base.values[:, list(perm)], [f"x{j}" for j in range(n_vars)])
