"""The fusion-operator interface shared by every consensus method."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.graph import DAG

__all__ = ["FusionOperator"]


@runtime_checkable
class FusionOperator(Protocol):
    """Combine several DAGs over the same node set into one consensus DAG.

    Every consensus method in :mod:`fedcausal.fusion` -- the full union and
    each limited variant -- implements this single method, so the federated
    the server and the clients can use them interchangeably.
    """

    def fuse(self, dags: Sequence[DAG]) -> DAG:
        """Return the consensus DAG of `dags` (all sharing one node set)."""
        ...
