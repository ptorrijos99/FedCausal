"""The interface every local score implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Collection


@runtime_checkable
class Score(Protocol):
    """A decomposable score: a DAG scores the sum of its families' local scores."""

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of `node` given `parents`."""
        ...
