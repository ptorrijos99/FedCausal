"""The interface every local score implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy as np
    import numpy.typing as npt


@runtime_checkable
class Score(Protocol):
    """A decomposable score: a DAG scores the sum of its families' local scores."""

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of `node` given `parents`."""
        ...

    def local_scores_with(
        self, node: int, parents: frozenset[int], extra: Sequence[int] | npt.NDArray[np.intp]
    ) -> Sequence[float] | npt.NDArray[np.float64]:
        """Return the local score of `node` given ``parents | {x}``, for each `x` in `extra`."""
        ...
