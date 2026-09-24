"""Stopping rules for the FedGES loop.

The server asks its criterion after every site has rebuilt its model, and stops
when it answers ``True``. Criteria keep state between rounds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.fedges.model import BNModel

__all__ = [
    "Convergence",
    "NoneConvergence",
    "ScoreImprovement",
    "ModelEquality",
    "IterationEquality",
    "MultipleConvergence",
]


@runtime_checkable
class Convergence(Protocol):
    """A federated convergence criterion."""

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff the local models have converged."""
        ...


class NoneConvergence:
    """Never converges; the loop runs for the full iteration budget."""

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``False`` always."""
        return False


class ScoreImprovement:
    """Converged once the mean local score stops strictly improving."""

    def __init__(self) -> None:
        self._last_mean = -math.inf

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff ``mean(score) <= last_mean``."""
        mean = sum(m.score for m in local_models) / len(local_models)
        converged = not (mean > self._last_mean)
        self._last_mean = mean
        return converged


class ModelEquality:
    """Converged when every client produces a model it has produced before."""

    def __init__(self) -> None:
        self._seen: list[set[BNModel]] | None = None

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff every client's current model was seen earlier."""
        if self._seen is None:
            self._seen = [set() for _ in local_models]
        converged = True
        for i, m in enumerate(local_models):
            if m not in self._seen[i]:
                converged = False
                break
        for i, m in enumerate(local_models):
            self._seen[i].add(m)
        return converged


class IterationEquality:
    """Converged when the current round repeats one of the last few.

    Rounds are compared by score, each site within 0.001, when the models carry
    scores, and by structure otherwise.

    Parameters
    ----------
    max_models_saved : int
        How many past rounds to compare against. A negative value compares
        against every past round.
    """

    def __init__(self, max_models_saved: int) -> None:
        self._max = max_models_saved
        self._history: list[tuple[BNModel, ...]] = []

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff a remembered round matches the current one."""
        by_score = local_models[0].score != 0
        converged = any(
            all(
                abs(m.score - p.score) < 0.001 if by_score else m == p
                for m, p in zip(local_models, past, strict=True)
            )
            for past in self._history
        )
        self._history.append(tuple(local_models))
        if 0 <= self._max < len(self._history):
            self._history.pop(0)
        return converged


class MultipleConvergence:
    """Converged when any of its criteria is.

    Every criterion is asked every round, so each keeps its state up to date.
    """

    def __init__(self, criteria: Sequence[Convergence]) -> None:
        self._criteria = list(criteria)

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff any sub-criterion reports convergence."""
        results = [c.check_convergence(local_models) for c in self._criteria]
        return any(results)
