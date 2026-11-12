"""When to stop the FedGES rounds.

After every site has rebuilt its model the loop asks its criterion, and stops
when it answers ``True``. Criteria keep state between rounds, so use a fresh
one per run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.fedges.model import BNModel

__all__ = ["AnyOf", "Convergence", "ModelEquality", "ScoreImprovement", "default_convergence"]


@runtime_checkable
class Convergence(Protocol):
    """A stopping rule for the rounds."""

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return whether the sites' latest models mean the rounds can stop."""
        ...


class ScoreImprovement:
    """Stop once the sites' mean score no longer improves."""

    def __init__(self) -> None:
        self._last_mean = -math.inf

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return whether the mean score is no better than last round's."""
        mean = sum(m.score for m in local_models) / len(local_models)
        converged = mean <= self._last_mean
        self._last_mean = mean
        return converged


class ModelEquality:
    """Stop once every site rebuilds a model it has built before."""

    def __init__(self) -> None:
        self._seen: list[set[BNModel]] = []

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return whether every site's model repeats one of its earlier ones."""
        if not self._seen:
            self._seen = [set() for _ in local_models]
        converged = all(m in seen for m, seen in zip(local_models, self._seen, strict=True))
        for m, seen in zip(local_models, self._seen, strict=True):
            seen.add(m)
        return converged


class AnyOf:
    """Stop once any of `criteria` would. Every criterion is asked every round."""

    def __init__(self, *criteria: Convergence) -> None:
        self.criteria = criteria

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return whether any criterion reports convergence."""
        return any([c.check_convergence(local_models) for c in self.criteria])


def default_convergence() -> Convergence:
    """Return the default rule: no score improvement, or every model repeated."""
    return AnyOf(ScoreImprovement(), ModelEquality())
