"""Convergence criteria for the federated loop.

Each criterion is stateful and inspects the array of freshly built local models
each round. ``Server`` calls :meth:`Convergence.check_convergence` after every
client has rebuilt its model and breaks the round loop when it returns ``True``.
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
    """Converged once the mean local score stops strictly improving.

    the mean of the clients'
    scores is compared with the previous round's; convergence is declared when
    it fails to strictly increase.
    """

    def __init__(self) -> None:
        self._last_mean = -math.inf

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff ``mean(score) <= last_mean``."""
        mean = sum(m.score for m in local_models) / len(local_models)
        converged = not (mean > self._last_mean)
        self._last_mean = mean
        return converged


class ModelEquality:
    """Converged when every client re-emits a model it has produced before.

    each client keeps a set of
    all models it has ever produced; convergence is declared when every client's
    current model is already in its own set. Current models are added *after* the
    check, so a first sighting never converges.
    """

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
    """Converged when the current models repeat a recent round's models.

    keeps a sliding window of
    the last ``max_models_saved`` model arrays. If the first model carries a
    nonzero score, rounds are compared by score (``|delta| < 0.001`` per client);
    otherwise by model equality.
    """

    def __init__(self, max_models_saved: int) -> None:
        self._max = max_models_saved
        self._history: list[tuple[BNModel, ...]] = []

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff a windowed past round matches the current one."""
        max_saved = self._max if self._max >= 0 else len(self._history)
        by_score = local_models[0].score != 0
        converged = (
            self._check_stats(local_models, max_saved)
            if by_score
            else self._check_models(local_models, max_saved)
        )
        self._history.append(tuple(local_models))
        if len(self._history) > max_saved:
            self._history.pop(0)
        return converged

    def _check_stats(self, local_models: Sequence[BNModel], max_saved: int) -> bool:
        for k, past in enumerate(reversed(self._history)):
            if k >= max_saved:
                break
            if all(abs(m.score - p.score) < 0.001 for m, p in zip(local_models, past, strict=True)):
                return True
        return False

    def _check_models(self, local_models: Sequence[BNModel], max_saved: int) -> bool:
        for k, past in enumerate(reversed(self._history)):
            if k >= max_saved:
                break
            if all(m == p for m, p in zip(local_models, past, strict=True)):
                return True
        return False


class MultipleConvergence:
    """Converged when any sub-criterion converges (OR).

    Every sub-criterion is
    evaluated each round (they are stateful), so this never short-circuits.
    """

    def __init__(self, criteria: Sequence[Convergence]) -> None:
        self._criteria = list(criteria)

    def check_convergence(self, local_models: Sequence[BNModel]) -> bool:
        """Return ``True`` iff any sub-criterion reports convergence."""
        results = [c.check_convergence(local_models) for c in self._criteria]
        return any(results)
