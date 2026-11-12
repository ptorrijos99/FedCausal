"""What every built-in score shares: the structure prior and the cache of families."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy.typing as npt

    from fedcausal.graph import DAG

__all__ = ["FamilyScore"]


class FamilyScore:
    """A decomposable score whose families are computed once and cached.

    Subclasses give the score of one family, :meth:`_family`, and may give a
    faster :meth:`_extensions` for the families an Insert compares. The local
    score adds a structure prior on the number of parents ``k``::

        k * log(p) + (m - k) * log(1 - p),    p = min(expected_parents / m, 1/2)

    a Bernoulli prior in which each of the ``m = n_vars - 1`` candidate parents
    is present with probability ``p``, so `expected_parents` is the expected
    in-degree under the prior. ``p`` is capped at one half, where the prior
    stops penalising edges, so on very few variables it has no effect. Only
    the ``k * log(p / (1 - p))`` part is added; the rest is the same for every
    family of a node.

    Parameters
    ----------
    n_vars : int
        Number of variables.
    expected_parents : float or None
        Expected in-degree under the structure prior; ``None`` is a uniform
        prior over structures.
    """

    def __init__(self, n_vars: int, expected_parents: float | None) -> None:
        if expected_parents is not None and expected_parents <= 0:
            raise ValueError(f"expected_parents must be positive, got {expected_parents}")
        self.n_vars = n_vars
        self.expected_parents = expected_parents
        if expected_parents is None:
            self._log_odds = 0.0
        else:
            p = min(expected_parents / max(n_vars - 1, 1), 0.5)
            self._log_odds = math.log(p) - math.log1p(-p)
        # A family is cached under one integer: a bit per parent, shifted past
        # the bits of the child's index. Far smaller than a frozenset key, which
        # is what bounds memory on dense searches.
        self._node_bits = max(1, n_vars.bit_length())
        self._cache: dict[int, float] = {}
        # What an Insert compares: every extension of one family by one more
        # parent, an array per family indexed by the new parent.
        self._rows: dict[int, npt.NDArray[np.float64]] = {}

    def _family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return the score of `node` given the sorted `parents`, without the prior."""
        raise NotImplementedError

    def _extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return :meth:`_family` of `node` given ``parents + [x]``, for each `x` in `extra`."""
        return np.array([self._family(node, np.sort(np.append(parents, x))) for x in extra])

    def _key(self, node: int, mask: int) -> int:
        return (mask << self._node_bits) | node

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of `node` given `parents`.

        Parameters
        ----------
        node : int
            Index of the child.
        parents : Collection[int]
            Indices of the parents, in any order.

        Returns
        -------
        float
        """
        mask = 0
        for p in parents:
            mask |= 1 << p
        key = self._key(node, mask)
        value = self._cache.get(key)
        if value is None:
            ordered = np.fromiter(sorted(parents), dtype=np.int64, count=len(parents))
            value = float(self._family(node, ordered)) + len(parents) * self._log_odds
            self._cache[key] = value
        return value

    def local_scores_with(
        self, node: int, parents: frozenset[int], extra: Sequence[int] | npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        """Return the local score of `node` given ``parents | {x}``, for each `x` in `extra`.

        Equivalent to calling :meth:`local_score` once per `x`, but the
        families are cached together, one array per `node` and `parents`, and
        the ones not yet cached are computed together.
        """
        mask = 0
        for p in parents:
            mask |= 1 << p
        key = self._key(node, mask)
        row = self._rows.get(key)
        if row is None:
            row = np.full(self.n_vars, np.nan)
            self._rows[key] = row
        extra_array = np.asarray(extra, dtype=np.int64)
        scores = row[extra_array]
        missing = np.isnan(scores)
        if missing.any():
            ordered = np.fromiter(sorted(parents), dtype=np.int64, count=len(parents))
            todo = extra_array[missing]
            row[todo] = self._extensions(node, ordered, todo) + (len(parents) + 1) * self._log_odds
            scores = row[extra_array]
        return scores

    def score_dag(self, dag: DAG) -> float:
        """Return the score of `dag`: the sum of its families' local scores."""
        return sum(self.local_score(v, dag.parents(v)) for v in range(dag.num_nodes))
