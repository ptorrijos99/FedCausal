"""The BDeu score, with a vectorised counting kernel and a memoised cache.

A family's score is its log marginal likelihood under a BDeu prior plus a
structure prior on the number of parents::

    e = structure_prior
    vm = n_rows - 1
    prior = num_parents * log(e / vm) + (vm - num_parents) * log(1 - e / vm)

The prior is zero when ``e == 0`` and depends only on the number of parents,
so it cancels between parent sets of the same size.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

if TYPE_CHECKING:
    from collections.abc import Collection

    from fedcausal.data import DiscreteDataset
    from fedcausal.graph import DAG

__all__ = ["BDeuScore"]


def _counts(
    values: np.ndarray,
    cards: np.ndarray,
    node: int,
    parents: Collection[int],
) -> np.ndarray:
    """Return the contingency counts ``n_jk`` for ``node`` given ``parents``.

    Parameters
    ----------
    values : np.ndarray
        Integer-coded data of shape ``(n_rows, n_vars)``.
    cards : np.ndarray
        Per-variable cardinalities, shape ``(n_vars,)``.
    node : int
        Index of the child variable.
    parents : tuple[int, ...]
        Indices of the parent variables.

    Returns
    -------
    np.ndarray
        Shape ``(q_seen, r)``: one row per parent configuration that occurs in
        the data, one column per level of `node`.

    Notes
    -----
    Configurations that never occur contribute exactly zero to BDeu, so
    skipping them is exact. The caller must still form the pseudocounts from
    the full number of configurations ``q``, not ``q_seen``.
    """
    child = values[:, node].astype(np.int64)
    r = int(cards[node])
    n_rows = values.shape[0]
    if not parents:
        return np.bincount(child, minlength=r).reshape(1, r)

    code = np.zeros(n_rows, dtype=np.int64)
    q = 1
    for p in parents:
        code = code * int(cards[p]) + values[:, p]
        q *= int(cards[p])

    if q * r <= 4 * n_rows:
        # The full table is small enough to count directly.
        return np.bincount(code * r + child, minlength=q * r).reshape(q, r)

    # Too many configurations for a full table: count only the ones present.
    _, inv = np.unique(code, return_inverse=True)
    inv = inv.astype(np.int64).ravel()
    k = int(inv.max()) + 1
    return np.bincount(inv * r + child, minlength=k * r).reshape(k, r)


class BDeuScore:
    """BDeu local score with a thread-safe cache.

    Parameters
    ----------
    dataset : DiscreteDataset
        The discrete data to score against.
    sample_prior : float, optional
        The equivalent sample size. Default ``1.0``.
    structure_prior : float, optional
        The per-parent penalty ``e`` in the module docstring; a larger value is
        a weaker penalty and ``0.0`` disables it. Default ``1.0``, which
        recovers better structure on large networks; pass ``0.0`` to reproduce
        published FedGES results. See ``docs/algorithms.md``.
    cache : bool, optional
        Memoise local scores. Default ``True``.
    """

    def __init__(
        self,
        dataset: DiscreteDataset,
        sample_prior: float = 1.0,
        structure_prior: float = 1.0,
        cache: bool = True,
    ) -> None:
        self._values = dataset.values
        self._cards = dataset.cardinalities
        self._n_rows = dataset.n_rows
        self.sample_prior = float(sample_prior)
        self.structure_prior = float(structure_prior)
        self._use_cache = cache
        self._cache: dict[tuple[int, frozenset[int]], float] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _prior_term(self, num_parents: int) -> float:
        """Return the structure prior for a family with `num_parents` parents."""
        e = self.structure_prior
        if e == 0:
            return 0.0
        vm = self._n_rows - 1
        return num_parents * math.log(e / vm) + (vm - num_parents) * math.log(1.0 - (e / vm))

    def _compute(self, node: int, parents: Collection[int]) -> float:
        """Compute the BDeu local score without touching the cache."""
        r = int(self._cards[node])
        q = 1
        for p in parents:
            q *= int(self._cards[p])

        n_jk = _counts(self._values, self._cards, node, parents)
        n_j = n_jk.sum(axis=1)

        a_j = self.sample_prior / q
        a_jk = self.sample_prior / (q * r)

        lik = float(
            (gammaln(a_jk + n_jk) - gammaln(a_jk)).sum() + (gammaln(a_j) - gammaln(a_j + n_j)).sum()
        )
        score = lik + self._prior_term(len(parents))
        if not math.isfinite(score):
            return math.nan
        return score

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the BDeu local score of `node` given `parents`.

        Parameters
        ----------
        node : int
            Index of the child variable.
        parents : Collection[int]
            Indices of the parents, in any order. A ``frozenset`` is used as
            the cache key as is.

        Returns
        -------
        float
            The local score, or NaN if it is not finite.
        """
        key_parents = parents if isinstance(parents, frozenset) else frozenset(parents)
        if not self._use_cache:
            return self._compute(node, key_parents)

        # Reads skip the lock: dict.get is atomic, and a concurrent writer can
        # only store the same value. The hit and miss counters are approximate
        # under concurrency.
        key = (node, key_parents)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        value = self._compute(node, key_parents)
        with self._lock:
            self._cache[key] = value
        return value

    def score_dag(self, dag: DAG) -> float:
        """Return the total BDeu score of `dag` (sum of local family scores)."""
        total = 0.0
        for node in range(dag.num_nodes):
            total += self.local_score(node, dag.parents(node))
        return total

    def cache_stats(self) -> tuple[int, int]:
        """Return ``(hits, misses)`` for the local-score cache."""
        with self._lock:
            return (self._hits, self._misses)
