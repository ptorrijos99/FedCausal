"""BDeu (Bayesian Dirichlet equivalent uniform) score with a counting cache.

This module ports Tetrad's ``BdeuScore.localScore`` (tetrad-lib 7.6.5, see
``tools/java-golden/reference-src/BdeuScore.java``) to a vectorized NumPy
kernel, wrapped in :class:`BDeuScore`. It is the numerically critical inner
loop of GES-style search: at ~1000 variables the local score is evaluated
millions of times, so a fast counting kernel and a memoizing cache matter.

The score is the log marginal likelihood of a node given its parents under a
BDeu prior, plus Tetrad's structure-prior term. Higher is more dependence;
negative indicates independence (Tetrad's convention).

Prior term (ported verbatim from ``BdeuScore.getPriorForStructure``)::

    e = structure_prior
    if e == 0: return 0.0
    vm = N - 1                       # N = number of complete rows (== n_rows here)
    return num_parents * log(e / vm) + (vm - num_parents) * log(1 - e / vm)

Note that Tetrad passes the *row count* ``N`` (not the variable count) as the
second argument, so ``vm = n_rows - 1``. Because it depends only on
``num_parents`` and the (constant) sample size, the structure-prior term is
identical for any two parent sets of the same size and therefore cancels in
score *differences* between equal-sized parent sets.
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
        Counts of shape ``(q_realized, r)`` -- one row per **realized**
        parent configuration and one column per child level ``r =
        cards[node]``. Parent configurations that occur in zero rows are
        omitted (see Notes).

    Notes
    -----
    **Only realized configurations are counted, and this is exact.** A
    parent configuration with zero rows contributes
    ``lgamma(a_j) - lgamma(a_j) = 0`` to the row term and, per child level,
    ``lgamma(a_jk) - lgamma(a_jk) = 0`` to the cell term, so absent
    configurations add nothing to the likelihood. Summing over the realized
    configurations alone therefore reproduces the full sum over
    ``q`` configurations exactly. **The caller must still use the full ``q``
    (product of parent cardinalities), not ``q_realized``, when forming the
    Dirichlet pseudocounts** ``a_j = ess / q`` and ``a_jk = ess / (q * r)``.

    Parent configurations are encoded as mixed-radix ``int64`` codes. The
    encoding is injective for ``q <= 2**63``; realistic GES families are far
    below that bound, so ``int64`` never overflows in practice.
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
        # Dense path: bincount over the full q*r table. Cheap because the
        # guard bounds q*r by 4*n_rows, so the flattened table is small.
        return np.bincount(code * r + child, minlength=q * r).reshape(q, r)

    # Sparse path: relabel to realized configurations only via unique(),
    # which avoids allocating a length-(q*r) table when q is huge.
    _, inv = np.unique(code, return_inverse=True)
    inv = inv.astype(np.int64).ravel()
    k = int(inv.max()) + 1
    return np.bincount(inv * r + child, minlength=k * r).reshape(k, r)


class BDeuScore:
    """BDeu local score with a memoizing, thread-safe cache.

    The score of
    a family ``(node, parents)`` is the BDeu log marginal likelihood plus
    Tetrad's structure-prior term (see the module docstring).

    Parameters
    ----------
    dataset : DiscreteDataset
        The discrete data to score against.
    sample_prior : float, optional
        The equivalent sample size (ESS). Default
        ``1.0``.
    structure_prior : float, optional
        Penalises parent count; ``0.0`` disables the structure-prior
        term. Default ``0.0``.
    cache : bool, optional
        If ``True`` (default), memoize local scores keyed by
        ``(node, sorted_parents)``.

    Notes
    -----
    The defaults reproduce what the original Java implementation actually
    does. It declares ``structurePrior = 0.001`` and ``samplePrior = 10.0``
    but never propagates them to the score object, so it scores with Tetrad's
    own defaults of 1.0 and 0.0. The committed reference outputs record the
    values actually used, so ``BDeuScore(ds)`` reproduces published results
    out of the box. Pass the priors explicitly for anything else.

    **Full vs. realized configurations.** The counting kernel returns one
    row per *realized* parent configuration, but the Dirichlet pseudocounts
    always use the *full* ``q`` (product of parent cardinalities):
    ``a_j = sample_prior / q`` and ``a_jk = sample_prior / (q * r)``. This is
    exact because empty configurations contribute zero to the likelihood.
    """

    def __init__(
        self,
        dataset: DiscreteDataset,
        sample_prior: float = 1.0,
        structure_prior: float = 0.0,
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
        """Return Tetrad's structure-prior term for ``num_parents`` parents.

        Verbatim port of ``BdeuScore.getPriorForStructure`` with ``N`` equal
        to the (constant) number of complete rows, i.e. ``vm = n_rows - 1``.
        """
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
        # Final guard: non-finite scores collapse to NaN.
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
            Indices of the parent variables, in any order (a ``frozenset``,
            ``tuple``, or any other collection). The cache key is canonicalized
            to a ``frozenset``, so ``(2, 1)`` and ``(1, 2)`` hit the same entry;
            passing a ``frozenset`` directly (as the GES operators do) skips
            even that rebuild. The score itself is invariant to parent order
            (parent order only permutes the contingency configurations, which
            sum identically), so keying on the unordered set is exact.

        Returns
        -------
        float
            The local score of the family ``(node, parents)``.
        """
        key_parents = parents if isinstance(parents, frozenset) else frozenset(parents)
        if not self._use_cache:
            return self._compute(node, key_parents)

        # Lock-free read on the hot path. ``dict.get`` is atomic under the GIL,
        # and because the kernel is pure a concurrent writer only ever installs
        # the *same* value under this key, so a hit never returns a wrong or
        # torn result. GES hits this ~97% of the time (see the profile in note
        # 2026-07-25), where taking the lock was pure overhead. The hit/miss
        # counters are updated without the lock and so are best-effort under
        # concurrency -- diagnostics only, never correctness.
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
