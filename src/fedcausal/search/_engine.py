"""The greedy hill climb that drives both phases of GES.

Each phase caches the best operator for every pair, grouped by child. An
operator into `y` depends on the neighbourhood of `y` and on which neighbours
of `y` each `x` is adjacent to, so after a step only the children whose own
neighbourhood changed, or one of whose undirected neighbours changed, are
re-scored.

Whether a semi-directed path is blocked depends on the whole graph, though. The
cached best is therefore re-checked before it is applied, and when the cache
runs dry every child is re-scored once more before the phase stops. The result
is a local optimum of the full neighbourhood.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from fedcausal.graph import PDAG
from fedcausal.graph._kernels import consistent_extension, cpdag_of_dag
from fedcausal.search import _operators as ops

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fedcausal.scores import Score
    from fedcausal.search._knowledge import Knowledge

__all__ = ["backward_phase", "complete", "forward_phase"]

Pair = tuple[int, int]
_Finder = Callable[["PDAG", int], "dict[int, ops.Candidate]"]
_Applier = Callable[["PDAG", int, int, frozenset[int]], None]


def complete(pdag: PDAG) -> PDAG:
    """Return the CPDAG of the equivalence class that `pdag` extends to.

    Raises
    ------
    ValueError
        If `pdag` has no consistent extension, which a valid operator never causes.
    """
    dag, ok = consistent_extension(pdag.adjacency().view(np.uint8))
    if not ok:
        raise ValueError("the graph has no consistent extension")
    return PDAG.from_adjacency(cpdag_of_dag(dag))


def _complete_in_place(pdag: PDAG) -> None:
    dag, ok = consistent_extension(pdag._adjacency().view(np.uint8))
    if not ok:
        raise ValueError("the graph has no consistent extension")
    pdag.update_to(cpdag_of_dag(dag))


def _climb(
    pdag: PDAG,
    find: _Finder,
    apply: _Applier,
    candidates: frozenset[Pair] | None,
    allowed: Callable[[int, int], bool] | None = None,
) -> PDAG:
    current = pdag.copy()
    n = current.num_nodes
    cache: list[dict[int, ops.Candidate]] = [{} for _ in range(n)]
    # Max-heap by improvement, then smallest pair. An entry is stale, and
    # skipped, once its pair's cached improvement no longer matches it.
    heap: list[tuple[float, int, int, int]] = []
    tiebreak = itertools.count()

    def rescore(children: Iterable[int]) -> None:
        for y in children:
            found = find(current, y)
            if candidates is not None:
                found = {x: c for x, c in found.items() if (x, y) in candidates}
            if allowed is not None:
                found = {x: c for x, c in found.items() if allowed(x, y)}
            previous = cache[y]
            cache[y] = found
            for x, (delta, _) in found.items():
                old = previous.get(x)
                if old is None or old[0] != delta:
                    heapq.heappush(heap, (-delta, x, y, next(tiebreak)))

    def pop_best() -> tuple[int, int, frozenset[int]] | None:
        while heap:
            neg_delta, x, y, _ = heap[0]
            cached = cache[y].get(x)
            if cached is None or cached[0] != -neg_delta:
                heapq.heappop(heap)
                continue
            rescore([y])
            if cache[y].get(x) == cached:
                heapq.heappop(heap)
                return x, y, cached[1]
        return None

    rescore(range(n))
    while True:
        best = pop_best()
        if best is None:
            rescore(range(n))
            best = pop_best()
            if best is None:
                return current
        x, y, subset = best
        before = current.adjacency()
        apply(current, x, y, subset)
        _complete_in_place(current)
        differ = before != current.adjacency()
        changed = np.flatnonzero(differ.any(axis=0) | differ.any(axis=1)).tolist()
        affected = set(changed)
        for v in changed:
            affected |= current.undirected_set(v)
        rescore(sorted(affected))


def forward_phase(
    score: Score,
    pdag: PDAG,
    candidates: frozenset[Pair] | None = None,
    max_t_size: int | None = 2,
    knowledge: Knowledge | None = None,
    max_parents: int | None = None,
) -> PDAG:
    """Apply the best Insert until none improves the score; return the CPDAG.

    Parameters
    ----------
    score : Score
        Decomposable score.
    pdag : PDAG
        Starting CPDAG. It is not modified.
    candidates : frozenset of (int, int), optional
        If given, only these ordered pairs ``(x, y)`` are considered for
        ``x -> y``.
    max_t_size : int or None, default=2
        Largest `T` tried per pair; ``None`` tries every subset.
    knowledge : Knowledge, optional
        Forbidden adjacencies are never inserted.
    max_parents : int or None, optional
        No Insert gives its child more parents than this.
    """

    def find(g: PDAG, y: int) -> dict[int, ops.Candidate]:
        return ops.inserts_into(score, g, y, max_t_size, max_parents)

    allowed = knowledge.may_add if knowledge else None
    return _climb(pdag, find, ops.apply_insert, candidates, allowed)


def backward_phase(
    score: Score,
    pdag: PDAG,
    candidates: frozenset[Pair] | None = None,
    knowledge: Knowledge | None = None,
) -> PDAG:
    """Apply the best Delete until none improves the score; return the CPDAG.

    Parameters
    ----------
    score : Score
        Decomposable score.
    pdag : PDAG
        Starting CPDAG. It is not modified.
    candidates : frozenset of (int, int), optional
        If given, only the edges between these ordered pairs ``(x, y)`` can be
        deleted. An undirected edge is reachable from either ordering.
    knowledge : Knowledge, optional
        Required adjacencies are never deleted.
    """

    def find(g: PDAG, y: int) -> dict[int, ops.Candidate]:
        return ops.deletes_into(score, g, y)

    allowed = knowledge.may_remove if knowledge else None
    return _climb(pdag, find, ops.apply_delete, candidates, allowed)
