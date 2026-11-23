"""The greedy hill climb that drives both phases of GES.

Each phase caches the best operator into every child. An operator into `y`
depends on the neighbourhood of `y` and on which neighbours of `y` each `x` is
adjacent to, so after a step only the children whose own neighbourhood changed,
or one of whose undirected neighbours changed, are re-scored.

Whether a semi-directed path is blocked depends on the whole graph, though. The
cached best is therefore re-checked before it is applied, and when the cache
runs dry every child is re-scored once more before the phase stops. The result
is a local optimum of the full neighbourhood.
"""

from __future__ import annotations

import heapq
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
_Finder = Callable[["PDAG", int], "ops.Best"]
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


def _climb(
    pdag: PDAG,
    find: _Finder,
    apply: _Applier,
    candidates: frozenset[Pair] | None,
    allowed: Callable[[int, int], bool] | None = None,
) -> PDAG:
    current = pdag.copy()
    n = current.num_nodes
    # The best operator into each child: its parent, improvement and subset.
    cache: list[tuple[int, float, frozenset[int]] | None] = [None] * n
    # Max-heap of those by improvement, then smallest pair. Each scoring of a
    # child stamps its entry; an entry with an older stamp is stale, and skipped.
    heap: list[tuple[float, int, int, int]] = []
    stamp = [0] * n

    def score_child(y: int) -> ops.Best:
        """Score the operators into `y`, cache the best one and return them all."""
        xs, deltas, which, subsets = find(current, y)
        if candidates is not None or allowed is not None:
            keep = [
                i
                for i, x in enumerate(xs.tolist())
                if (candidates is None or (x, y) in candidates)
                and (allowed is None or allowed(x, y))
            ]
            xs, deltas, which = xs[keep], deltas[keep], which[keep]
        stamp[y] += 1
        if xs.size:
            i = int(np.argmax(deltas))  # the first of the largest: the smallest parent
            best = (int(xs[i]), float(deltas[i]), subsets[which[i]])
            cache[y] = best
            heapq.heappush(heap, (-best[1], best[0], y, stamp[y]))
        else:
            cache[y] = None
        return xs, deltas, which, subsets

    def rescore(children: Iterable[int]) -> None:
        for y in children:
            score_child(y)

    def pop_best() -> tuple[int, int, frozenset[int]] | None:
        while heap:
            _, x, y, stamped = heap[0]
            cached = cache[y]
            if stamped != stamp[y] or cached is None:
                heapq.heappop(heap)
                continue
            # Apply it only if scoring `y` again still gives this operator for `x`.
            xs, deltas, which, subsets = score_child(y)
            i = int(np.searchsorted(xs, x))
            if i < xs.size and xs[i] == x and (deltas[i], subsets[which[i]]) == cached[1:]:
                return x, y, cached[2]
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
        apply(current, x, y, subset)
        # Every node whose marks may differ from before the step: those the operator
        # edited and those the completion changed. A node edited and then restored is
        # re-scored for nothing, which leaves its cached operators as they were.
        changed = {x, y, *subset, *current._complete_in_place().ravel().tolist()}
        affected = set(changed)
        for v in changed:
            affected |= current._undirected_set(v)
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

    def find(g: PDAG, y: int) -> ops.Best:
        return ops.best_inserts_into(score, g, y, max_t_size, max_parents)

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

    def find(g: PDAG, y: int) -> ops.Best:
        return ops.best_deletes_into(score, g, y)

    allowed = knowledge.may_remove if knowledge else None
    return _climb(pdag, find, ops.apply_delete, candidates, allowed)
