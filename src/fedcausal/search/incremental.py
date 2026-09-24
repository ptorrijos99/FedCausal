"""Incremental forward and backward phases: re-score only what an operator changed.

The full scan in :mod:`fedcausal.search.ges` re-scores every candidate pair on
every step. These engines cache the best operator per pair and, after each
step, re-score only the pairs touching a node whose neighbourhood changed.

A candidate's score delta depends only on its endpoints' neighbourhoods, so the
cached deltas stay correct. Validity is not fully local (clique and path checks
look further), so a cached candidate is re-checked before it is applied, and
``tests/search/test_incremental.py`` checks that both engines give the same
CPDAG on every shipped network. ``incremental=False`` in :class:`GES` selects
the full scan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.search import gesops as ops
from fedcausal.search._candidates import (
    EPS,
    H_POOLS,
    adjacent_ordered_pairs,
    best_delete_for_pair,
    best_insert_for_pair,
    invalid_h_pool_error,
    rebuild_cpdag,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from fedcausal.graph.pdag import PDAG
    from fedcausal.scores import Score
    from fedcausal.search._candidates import HPool

__all__ = ["run_fes_incremental", "run_bes_incremental"]

# (parents, children, undirected neighbours): equal iff no incident edge changed.
_Sig = tuple[frozenset[int], frozenset[int], frozenset[int]]
_Cache = dict[tuple[int, int], tuple[float, frozenset[int]]]


def _sig(pdag: PDAG, v: int) -> _Sig:
    """Return node `v`'s neighbourhood signature."""
    return (
        frozenset(pdag.parents_set(v)),
        frozenset(pdag.children_set(v)),
        frozenset(pdag.undirected_set(v)),
    )


def _changed_nodes(before: list[_Sig], after: list[_Sig], n: int) -> set[int]:
    """Return the nodes whose signature differs between two graph states."""
    return {v for v in range(n) if before[v] != after[v]}


def _incident_ordered_pairs(changed: set[int], n: int) -> Iterator[tuple[int, int]]:
    """Yield every ordered pair with an endpoint in `changed`, each once."""
    seen: set[tuple[int, int]] = set()
    for v in changed:
        for u in range(n):
            if u == v:
                continue
            for pair in ((u, v), (v, u)):
                if pair not in seen:
                    seen.add(pair)
                    yield pair


def _all_ordered_pairs(n: int) -> Iterator[tuple[int, int]]:
    """Yield every ordered pair ``(x, y)`` with ``x != y``."""
    for x in range(n):
        for y in range(n):
            if x != y:
                yield (x, y)


def _select(cache: _Cache) -> tuple[float, int, int, frozenset[int]] | None:
    """Return the best cached operator ``(delta, x, y, subset)``, or `None`.

    Uses the full scan's order and tie rule, so both engines pick the same one.
    """
    best: tuple[float, int, int, frozenset[int]] | None = None
    for x, y in sorted(cache):
        delta, subset = cache[(x, y)]
        if best is None or delta > best[0] + EPS:
            best = (delta, x, y, subset)
    return best


def _pop_best_valid(
    cache: _Cache,
    fresh: Callable[[int, int], tuple[float, frozenset[int]] | None],
) -> tuple[float, int, int, frozenset[int]] | None:
    """Return the best cached candidate once a fresh re-score confirms it.

    A change elsewhere in the graph can invalidate a cached candidate, so the
    best one is re-scored with `fresh`; if that disagrees, the cache is
    corrected and the selection repeats. This never applies a stale operator,
    but cannot find a candidate that became valid without its endpoints
    changing; the differential test covers that case.
    """
    while True:
        best = _select(cache)
        if best is None:
            return None
        _, x, y, _sub = best
        refreshed = fresh(x, y)
        if refreshed is None:
            cache.pop((x, y), None)
            continue
        if refreshed != cache[(x, y)]:
            cache[(x, y)] = refreshed
            continue
        return best


def run_fes_incremental(
    score: Score,
    pdag: PDAG,
    candidate_edges: frozenset[tuple[int, int]] | None = None,
    max_t_size: int | None = None,
    empty_t_gate: bool = False,
) -> tuple[PDAG, bool]:
    """Run the forward phase incrementally; see :func:`~fedcausal.search.ges.run_fes`."""
    current = pdag.copy()
    n = current.num_nodes
    cache: _Cache = {}

    def rescore(pairs: Iterable[tuple[int, int]]) -> None:
        for x, y in pairs:
            cache.pop((x, y), None)
            if x == y:
                continue
            if candidate_edges is not None and (x, y) not in candidate_edges:
                continue
            if x in current.adjacent_set(y):
                continue
            pair = best_insert_for_pair(score, current, x, y, max_t_size, empty_t_gate)
            if pair is not None:
                cache[(x, y)] = pair

    def fresh_insert(x: int, y: int) -> tuple[float, frozenset[int]] | None:
        if x in current.adjacent_set(y):
            return None
        return best_insert_for_pair(score, current, x, y, max_t_size, empty_t_gate)

    rescore(_all_ordered_pairs(n))
    changed = False
    while True:
        best = _pop_best_valid(cache, fresh_insert)
        if best is None:
            break
        _, x, y, t = best
        before = [_sig(current, v) for v in range(n)]
        ops.apply_insert(current, x, y, t)
        current = rebuild_cpdag(current)
        after = [_sig(current, v) for v in range(n)]
        rescore(_incident_ordered_pairs(_changed_nodes(before, after, n), n))
        changed = True
    return current, changed


def _is_delete_pair(pdag: PDAG, x: int, y: int) -> bool:
    """Return whether ``x -> y`` or ``x - y`` is in `pdag`."""
    return y in pdag.children_set(x) or y in pdag.undirected_set(x)


def run_bes_incremental(
    score: Score,
    pdag: PDAG,
    candidate_edges: frozenset[tuple[int, int]] | None = None,
    h_pool: HPool = "nayx",
) -> tuple[PDAG, bool]:
    """Run the backward phase incrementally; see :func:`~fedcausal.search.ges.run_bes`."""
    if h_pool not in H_POOLS:
        raise invalid_h_pool_error(h_pool)

    current = pdag.copy()
    n = current.num_nodes
    cache: _Cache = {}

    def rescore(pairs: Iterable[tuple[int, int]]) -> None:
        for x, y in pairs:
            cache.pop((x, y), None)
            if candidate_edges is not None and (x, y) not in candidate_edges:
                continue
            if not _is_delete_pair(current, x, y):
                continue
            pair = best_delete_for_pair(score, current, x, y, h_pool)
            if pair is not None:
                cache[(x, y)] = pair

    def fresh_delete(x: int, y: int) -> tuple[float, frozenset[int]] | None:
        if not _is_delete_pair(current, x, y):
            return None
        return best_delete_for_pair(score, current, x, y, h_pool)

    rescore(adjacent_ordered_pairs(current))
    changed = False
    while True:
        best = _pop_best_valid(cache, fresh_delete)
        if best is None:
            break
        _, x, y, h = best
        before = [_sig(current, v) for v in range(n)]
        ops.apply_delete(current, x, y, h)
        current = rebuild_cpdag(current)
        after = [_sig(current, v) for v in range(n)]
        rescore(_incident_ordered_pairs(_changed_nodes(before, after, n), n))
        changed = True
    return current, changed
