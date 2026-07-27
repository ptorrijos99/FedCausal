"""Incremental FES/BES engines: cache per-pair operator scores across steps.

The full-scan engines in :mod:`fedcausal.search.ges` re-score *every* candidate
pair on *every* step. Because the BDeu score is memoized, the repeated
arithmetic is cheap, but the per-step Python-level rescan (validity tests,
``na_yx``, subset enumeration, semi-directed-path checks) is not -- it dominates
GES on larger graphs (see the profile in ``docs/benchmarks.md``). This module
keeps a persistent cache of the best improving operator per candidate pair and,
after each applied operator, re-scores only the pairs whose neighbourhood
changed. The original Java implementation has an equivalent path but does not use it by
default, so this is an optimisation over the faithful full scan rather than a
port of it.

**Exactness vs the full scan.** A candidate's score *delta* depends only on the
neighbourhoods of its two endpoints, so invalidating pairs incident to any node
whose CPDAG neighbourhood changed keeps every cached delta correct. Operator
*validity* is not fully local, though: :func:`~fedcausal.search.gesops.is_clique`
inspects edges among an endpoint's neighbours, and
:func:`~fedcausal.search.gesops.is_semidirected_blocked` traverses the whole
graph. So this incremental invalidation is not *proven* identical to the full
scan on every conceivable graph. It is therefore validated empirically:
``tests/search/test_incremental.py`` asserts the incremental and full-scan
engines produce byte-identical CPDAGs on every benchmark network. The full scan
(:mod:`fedcausal.search.ges`) stays the guaranteed-exact reference; the GES
entry point selects this engine via ``incremental=True``.
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

# A node's CPDAG-local signature: (parents, children, undirected neighbours).
# Two signatures are equal iff the node's incident edges and their orientations
# are unchanged, which is exactly the condition under which every candidate
# score touching that node is unchanged.
_Sig = tuple[frozenset[int], frozenset[int], frozenset[int]]
_Cache = dict[tuple[int, int], tuple[float, frozenset[int]]]


def _sig(pdag: PDAG, v: int) -> _Sig:
    """Return node `v`'s immutable neighbourhood signature (parents/children/undirected)."""
    return (
        frozenset(pdag.parents_set(v)),
        frozenset(pdag.children_set(v)),
        frozenset(pdag.undirected_set(v)),
    )


def _changed_nodes(before: list[_Sig], after: list[_Sig], n: int) -> set[int]:
    """Return the nodes whose signature differs between two graph states."""
    return {v for v in range(n) if before[v] != after[v]}


def _incident_ordered_pairs(changed: set[int], n: int) -> Iterator[tuple[int, int]]:
    """Yield every ordered pair with an endpoint in `changed`, each once.

    These are exactly the pairs whose cached score/validity may have gone stale
    after an edit that changed the signatures of `changed`.
    """
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
    """Yield every ordered pair ``(x, y)`` with ``x != y`` (the initial scan)."""
    for x in range(n):
        for y in range(n):
            if x != y:
                yield (x, y)


def _select(cache: _Cache) -> tuple[float, int, int, frozenset[int]] | None:
    """Return the best cached operator ``(delta, x, y, subset)`` with the full-scan tie rule.

    Iterates candidates in ascending ``(x, y)`` order and keeps an incumbent only
    when a later one beats it by more than :data:`~fedcausal.search._candidates.EPS` --
    reproducing the full scan's first-seen tie-break exactly, so the incremental
    engine selects the same operator the full scan would at each step. `None`
    when no improving candidate remains.
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
    """Select the best candidate, re-scoring it against the current graph first.

    Operator *validity* is not fully local (a semi-directed-path or clique change
    can invalidate a candidate whose endpoints did not change), so a cached entry
    can be stale-high: still in the cache, but no longer valid or no longer the
    best `T`/`H`. Before committing to the current maximum this re-scores it with
    `fresh`; if the fresh value is gone it drops the entry, if it differs it
    corrects the entry, and in either case it re-selects. It returns only once the
    maximum equals its own fresh re-score, so the engine never applies a stale or
    invalid operator. (This corrects stale-*high* entries; the empirical
    differential test guards against stale-*low* misses, which do not arise on the
    benchmark networks -- see the module docstring.)
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
    """Incremental FES: same contract as :func:`~fedcausal.search.ges.run_fes`.

    Scores every non-adjacent ordered pair once, then after each applied Insert
    re-scores only the pairs incident to a node whose signature changed, keeping
    the cache in step with what a full rescan would hold. Returns the closed
    CPDAG and whether any Insert was applied. `pdag` is not mutated.
    """
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
    """Return whether ``(x, y)`` is an eligible Delete pair (``x -> y`` or ``x - y``).

    Matches :func:`~fedcausal.search._candidates.adjacent_ordered_pairs`: a directed
    edge contributes only its forward ordering, an undirected edge both.
    """
    return y in pdag.children_set(x) or y in pdag.undirected_set(x)


def run_bes_incremental(
    score: Score,
    pdag: PDAG,
    candidate_edges: frozenset[tuple[int, int]] | None = None,
    h_pool: HPool = "nayx",
) -> tuple[PDAG, bool]:
    """Incremental BES: same contract as :func:`~fedcausal.search.ges.run_bes`.

    Scores every eligible Delete pair once, then after each applied Delete
    re-scores only the pairs incident to a changed node (adding pairs newly made
    eligible by a reorientation, dropping those whose edge was removed). Returns
    the closed CPDAG and whether any Delete was applied. `pdag` is not mutated.
    """
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
