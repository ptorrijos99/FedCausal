"""Candidate scanning and scoring shared by the full-scan and incremental engines.

Both :mod:`fedcausal.search.ges` and :mod:`fedcausal.search.incremental` score a
candidate pair through the functions here, so the two engines cannot drift in
their tie-breaking or validity rules. Nothing in this module is public API.

Determinism is a hard requirement: candidate ``(x, y, T)`` / ``(x, y, H)``
triples are scanned in a fixed order -- `x` ascending, then `y` ascending, then
subsets in :func:`subsets` order -- and ties are kept first-seen, a later
candidate replacing the incumbent only when it wins by more than :data:`EPS`.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Literal, get_args

from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.search import gesops as ops

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fedcausal.graph.pdag import PDAG
    from fedcausal.scores import Score

__all__ = [
    "EPS",
    "HPool",
    "H_POOLS",
    "invalid_h_pool_error",
    "subsets",
    "rebuild_cpdag",
    "adjacent_ordered_pairs",
    "h_candidates",
    "best_insert_for_pair",
    "best_delete_for_pair",
]

EPS = 1e-12

HPool = Literal["nayx", "bayesfl"]
"""Which candidate pool `H` is drawn from during the backward phase.

``"nayx"`` (default) draws `H` from `NaYX(x, y)`, matching Chickering (2002)
Definition 13. ``"bayesfl"`` draws it from `T0(x, y)` for compatibility with the
original Java implementation, where the pool is disjoint from `NaYX` and the search
therefore only ever applies ``Delete(x, y, empty)``. See ``docs/algorithms.md``.
"""

H_POOLS: tuple[HPool, ...] = get_args(HPool)


def invalid_h_pool_error(h_pool: str) -> ValueError:
    """Return (not raise) the `ValueError` for an unrecognised `h_pool`.

    Returned rather than raised so call sites read ``raise
    invalid_h_pool_error(...)``, keeping the ``raise`` visible where it happens.
    """
    expected = " or ".join(f"{p!r}" for p in H_POOLS)
    return ValueError(f"Unknown h_pool: {h_pool!r}; expected {expected}.")


def subsets(items: list[int], max_size: int | None = None) -> Iterator[frozenset[int]]:
    """Yield every subset of `items`, ascending size then lexicographic.

    Callers pass `items` pre-sorted so the emitted order is deterministic.
    Truncating at `max_size` only drops trailing size classes, so a bounded
    run's order is a prefix of the unbounded one.
    """
    largest = len(items) if max_size is None else min(max_size, len(items))
    for size in range(largest + 1):
        for combo in combinations(items, size):
            yield frozenset(combo)


def rebuild_cpdag(pdag: PDAG) -> PDAG:
    """Re-canonicalise `pdag` to a CPDAG via a consistent DAG extension.

    Raises
    ------
    ValueError
        Propagated from :meth:`~fedcausal.graph.pdag.PDAG.to_dag` if `pdag` has
        no consistent extension. A valid operator applied to a CPDAG always
        leaves one, so this firing means a bug upstream; it is deliberately not
        caught.
    """
    return dag_to_cpdag(pdag.to_dag())


def adjacent_ordered_pairs(pdag: PDAG) -> list[tuple[int, int]]:
    """Return every ordered adjacent pair `(x, y)` eligible for Delete, sorted.

    A directed edge `x -> y` contributes only `(x, y)`; an undirected edge
    contributes both orderings, since either endpoint may play the role of `x`.
    """
    pairs = set(pdag.directed_edges())
    for u, v in pdag.undirected_edges():
        pairs.add((u, v))
        pairs.add((v, u))
    return sorted(pairs)


def h_candidates(pdag: PDAG, x: int, y: int, h_pool: HPool) -> frozenset[int]:
    """Return the superset `H` is drawn from for the Delete pair `(x, y)`.

    Raises
    ------
    ValueError
        If `h_pool` is not a recognised pool.
    """
    if h_pool == "nayx":
        return ops.na_yx(pdag, x, y)
    if h_pool == "bayesfl":
        return ops.t0_neighbors(pdag, x, y)
    raise invalid_h_pool_error(h_pool)


def best_insert_for_pair(
    score: Score,
    pdag: PDAG,
    x: int,
    y: int,
    max_t_size: int | None,
    empty_t_gate: bool,
) -> tuple[float, frozenset[int]] | None:
    """Return the best valid improving Insert `(delta, T)` for the pair `(x, y)`.

    `x` and `y` must be non-adjacent -- the caller checks. Returns `None` if no
    `T` yields an improving valid Insert.
    """
    empty: frozenset[int] = frozenset()
    na = ops.na_yx(pdag, x, y)
    if empty_t_gate and not (
        ops.valid_insert(pdag, x, y, empty, na)
        and ops.insert_eval(score, pdag, x, y, empty, na) > EPS
    ):
        return None
    best: tuple[float, frozenset[int]] | None = None
    # Monotonicity shortcut for the dominant cost, `is_semidirected_blocked`:
    # the check is monotone increasing in the conditioning set (a larger seed
    # only removes reachable nodes) and `na` is a subset of every `na | t`. So
    # once the na-only path check is blocked, every `na | t` is blocked too and
    # the traversal can be skipped for the remaining subsets. This inlines
    # `valid_insert`'s two checks -- clique first, then blocked, order unchanged
    # -- to reuse the na-only result, computed lazily on the first `T` passing
    # the clique test so pairs failing `is_clique` throughout pay nothing extra.
    na_blocked: bool | None = None
    for t in subsets(sorted(ops.t0_neighbors(pdag, x, y)), max_t_size):
        nat = na | t
        if not ops.is_clique(pdag, nat):
            continue
        if na_blocked is None:
            na_blocked = ops.is_semidirected_blocked(pdag, y, x, na)
        if not (na_blocked or (nat != na and ops.is_semidirected_blocked(pdag, y, x, nat))):
            continue
        delta = ops.insert_eval(score, pdag, x, y, t, na)
        if delta > EPS and (best is None or delta > best[0] + EPS):
            best = (delta, t)
    return best


def best_delete_for_pair(
    score: Score,
    pdag: PDAG,
    x: int,
    y: int,
    h_pool: HPool,
) -> tuple[float, frozenset[int]] | None:
    """Return the best valid improving Delete `(delta, H)` for the pair `(x, y)`.

    Assumes `h_pool` is already validated. Returns `None` if no `H` yields an
    improving valid Delete.
    """
    na = ops.na_yx(pdag, x, y)
    best: tuple[float, frozenset[int]] | None = None
    for h in subsets(sorted(h_candidates(pdag, x, y, h_pool))):
        if not ops.valid_delete(pdag, x, y, h, na):
            continue
        delta = ops.delete_eval(score, pdag, x, y, h, na)
        if delta > EPS and (best is None or delta > best[0] + EPS):
            best = (delta, h)
    return best
