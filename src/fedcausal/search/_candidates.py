"""Candidate scoring shared by the full-scan and incremental engines.

Both engines score candidates through these functions, so they cannot drift
apart. Candidates are scanned with `x` ascending, then `y`, then subsets in
:func:`subsets` order, and a later candidate replaces the best so far only if
it wins by more than :data:`EPS`.
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

``"nayx"`` (default) draws `H` from `NaYX(x, y)`, as in Chickering (2002),
Definition 13. ``"bayesfl"`` draws it from `T0(x, y)`, as the original Java
implementation does; that pool is disjoint from `NaYX`, so only
``Delete(x, y, {})`` is ever applied. See ``docs/algorithms.md``.
"""

H_POOLS: tuple[HPool, ...] = get_args(HPool)


def invalid_h_pool_error(h_pool: str) -> ValueError:
    """Return the `ValueError` for an unrecognised `h_pool`, for the caller to raise."""
    expected = " or ".join(f"{p!r}" for p in H_POOLS)
    return ValueError(f"Unknown h_pool: {h_pool!r}; expected {expected}.")


def subsets(items: list[int], max_size: int | None = None) -> Iterator[frozenset[int]]:
    """Yield the subsets of `items` by ascending size, up to `max_size`.

    Within a size, subsets follow the order of `items`, so sorted input gives
    a deterministic order.
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
        If `pdag` has no consistent extension, which a valid operator never
        causes.
    """
    return dag_to_cpdag(pdag.to_dag())


def adjacent_ordered_pairs(pdag: PDAG) -> list[tuple[int, int]]:
    """Return every ordered adjacent pair `(x, y)` eligible for Delete, sorted.

    A directed edge `x -> y` gives `(x, y)`; an undirected edge gives both.
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

    `x` and `y` must not be adjacent. Returns `None` if no `T` improves.
    """
    empty: frozenset[int] = frozenset()
    na = ops.na_yx(pdag, x, y)
    if empty_t_gate and not (
        ops.valid_insert(pdag, x, y, empty, na)
        and ops.insert_eval(score, pdag, x, y, empty, na) > EPS
    ):
        return None
    best: tuple[float, frozenset[int]] | None = None
    # valid_insert inlined, to save path checks: blocking is monotone in the
    # conditioning set and na is inside every na | t, so once na alone blocks
    # every path, every na | t does too.
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

    `h_pool` must already be valid. Returns `None` if no `H` improves.
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
