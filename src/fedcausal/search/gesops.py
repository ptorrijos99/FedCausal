r"""The Insert and Delete operators of GES (Chickering, 2002, Definitions 12 and 13).

* ``NaYX(x, y)`` (:func:`na_yx`): the undirected neighbours of `y` that are
  adjacent to `x`.
* `T` is drawn from the undirected neighbours of `y` not adjacent to `x`
  (:func:`t0_neighbors`).
* Insert `(x, y, T)` is valid iff ``NaYX(x, y) | T`` is a clique and blocks
  every semi-directed path from `y` to `x` (:func:`valid_insert`).
* Delete `(x, y, H)` is valid iff ``NaYX(x, y) \ H`` is a clique
  (:func:`valid_delete`).

Note the argument order of :func:`is_semidirected_blocked`: source `y` first,
then target `x`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedcausal.graph import PDAG
    from fedcausal.scores import Score

from fedcausal.search._blocked_numba import HAVE_NUMBA as _HAVE_NUMBA
from fedcausal.search._blocked_numba import blocked_numba as _blocked_numba

__all__ = [
    "na_yx",
    "t0_neighbors",
    "is_clique",
    "is_semidirected_blocked",
    "valid_insert",
    "insert_eval",
    "apply_insert",
    "valid_delete",
    "delete_eval",
    "apply_delete",
]

# Below this size the set walk beats numba's call overhead. Both give the same answer.
_NUMBA_MIN_NODES = 50

_REL_TOL = 1e-11
"""Score differences smaller than this, relative to the scores, count as zero.

A delta subtracts two scores of similar size built from many ``lgamma`` terms,
so rounding alone can leave a residue of tens of ulps (7e-12 against scores of
7e3 on ``link``). Without this, the search would insert edges that exist only
in the rounding, and results would depend on the platform's ``lgamma``. A real
improvement is many orders of magnitude larger.
"""


def _denoise(delta: float, first: float, second: float) -> float:
    """Return `delta`, or exactly zero when it is indistinguishable from rounding."""
    if abs(delta) <= _REL_TOL * max(abs(first), abs(second), 1.0):
        return 0.0
    return delta


def na_yx(pdag: PDAG, x: int, y: int) -> frozenset[int]:
    """Return `NaYX(x, y)`: undirected neighbours of `y` also adjacent to `x`."""
    adj_x = pdag.adjacent_set(x)
    return frozenset(w for w in pdag.undirected_set(y) if w in adj_x)


def t0_neighbors(pdag: PDAG, x: int, y: int) -> frozenset[int]:
    """Return the undirected neighbours of `y` not adjacent to `x`, the pool for `T`."""
    adj_x = pdag.adjacent_set(x)
    return frozenset(w for w in pdag.undirected_set(y) if w != x and w not in adj_x)


def is_clique(pdag: PDAG, nodes: frozenset[int]) -> bool:
    """Return whether every pair in `nodes` is adjacent, by any edge kind."""
    ns = sorted(nodes)
    for i in range(len(ns)):
        adj_i = pdag.adjacent_set(ns[i])
        for j in range(i + 1, len(ns)):
            if ns[j] not in adj_i:
                return False
    return True


def is_semidirected_blocked(pdag: PDAG, y: int, x: int, cond: frozenset[int]) -> bool:
    """Return whether every semi-directed path from `y` to `x` hits `cond`.

    A semi-directed path follows undirected edges, and directed edges only
    forwards. A path through a node of `cond` is blocked, including one ending
    at `x` when `x` is in `cond`.
    """
    if _HAVE_NUMBA and pdag.num_nodes >= _NUMBA_MIN_NODES:
        return _blocked_numba(pdag, y, x, cond)

    if y in cond:
        return True
    if y == x:
        return False

    stack: list[int] = [y]
    visited: set[int] = set(cond) | {y}
    while stack:
        v = stack.pop()
        # Visiting order does not affect reachability, so no sorting.
        for group in (pdag.children_set(v), pdag.undirected_set(v)):
            for w in group:
                if w in visited:
                    continue
                if w == x:
                    return False
                visited.add(w)
                stack.append(w)
    return True


def valid_insert(pdag: PDAG, x: int, y: int, t: frozenset[int], na: frozenset[int]) -> bool:
    """Return whether ``na | t`` is a clique blocking every semi-directed path `y` to `x`."""
    nat = na | t
    return is_clique(pdag, nat) and is_semidirected_blocked(pdag, y, x, nat)


def insert_eval(
    score: Score, pdag: PDAG, x: int, y: int, t: frozenset[int], na: frozenset[int]
) -> float:
    """Return the score delta of Insert `(x, y, T)` without applying it.

    ``local_score(y, na | t | Pa(y) | {x}) - local_score(y, na | t | Pa(y))``.
    """
    base = na | t | pdag.parents_set(y)
    with_x = score.local_score(y, base | {x})
    without_x = score.local_score(y, base)
    return _denoise(with_x - without_x, with_x, without_x)


def apply_insert(pdag: PDAG, x: int, y: int, t: frozenset[int]) -> None:
    """Apply Insert `(x, y, T)` in place: add `x -> y` and point every `w` in `t` into `y`.

    The result is not a CPDAG until the caller re-canonicalises it.

    Raises
    ------
    ValueError
        If `x` and `y` are equal or already adjacent.
    KeyError
        If some `w` in `t` has no edge to `y`.
    """
    pdag.add_directed(x, y)
    for w in sorted(t):
        if pdag.has_undirected(w, y):
            pdag.orient(w, y)
        else:
            pdag.remove_edge(w, y)
            pdag.add_directed(w, y)


def valid_delete(pdag: PDAG, x: int, y: int, h: frozenset[int], na: frozenset[int]) -> bool:
    """Return whether ``na - h`` is a clique."""
    return is_clique(pdag, na - h)


def delete_eval(
    score: Score, pdag: PDAG, x: int, y: int, h: frozenset[int], na: frozenset[int]
) -> float:
    """Return the score delta of Delete `(x, y, H)` without applying it.

    ``local_score(y, (na - h) | (Pa(y) - {x}))`` minus the same with `x` added.
    """
    before = (na - h) | (pdag.parents_set(y) - {x})
    without_x = score.local_score(y, before)
    with_x = score.local_score(y, before | {x})
    return _denoise(without_x - with_x, without_x, with_x)


def apply_delete(pdag: PDAG, x: int, y: int, h: frozenset[int]) -> None:
    """Apply Delete `(x, y, H)` in place.

    Removes the edge between `x` and `y`; for every `w` in `h`, points the
    `y`-`w` edge to `y -> w`, and an undirected `x - w` to `x -> w`. The
    result is not a CPDAG until the caller re-canonicalises it.

    Raises
    ------
    KeyError
        If `x` and `y` are not adjacent, or some `w` in `h` is not adjacent to
        both.
    """
    pdag.remove_edge(x, y)
    for w in sorted(h):
        if not pdag.has_directed(w, x) and not pdag.has_directed(x, w):
            if pdag.has_undirected(x, w):
                pdag.orient(x, w)
            else:
                pdag.remove_edge(x, w)
                pdag.add_directed(x, w)
        if pdag.has_undirected(y, w):
            pdag.orient(y, w)
        else:
            pdag.remove_edge(y, w)
            pdag.add_directed(y, w)
