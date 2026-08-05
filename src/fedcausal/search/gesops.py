r"""GES Insert and Delete operators: candidate sets, validity tests, score deltas.

Implements the Insert and Delete halves of Chickering's (2002) Greedy
Equivalence Search, Definitions 12 and 13. The search loops that drive these
primitives live in :mod:`fedcausal.search.ges`.

Conventions:

* ``NaYX(x, y)`` (:func:`na_yx`) is the set of undirected neighbours of `y` that
  are also adjacent to `x`, by any edge kind.
* `T` ranges over the undirected neighbours of `y` *not* adjacent to `x`;
  :func:`t0_neighbors` computes that superset.
* Insert `(x, y, T)` is valid iff ``NaYX(x, y) | T`` is a clique and every
  semi-directed path from `y` to `x` is blocked by a node of it
  (:func:`valid_insert`).
* Delete `(x, y, H)` is valid iff ``NaYX(x, y) \ H`` is a clique
  (:func:`valid_delete`), with `H` ranging over subsets of ``NaYX(x, y)``.

:func:`is_semidirected_blocked` takes ``(pdag, y, x, cond)`` -- path *source*
first, then target. Read it that way rather than by position.

Two behaviours look like bugs and are not. Both :func:`apply_insert` and
:func:`apply_delete` *force-flip* edges on the `y` side: for every `w` in the
operator's set, whatever edge currently sits between `w` and `y` is replaced
outright, even one pointing the wrong way. The `x` side of :func:`apply_delete`
is guarded instead, and only flips `x - w` when no directed edge already exists
either way. Both reproduce the original Java implementation deliberately; see
``docs/algorithms.md``.
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

# Below this node count the pure-Python set walk beats numba's per-call overhead;
# at or above it the JIT'd CSR DFS wins (~4x, see the perf-spike note). Correctness
# is independent of the threshold -- both paths return the identical boolean.
_NUMBA_MIN_NODES = 50

_REL_TOL = 1e-11
"""Relative tolerance below which a score difference is treated as exactly zero.

Both operator evaluations subtract two local scores of similar magnitude, so
the result loses most of the significance the inputs had. The summed ``lgamma``
terms behind each score contribute rounding of their own, and on a large
network with wide parent sets the residue reaches tens of ulps: on ``link``,
differences of 7e-12 were observed between families whose true difference is
exactly zero, against scores of magnitude 7e3.

An absolute threshold cannot separate that from a real improvement, because the
noise floor scales with the scores while the threshold does not. Anything below
this relative tolerance is therefore snapped to zero, which keeps the search
from inserting edges that only exist in the rounding, and keeps results
reproducible across platforms whose ``lgamma`` rounds differently.

The bound is many orders of magnitude below any score difference that carries
information: a meaningful edge moves a log-likelihood by at least a fraction of
a nat.
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
    """Return the superset `T` is drawn from: undirected neighbours of `y` not adjacent to `x`.

    Excludes `x` itself, which only matters for adjacent `x, y` -- a case the
    Insert candidate scan never reaches.
    """
    adj_x = pdag.adjacent_set(x)
    return frozenset(w for w in pdag.undirected_set(y) if w != x and w not in adj_x)


def is_clique(pdag: PDAG, nodes: frozenset[int]) -> bool:
    """Return whether every pair in `nodes` is adjacent, by any edge kind.

    The empty set and singletons are trivially cliques.
    """
    ns = sorted(nodes)
    for i in range(len(ns)):
        adj_i = pdag.adjacent_set(ns[i])  # hoisted out of the inner loop: O(k) calls, not O(k^2)
        for j in range(i + 1, len(ns)):
            if ns[j] not in adj_i:
                return False
    return True


def is_semidirected_blocked(pdag: PDAG, y: int, x: int, cond: frozenset[int]) -> bool:
    """Return whether every semi-directed path from `y` to `x` hits `cond`.

    A semi-directed path steps along undirected edges, or along directed edges
    in their forward direction, never against an arrowhead. Arguments are
    ``(pdag, source, target, cond)``.

    `cond` seeds the visited set, so a node in it is never discovered -- and
    if `x` itself is in `cond` the answer is ``True`` (blocked), because the
    traversal never gets the chance to reach it.

    Returns
    -------
    bool
        ``True`` iff no semi-directed path from `y` to `x` avoids `cond`
        entirely (i.e. all such paths are blocked).
    """
    # Optional numba fast path (byte-identical result). Above _NUMBA_MIN_NODES the
    # JIT'd CSR DFS beats this pure-Python walk; below it the set walk is faster.
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
        # Semi-directed successors of v: forward along a directed edge (its
        # children) or along an undirected edge. This is a pure reachability
        # query -- whether x is reachable avoiding `cond` does not depend on the
        # order successors are visited -- so the raw internal sets are iterated
        # directly, with no union or sort, both of which were the dominant cost here.
        # Determinism of the FES/BES *result*
        # comes from the fixed candidate scan order in ges.py, not from this
        # internal traversal, whose only observable output is a boolean.
        for group in (pdag.children_set(v), pdag.undirected_set(v)):
            for w in group:
                if w in visited:
                    continue
                if w == x:
                    return False  # reached x without hitting cond -> not blocked
                visited.add(w)
                stack.append(w)
    return True


def valid_insert(pdag: PDAG, x: int, y: int, t: frozenset[int], na: frozenset[int]) -> bool:
    """Return whether Insert `(x, y, T)` satisfies both validity conditions.

    True iff ``na | t`` is a clique and every semi-directed path from `y` to
    `x` is blocked by it. `na` is :func:`na_yx`, `t` a subset of
    :func:`t0_neighbors`.
    """
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
    """Apply Insert `(x, y, T)` to `pdag` in place: a raw edit, not re-canonicalised.

    Adds `x -> y`, then force-flips every `w` in `t` to `w -> y` -- whatever
    edge was there before, including one pointing the other way. That is
    deliberate; see the module docstring. The caller re-canonicalises.

    Mutates in place and does not roll back on failure. Under this module's
    calling convention the only reachable exception fires before any `w` edit,
    so no partial state is reachable, but a caller that breaks the convention
    could leave one behind.

    Raises
    ------
    ValueError
        From :meth:`PDAG.add_directed` if `x == y` or `x` and `y` are already
        adjacent. A valid Insert only considers non-adjacent pairs, so this
        signals a caller bug.
    KeyError
        From :meth:`PDAG.remove_edge` if some `w` in `t` has no edge to `y`.
        Unreachable in the forward phase; it signals a stale `T`.
    """
    pdag.add_directed(x, y)
    for w in sorted(t):
        if pdag.has_undirected(w, y):
            pdag.orient(w, y)
        else:
            pdag.remove_edge(w, y)
            pdag.add_directed(w, y)


def valid_delete(pdag: PDAG, x: int, y: int, h: frozenset[int], na: frozenset[int]) -> bool:
    r"""Return whether Delete `(x, y, H)` satisfies the validity condition.

    True iff ``na \ h`` is a clique. `na` is :func:`na_yx`, `h` a subset of it.
    """
    return is_clique(pdag, na - h)


def delete_eval(
    score: Score, pdag: PDAG, x: int, y: int, h: frozenset[int], na: frozenset[int]
) -> float:
    """Return the score delta of Delete `(x, y, H)` without applying it.

    The score of `y` without `x` as a parent, minus the score with it:
    ``local_score(y, (na - h) | (Pa(y) - {x}))`` minus the same set plus `x`.
    """
    before = (na - h) | (pdag.parents_set(y) - {x})
    without_x = score.local_score(y, before)
    with_x = score.local_score(y, before | {x})
    return _denoise(without_x - with_x, without_x, with_x)


def apply_delete(pdag: PDAG, x: int, y: int, h: frozenset[int]) -> None:
    """Apply Delete `(x, y, H)` to `pdag` in place: a raw edit, not re-canonicalised.

    Removes the edge between `x` and `y`, then for every `w` in `h`:

    * force-flips whatever edge sits between `y` and `w` into `y -> w`;
    * force-flips `x - w` into `x -> w`, but only when no directed edge
      already exists between them in either direction.

    The asymmetry between the two sides is deliberate; see the module
    docstring. The caller re-canonicalises. Mutates in place, with no rollback
    on failure.

    Raises
    ------
    KeyError
        From :meth:`PDAG.remove_edge` when an edge it expects is absent. Two
        of the three cases are unreachable, since a valid Delete considers
        only adjacent `x, y` and draws `h` from the undirected neighbours of
        `y`. The third -- some `w` in `h` having no edge to `x` -- cannot
        happen when `h` comes from :func:`na_yx`, but is the ordinary case
        under the ``"bayesfl"`` pool, which selects for exactly that. See
        ``docs/algorithms.md``.
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
