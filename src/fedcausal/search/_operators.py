r"""The Insert and Delete operators of Greedy Equivalence Search (Chickering, 2002).

* Insert `(x, y, T)` adds ``x -> y`` between non-adjacent nodes and orients
  ``t - y`` as ``t -> y`` for each `t` in `T`, a subset of the undirected
  neighbours of `y` not adjacent to `x`. It is valid iff ``NaYX(x, y) | T`` is a
  clique and blocks every semi-directed path from `y` to `x`.
* Delete `(x, y, H)` removes the edge between `x` and `y` and orients
  ``y - h`` as ``y -> h`` and ``x - h`` as ``x -> h`` for each `h` in `H`, a
  subset of ``NaYX(x, y)``. It is valid iff ``NaYX(x, y) \ H`` is a clique.

``NaYX(x, y)`` is the set of undirected neighbours of `y` adjacent to `x`.

Operators are evaluated a child `y` at a time. Every `x` with the same
``NaYX(x, y)`` shares its candidate subsets, so one traversal of the graph
answers the path condition for all of them, and their families share the
encoding of the common parents.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from fedcausal.graph._kernels import semidirected_reach

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from fedcausal.graph import PDAG
    from fedcausal.scores import Score

__all__ = [
    "Candidate",
    "apply_delete",
    "apply_insert",
    "deletes_into",
    "inserts_into",
    "is_clique",
]

Candidate = tuple[float, frozenset[int]]
"""An operator's score improvement and its subset (`T` or `H`)."""

_REL_TOL = 1e-10
"""Improvements smaller than this, relative to the scores involved, are rounding.

Each improvement is the difference of two sums of ``lgamma`` terms of similar
size, so equal families can differ by a few ulps of the score. Treating that
residue as an improvement would add edges that exist only in the rounding.
"""

_NO_NODES = np.zeros(0, dtype=np.int32)
_EMPTY: frozenset[int] = frozenset()


def is_clique(pdag: PDAG, nodes: frozenset[int]) -> bool:
    """Return whether every two nodes of `nodes` are adjacent."""
    ordered = sorted(nodes)
    for i, u in enumerate(ordered):
        adj_u = pdag.adjacent_set(u)
        for v in ordered[i + 1 :]:
            if v not in adj_u:
                return False
    return True


def semidirected_reachable(pdag: PDAG, source: int, avoid: frozenset[int]) -> npt.NDArray[np.bool_]:
    """Return which nodes a semi-directed path from `source` reaches without passing `avoid`.

    A semi-directed path follows undirected edges, and directed edges in their
    own direction. A node of `avoid` can be reached but not passed through, so
    a node is unblocked from `source` exactly when it is reached and not in
    `avoid`.
    """
    avoid_array = np.fromiter(avoid, dtype=np.int32, count=len(avoid)) if avoid else _NO_NODES
    indptr, indices = pdag.successor_csr()
    reached: npt.NDArray[np.bool_] = semidirected_reach(indptr, indices, source, avoid_array)
    return reached


def _subsets(items: list[int], max_size: int | None) -> Iterator[frozenset[int]]:
    """Yield the subsets of `items` by increasing size, up to `max_size`."""
    largest = len(items) if max_size is None else min(max_size, len(items))
    for size in range(largest + 1):
        for combo in combinations(items, size):
            yield frozenset(combo)


def _cliques(pdag: PDAG, nodes: list[int]) -> list[frozenset[int]]:
    """Return every clique among `nodes`, the empty set included, largest first."""
    found: list[frozenset[int]] = [_EMPTY]

    def extend(clique: frozenset[int], candidates: list[int]) -> None:
        for i, v in enumerate(candidates):
            grown = clique | {v}
            found.append(grown)
            adjacent = pdag.adjacent_set(v)
            extend(grown, [w for w in candidates[i + 1 :] if w in adjacent])

    extend(_EMPTY, nodes)
    found.sort(key=len, reverse=True)
    return found


def _is_noise(delta: float, a: float, b: float) -> bool:
    return abs(delta) <= _REL_TOL * max(abs(a), abs(b), 1.0)


def inserts_into(
    score: Score, pdag: PDAG, y: int, max_t_size: int | None, max_parents: int | None = None
) -> dict[int, Candidate]:
    """Return, for each `x`, the best valid and improving Insert `(x, y, T)`.

    `T` ranges over subsets of at most `max_t_size` nodes, all of them when
    ``None``. Among subsets with the same improvement the smallest is kept.
    Inserts that would give `y` more than `max_parents` parents are skipped.
    Nodes with no improving Insert are left out.
    """
    undirected = pdag.undirected_set(y)
    parents = frozenset(pdag.parents_set(y))
    skeleton = pdag.skeleton_view()
    others = np.flatnonzero(~skeleton[y])
    others = others[others != y]
    # Group the candidate parents by NaYX, their neighbours among y's undirected ones.
    groups: dict[frozenset[int], npt.NDArray[np.intp]]
    if not undirected or others.size == 0:
        groups = {_EMPTY: others}
    else:
        neighbours = np.array(sorted(undirected), dtype=np.int64)
        shared = skeleton[np.ix_(others, neighbours)]
        keys, inverse = np.unique(shared, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        groups = {
            frozenset(neighbours[key].tolist()): others[inverse == g] for g, key in enumerate(keys)
        }

    best: dict[int, Candidate] = {}
    for na, xs in groups.items():
        for t in _subsets(sorted(undirected - na), max_t_size):
            cond = na | t
            if not is_clique(pdag, cond):
                continue
            reached = semidirected_reachable(pdag, y, cond)
            valid = xs[~reached[xs]]
            if valid.size == 0:
                continue
            base = parents | cond
            if max_parents is not None and len(base) >= max_parents:
                continue
            s_without = score.local_score(y, base)
            s_with = np.asarray(score.local_scores_with(y, base, valid), dtype=np.float64)
            delta = s_with - s_without
            tol = _REL_TOL * np.maximum(np.maximum(np.abs(s_with), abs(s_without)), 1.0)
            for k in np.flatnonzero((delta > 0) & (delta > tol)).tolist():
                x, d = int(valid[k]), float(delta[k])
                if x not in best or d > best[x][0]:
                    best[x] = (d, t)
    return best


def deletes_into(score: Score, pdag: PDAG, y: int) -> dict[int, Candidate]:
    """Return, for each `x` adjacent to `y`, the best valid and improving Delete `(x, y, H)`.

    `x` is a parent or an undirected neighbour of `y`. Among subsets with the
    same improvement the smallest is kept. Nodes with no improving Delete are
    left out.
    """
    undirected = pdag.undirected_set(y)
    parents = pdag.parents_set(y)
    best: dict[int, Candidate] = {}
    for x in sorted(parents | undirected):
        na = frozenset(undirected & pdag.adjacent_set(x))
        others = frozenset(parents) - {x}
        # A Delete is valid iff the part of NaYX it keeps is a clique, so the
        # cliques of NaYX are exactly the candidates, and enumerating them is
        # far cheaper than every subset when NaYX is large and sparse.
        for kept in _cliques(pdag, sorted(na)):
            base = kept | others
            s_without = score.local_score(y, base)
            s_with = score.local_score(y, base | {x})
            delta = s_without - s_with
            if delta <= 0 or _is_noise(delta, s_with, s_without):
                continue
            if x not in best or delta > best[x][0]:
                best[x] = (delta, na - kept)
    return best


def apply_insert(pdag: PDAG, x: int, y: int, t: frozenset[int]) -> None:
    """Apply Insert `(x, y, T)` to `pdag` in place, without re-completing it."""
    pdag.add_directed(x, y)
    for w in t:
        pdag.orient(w, y)


def apply_delete(pdag: PDAG, x: int, y: int, h: frozenset[int]) -> None:
    """Apply Delete `(x, y, H)` to `pdag` in place, without re-completing it."""
    pdag.remove_edge(x, y)
    for w in h:
        pdag.orient(y, w)
        if pdag.has_undirected(x, w):
            pdag.orient(x, w)
