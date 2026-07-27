"""Consensus topological order by sink-peeling (Puerta et al. 2021, AlphaOrder).

Ported from ``consensusBN/AlphaOrder.java``. The heuristic builds the order
from sinks upward: at each step it picks the node whose removal costs the
fewest edge changes (arc inversions plus moralization additions, summed over
every input DAG, :func:`_compute_next`), prepends it to the order, and peels
it from every working copy (:func:`_peel`).

Determinism: candidate nodes are scanned in ascending index and ties are kept
first-seen, so the lowest index wins -- this replaces the reference implementation's ``nodes``-list
order, which is the same ascending order. No RNG.

``_peel`` is a verbatim port of ``AlphaOrder.removeNode``: it reverses each
out-arc of the peeled node (skipping, on the first pass, any child still
reachable by an alternative directed path, to avoid a cycle on reversal),
moralizes the node's parents with its children, then strips every remaining
incident arc. fedcausal keeps a fixed ``num_nodes``, so "remove the node"
becomes "leave it isolated" -- invisible to every parent/child query, since
an isolated node is nobody's parent or child.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.graph import DAG

__all__ = ["alpha_order"]


def alpha_order(dags: Sequence[DAG]) -> list[int]:
    """Return a consensus topological order over the shared node set.

    Parameters
    ----------
    dags : Sequence[DAG]
        Input DAGs, all over the same ``num_nodes`` node set ``0..p-1``.
        Must be non-empty.

    Returns
    -------
    list[int]
        A permutation of ``0..p-1``: the sink-peeling consensus order, with
        sinks last.

    Raises
    ------
    ValueError
        If `dags` is empty.
    """
    if not dags:
        raise ValueError("alpha_order needs at least one DAG")
    work = [d.copy() for d in dags]
    remaining = list(range(dags[0].num_nodes))
    order: deque[int] = deque()
    while remaining:
        nxt = _compute_next(work, remaining)
        order.appendleft(nxt)
        for g in work:
            _peel(g, nxt)
        remaining.remove(nxt)
    return list(order)


def _compute_next(work: list[DAG], remaining: list[int]) -> int:
    """Return the remaining node whose peel costs the fewest edge changes.

    Parameters
    ----------
    work : list[DAG]
        The current working copies (already peeled of earlier picks).
    remaining : list[int]
        The not-yet-ordered node indices, ascending.

    Returns
    -------
    int
        The lowest-cost node; ties resolved to the lowest index.
    """
    best_node = remaining[0]
    best_cost: int | None = None
    for i in remaining:
        cost = 0
        for g in work:
            children = g.children(i)
            cost += len(children) - 1
            pa_i = g.parents(i)
            inserted: set[frozenset[int]] = set()
            for child in children:
                for p in pa_i:
                    if child not in g.adjacent(p):
                        cost += 1
                for c in g.parents(child):
                    if c != i and i not in g.adjacent(c):
                        key = frozenset((c, i))
                        if key not in inserted:
                            cost += 1
                            inserted.add(key)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_node = i
    return best_node


def _peel(g: DAG, alpha: int) -> None:
    """Reverse `alpha`'s out-arcs with moralization, then isolate `alpha`.

    Parameters
    ----------
    g : DAG
        Working graph, mutated in place.
    alpha : int
        The node being peeled.
    """
    children = sorted(g.children(alpha))
    while children:
        idx = 0
        while True:
            if idx >= len(children):
                child = children[0]
                g.remove_edge(alpha, child)
                break
            child = children[idx]
            idx += 1
            g.remove_edge(alpha, child)
            if g.has_path(alpha, child):
                g.add_edge(alpha, child)  # keep it; try the next child
                continue
            break

        pa_alpha = sorted(g.parents(alpha))
        pa_child = sorted(g.parents(child))  # alpha already removed as a parent
        g.add_edge(child, alpha)  # reverse: child -> alpha
        for p in pa_alpha:
            if child not in g.adjacent(p):
                g.add_edge(p, child)
        for p in pa_child:
            if p != alpha and alpha not in g.adjacent(p):
                g.add_edge(p, alpha)
        children.remove(child)

    # "removeNode": strip every remaining incident arc, leaving alpha isolated.
    for p in list(g.parents(alpha)):
        g.remove_edge(p, alpha)
    for c in list(g.children(alpha)):
        g.remove_edge(alpha, c)
