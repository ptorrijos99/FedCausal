"""Compiled kernels for the consensus order and the transform into it.

Graphs are ``uint8`` adjacency matrices: ``A[u, v] = 1`` is the arc ``u -> v``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["alpha_order_kernel", "beta_to_alpha_kernel"]


@njit(cache=True)
def _adjacent(g: np.ndarray, u: int, v: int) -> bool:
    return g[u, v] != 0 or g[v, u] != 0


@njit(cache=True)
def _has_path(g: np.ndarray, source: int, target: int) -> bool:
    n = g.shape[0]
    seen = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    top = 0
    for w in range(n):
        if g[source, w] and not seen[w]:
            seen[w] = True
            stack[top] = w
            top += 1
    while top > 0:
        top -= 1
        v = stack[top]
        if v == target:
            return True
        for w in range(n):
            if g[v, w] and not seen[w]:
                seen[w] = True
                stack[top] = w
                top += 1
    return False


@njit(cache=True)
def _peel_cost(g: np.ndarray, i: int, mark: np.ndarray) -> int:
    """Edge changes needed to make `i` a sink of `g`: reversals plus the moral edges they force."""
    n = g.shape[0]
    mark[:] = False
    n_children = 0
    cost = 0
    for child in range(n):
        if not g[i, child]:
            continue
        n_children += 1
        for p in range(n):
            if g[p, i] and not _adjacent(g, p, child):
                cost += 1
        for c in range(n):
            if g[c, child] and c != i and not mark[c] and not _adjacent(g, c, i):
                mark[c] = True
                cost += 1
    return cost + n_children


@njit(cache=True)
def _peel(g: np.ndarray, alpha: int) -> None:
    """Reverse `alpha`'s out-arcs, adding the moral edges each reversal needs, then isolate it."""
    n = g.shape[0]
    children = [c for c in range(n) if g[alpha, c]]
    while len(children) > 0:
        # Reverse first an arc whose removal leaves no other path to its head.
        chosen = -1
        for child in children:
            g[alpha, child] = 0
            if not _has_path(g, alpha, child):
                chosen = child
                break
            g[alpha, child] = 1
        if chosen < 0:
            chosen = children[0]
            g[alpha, chosen] = 0
        pa_alpha = [p for p in range(n) if g[p, alpha]]
        pa_child = [p for p in range(n) if g[p, chosen]]
        g[chosen, alpha] = 1
        for p in pa_alpha:
            if not _adjacent(g, p, chosen):
                g[p, chosen] = 1
        for p in pa_child:
            if p != alpha and not _adjacent(g, p, alpha):
                g[p, alpha] = 1
        children.remove(chosen)
    for v in range(n):
        g[v, alpha] = 0
        g[alpha, v] = 0


@njit(cache=True)
def alpha_order_kernel(graphs: np.ndarray) -> np.ndarray:
    """Return the consensus order of the stacked DAGs ``graphs[k]`` (Puerta et al., 2021).

    Built from the sinks up: each step takes the remaining node whose peel
    costs the fewest edge changes summed over every graph (lowest index on
    ties), places it before the nodes already ordered, and peels it from
    every graph.
    """
    work = graphs.copy()
    n_graphs, n, _ = work.shape
    remaining = np.ones(n, dtype=np.bool_)
    mark = np.zeros(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int64)
    for slot in range(n - 1, -1, -1):
        best = -1
        best_cost = 0
        for i in range(n):
            if not remaining[i]:
                continue
            cost = 0
            for k in range(n_graphs):
                cost += _peel_cost(work[k], i, mark)
            if best < 0 or cost < best_cost:
                best = i
                best_cost = cost
        order[slot] = best
        remaining[best] = False
        for k in range(n_graphs):
            _peel(work[k], best)
    return order


@njit(cache=True)
def beta_to_alpha_kernel(dag: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Return a minimal I-map of `dag` whose arcs all go forward in the order given by `rank`.

    First a topological order of `dag` close to the target (``beta``) is
    built by peeling sinks; then nodes are bubbled into the target order by
    adjacent swaps, each swap across an arc reversing it and adding the moral
    edges that keep every independence of the input.
    """
    n = dag.shape[0]
    peeled = np.zeros(n, dtype=np.bool_)
    beta: list[int] = []
    for _ in range(n):
        sink = -1
        for v in range(n):
            if peeled[v]:
                continue
            ready = True
            for c in range(n):
                if dag[v, c] and not peeled[c]:
                    ready = False
                    break
            if ready:
                sink = v
                break
        peeled[sink] = True
        # Before its first child, or the first node the target puts after it.
        i = 0
        while i < len(beta) and not (rank[beta[i]] > rank[sink] or dag[sink, beta[i]]):
            i += 1
        beta.insert(i, sink)

    out = dag.copy()
    aux: list[int] = [beta[0]]
    for j in range(1, n):
        aux.append(beta[j])
        i = len(aux) - 1
        while i > 0:
            y = aux[i]
            z = aux[i - 1]
            if rank[z] <= rank[y]:
                break
            if _adjacent(out, y, z):
                pa_z = [p for p in range(n) if out[p, z]]
                pa_y = [p for p in range(n) if out[p, y] and p != z]
                out[z, y] = 0
                out[y, z] = 1
                for p in pa_z:
                    if p != y and not _adjacent(out, p, y):
                        out[p, y] = 1
                for p in pa_y:
                    if not _adjacent(out, p, z):
                        out[p, z] = 1
            aux[i] = z
            aux[i - 1] = y
            i -= 1
    return out
