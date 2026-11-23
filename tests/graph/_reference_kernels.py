"""Graph kernels as they were before the sparse rewrite, the reference for the tests."""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["changed_pairs", "consistent_extension", "cpdag_of_dag", "semidirected_reach"]


@njit(cache=False)
def _neighbour_lists(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return CSR lists of each node's neighbours in `adj`, by any mark, in index order."""
    n = adj.shape[0]
    indptr = np.zeros(n + 1, dtype=np.int64)
    for x in range(n):
        for y in range(n):
            if adj[x, y] or adj[y, x]:
                indptr[x + 1] += 1
    for x in range(n):
        indptr[x + 1] += indptr[x]
    nbr = np.empty(indptr[n], dtype=np.int64)
    fill = indptr[:-1].copy()
    for x in range(n):
        for y in range(n):
            if adj[x, y] or adj[y, x]:
                nbr[fill[x]] = y
                fill[x] += 1
    return indptr, nbr


@njit(cache=False)
def consistent_extension(adj: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a DAG extending the PDAG `adj` (Dor & Tarsi, 1992), and whether one exists.

    Repeatedly removes a sink `x`: a node with no outgoing directed edge whose
    undirected neighbours are each adjacent to every other neighbour of `x`,
    orienting its undirected edges into it. The sink removed is always the
    lowest-numbered one. `adj` is not modified.
    """
    n = adj.shape[0]
    dag = np.zeros((n, n), dtype=np.uint8)
    removed = np.zeros(n, dtype=np.bool_)
    indptr, nbr = _neighbour_lists(adj)
    # Outgoing directed edges of each node, kept up to date as sinks go.
    out = np.zeros(n, dtype=np.int64)
    for x in range(n):
        for a in range(indptr[x], indptr[x + 1]):
            y = nbr[a]
            if adj[x, y] and not adj[y, x]:
                out[x] += 1
    for _ in range(n):
        sink = -1
        for x in range(n):
            if removed[x] or out[x] > 0:
                continue
            ok = True
            for a in range(indptr[x], indptr[x + 1]):
                y = nbr[a]
                if removed[y] or not adj[x, y]:  # Gone, or a parent: no condition.
                    continue
                for b in range(indptr[x], indptr[x + 1]):
                    z = nbr[b]
                    if z != y and not removed[z] and not (adj[y, z] or adj[z, y]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sink = x
                break
        if sink < 0:
            return dag, False
        for a in range(indptr[sink], indptr[sink + 1]):
            y = nbr[a]
            if removed[y]:
                continue
            if adj[y, sink]:
                dag[y, sink] = 1
                if not adj[sink, y]:
                    out[y] -= 1
        removed[sink] = True
    return dag, True


@njit(cache=False)
def cpdag_of_dag(dag: np.ndarray) -> np.ndarray:
    """Return the CPDAG of `dag`, by Chickering's (1995) edge labelling."""
    n = dag.shape[0]
    # Parents of each node in index order, and children for the topological sort.
    pptr = np.zeros(n + 1, dtype=np.int64)
    cptr = np.zeros(n + 1, dtype=np.int64)
    for u in range(n):
        for v in range(n):
            if dag[u, v]:
                pptr[v + 1] += 1
                cptr[u + 1] += 1
    for v in range(n):
        pptr[v + 1] += pptr[v]
        cptr[v + 1] += cptr[v]
    pa = np.empty(pptr[n], dtype=np.int64)
    ch = np.empty(cptr[n], dtype=np.int64)
    pfill = pptr[:-1].copy()
    cfill = cptr[:-1].copy()
    for u in range(n):
        for v in range(n):
            if dag[u, v]:
                pa[pfill[v]] = u
                pfill[v] += 1
                ch[cfill[u]] = v
                cfill[u] += 1

    indegree = np.empty(n, dtype=np.int64)
    for v in range(n):
        indegree[v] = pptr[v + 1] - pptr[v]
    order = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for v in range(n):
        if indegree[v] == 0:
            order[tail] = v
            tail += 1
    while head < tail:
        u = order[head]
        head += 1
        for c in range(cptr[u], cptr[u + 1]):
            v = ch[c]
            indegree[v] -= 1
            if indegree[v] == 0:
                order[tail] = v
                tail += 1
    position = np.empty(n, dtype=np.int64)
    for i in range(n):
        position[order[i]] = i

    # 0 unlabelled, 1 compelled, 2 reversible
    label = np.zeros((n, n), dtype=np.int8)
    ranked = np.empty(n, dtype=np.int64)
    for y in order:
        lo, hi = pptr[y], pptr[y + 1]
        # Parents of y, latest in the order first: the order edges are labelled in.
        k = hi - lo
        for i in range(k):
            ranked[i] = pa[lo + i]
        keys = np.empty(k, dtype=np.int64)
        for i in range(k):
            keys[i] = -position[ranked[i]]
        idx = np.argsort(keys)
        for e in range(k):
            x = ranked[idx[e]]
            if label[x, y] != 0:
                continue
            settled = False
            for w_i in range(pptr[x], pptr[x + 1]):
                w = pa[w_i]
                if label[w, x] != 1:
                    continue
                if not dag[w, y]:
                    for z_i in range(lo, hi):
                        label[pa[z_i], y] = 1
                    settled = True
                    break
                label[w, y] = 1
            if settled:
                continue
            forced = False
            for z_i in range(lo, hi):
                z = pa[z_i]
                if z != x and not dag[z, x]:
                    forced = True
                    break
            value = 1 if forced else 2
            label[x, y] = value
            for z_i in range(lo, hi):
                z = pa[z_i]
                if label[z, y] == 0:
                    label[z, y] = value
    out = np.zeros((n, n), dtype=np.uint8)
    for v in range(n):
        for p in range(pptr[v], pptr[v + 1]):
            u = pa[p]
            out[u, v] = 1
            if label[u, v] == 2:
                out[v, u] = 1
    return out


@njit(cache=False)
def changed_pairs(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Return the pairs ``(u, v)``, ``u < v``, whose marks differ between two adjacency matrices."""
    n = old.shape[0]
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if old[u, v] != new[u, v] or old[v, u] != new[v, u]:
                count += 1
    pairs = np.empty((count, 2), dtype=np.int64)
    k = 0
    for u in range(n):
        for v in range(u + 1, n):
            if old[u, v] != new[u, v] or old[v, u] != new[v, u]:
                pairs[k, 0] = u
                pairs[k, 1] = v
                k += 1
    return pairs


@njit(cache=False)
def semidirected_reach(
    indptr: np.ndarray, indices: np.ndarray, source: int, avoid: np.ndarray
) -> np.ndarray:
    """Return which nodes a semi-directed path from `source` reaches over CSR successors.

    Nodes flagged in `avoid` are reached but not passed through.
    """
    n = indptr.shape[0] - 1
    blocked = np.zeros(n, dtype=np.bool_)
    for v in avoid:
        blocked[v] = True
    reached = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int32)
    stack[0] = source
    top = 1
    reached[source] = True
    while top > 0:
        top -= 1
        v = stack[top]
        for k in range(indptr[v], indptr[v + 1]):
            w = indices[k]
            if not reached[w]:
                reached[w] = True
                if not blocked[w]:
                    stack[top] = w
                    top += 1
    return reached
