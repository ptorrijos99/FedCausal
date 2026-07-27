"""Undirected graph utilities: moralization, triangulation, cliques, treewidth.

This module ports the junction-tree helpers from the reference implementation
(``org.albacete.simd.utils.Utils``): moralizing a DAG into an undirected moral
graph, triangulating it via maximum cardinality search (MCS) and the Tarjan
and Yannakakis (1984) fill-in algorithm, extracting its maximal cliques, and
computing treewidth from those cliques.

All undirected graphs here are represented as symmetric, hollow (zero
diagonal) boolean adjacency matrices (``np.ndarray``), consistent with
:class:`fedcausal.graph.dag.DAG`'s ``adjacency()`` convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fedcausal.graph.dag import DAG

__all__ = [
    "moralize",
    "mcs_order",
    "fill_in",
    "maximal_cliques",
    "treewidth_java",
    "treewidth",
]


def moralize(dag: DAG) -> np.ndarray:
    """Moralize a DAG into an undirected moral graph.

    The moral graph consists of the DAG's skeleton (every directed edge
    becomes undirected) plus an edge between every pair of nodes that share
    a common child (i.e. every pair of co-parents), mirroring the reference implementation
    reference's ``Utils.moralize``.

    Parameters
    ----------
    dag : DAG
        The directed acyclic graph to moralize.

    Returns
    -------
    np.ndarray
        A symmetric, hollow boolean adjacency matrix of shape
        ``(dag.num_nodes, dag.num_nodes)`` representing the moral graph.
    """
    directed = dag.adjacency()
    moral: np.ndarray = directed | directed.T
    for v in range(dag.num_nodes):
        parents = sorted(dag.parents(v))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                p, q = parents[i], parents[j]
                moral[p, q] = True
                moral[q, p] = True
    return moral


def mcs_order(adj: np.ndarray) -> list[int]:
    """Compute a maximum cardinality search (MCS) ordering.

    Repeatedly selects the unnumbered node adjacent to the most already
    numbered nodes, breaking ties by choosing the lowest node index. This
    mirrors the reference implementation's ``Utils.getMaximumCardinalityOrdering``,
    made deterministic per the task's tie-breaking convention (the reference implementation
    priority-queue implementation does not guarantee a specific tie order).

    Parameters
    ----------
    adj : np.ndarray
        Symmetric boolean adjacency matrix of an undirected graph.

    Returns
    -------
    list[int]
        A permutation of ``range(adj.shape[0])``: the MCS visitation order.
    """
    n = adj.shape[0]
    numbered = [False] * n
    cardinality = [0] * n
    order: list[int] = []
    for _ in range(n):
        best = -1
        best_cardinality = -1
        for v in range(n):
            if not numbered[v] and cardinality[v] > best_cardinality:
                best_cardinality = cardinality[v]
                best = v
        order.append(best)
        numbered[best] = True
        for w in range(n):
            if adj[best, w] and not numbered[w]:
                cardinality[w] += 1
    return order


def fill_in(adj: np.ndarray, order: list[int]) -> np.ndarray:
    """Triangulate a graph using the Tarjan and Yannakakis (1984) fill-in.

    Processes nodes in reverse of `order`. For each node `v`, the set of its
    neighbours that appear earlier in `order` is computed, and every
    non-adjacent pair within that set is connected ("filled in"). This
    mirrors the reference implementation's ``Utils.fillIn``.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric boolean adjacency matrix of an undirected graph (typically
        a moral graph).
    order : list[int]
        A maximum cardinality ordering of `adj`, as returned by
        :func:`mcs_order`.

    Returns
    -------
    np.ndarray
        A new symmetric boolean adjacency matrix: a chordal (triangulated)
        supergraph of `adj`. The input `adj` is not modified.
    """
    n = len(order)
    filled = adj.copy()
    for i in range(n - 1, -1, -1):
        v = order[i]
        neighbors = [order[j] for j in range(i) if filled[v, order[j]]]
        for a in range(len(neighbors)):
            w = neighbors[a]
            for b in range(a + 1, len(neighbors)):
                x = neighbors[b]
                if not filled[w, x]:
                    filled[w, x] = True
                    filled[x, w] = True
    return filled


def maximal_cliques(adj: np.ndarray, order: list[int]) -> list[frozenset[int]]:
    """Extract the maximal cliques of a chordal graph.

    For each node `v` (processed in reverse of `order`), a candidate clique
    is formed from `v` plus every neighbour of `v` that appears earlier in
    `order`. On a chordal graph (e.g. the output of :func:`fill_in` with a
    matching ordering), these candidates are exactly the graph's cliques;
    candidates that are subsets of another candidate are discarded, leaving
    only the maximal cliques. This mirrors the reference implementation's
    ``Utils.getCliques``.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric boolean adjacency matrix of a chordal (triangulated)
        undirected graph.
    order : list[int]
        A maximum cardinality ordering of `adj`, as returned by
        :func:`mcs_order`.

    Returns
    -------
    list[frozenset[int]]
        The maximal cliques of the graph. No returned clique is a subset of
        another.
    """
    n = len(order)
    candidates: list[frozenset[int]] = []
    for i in range(n - 1, -1, -1):
        v = order[i]
        clique = {v}
        for j in range(i):
            w = order[j]
            if adj[v, w]:
                clique.add(w)
        candidates.append(frozenset(clique))

    # Sort by size descending; Python's sort is stable, so ties keep their
    # original (reverse-MCS) relative order.
    candidates.sort(key=len, reverse=True)

    kept: list[frozenset[int] | None] = list(candidates)
    for i in range(len(kept)):
        v1 = kept[i]
        if v1 is None:
            continue
        for j in range(i + 1, len(kept)):
            v2 = kept[j]
            if v2 is not None and v1 >= v2:
                kept[j] = None

    return [c for c in kept if c is not None]


def treewidth_java(dag: DAG) -> int:
    """Compute the treewidth of a DAG's moral graph, reference-style.

    Moralizes `dag`, triangulates the moral graph (MCS ordering followed by
    fill-in), recomputes an MCS ordering on the triangulated graph, and
    returns the size of its largest maximal clique. This mirrors the reference implementation
    reference's ``Utils.getTreeWidth`` **exactly**, including its
    off-by-one convention: it returns the maximum clique **size**, not
    ``size - 1``. Use :func:`treewidth` for the standard graph-theoretic
    definition.

    Parameters
    ----------
    dag : DAG
        The directed acyclic graph whose moral graph's treewidth is sought.

    Returns
    -------
    int
        The size (node count) of the largest maximal clique in the
        triangulated moral graph, or ``0`` if `dag` has no nodes.
    """
    moral = moralize(dag)
    order = mcs_order(moral)
    triangulated = fill_in(moral, order)
    order2 = mcs_order(triangulated)
    cliques = maximal_cliques(triangulated, order2)
    return max((len(c) for c in cliques), default=0)


def treewidth(dag: DAG) -> int:
    """Compute the treewidth of a DAG's moral graph.

    Standard graph-theoretic treewidth: the size of the largest maximal
    clique in a triangulation of the moral graph, minus one. This is
    :func:`treewidth_java` minus one; see that function's docstring for the
    fidelity rationale behind the two variants.

    Parameters
    ----------
    dag : DAG
        The directed acyclic graph whose moral graph's treewidth is sought.

    Returns
    -------
    int
        The standard treewidth (``treewidth_java(dag) - 1``).
    """
    return treewidth_java(dag) - 1
