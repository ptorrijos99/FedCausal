"""Partially directed acyclic graph (PDAG) for causal structure learning.

This module provides :class:`PDAG`, a graph over integer node indices
``0..num_nodes-1`` that carries a mixture of *directed* edges ``u -> v`` and
*undirected* edges ``{u, v}``. It is the working representation for CPDAG
(completed partially directed acyclic graph / pattern) construction, Meek-rule
orientation, and GES search elsewhere in ``fedcausal``.

The class mirrors :class:`fedcausal.graph.dag.DAG` conventions: parent/child
sets are stored per node for cheap incremental mutation, undirected neighbours
are stored per node as symmetric sets, and a boolean adjacency matrix is cached
lazily and invalidated on any mutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fedcausal.graph.dag import DAG

__all__ = ["PDAG"]


class PDAG:
    """A partially directed acyclic graph over integer nodes ``0..num_nodes-1``.

    A PDAG carries two kinds of edges: *directed* edges ``u -> v`` and
    *undirected* edges ``{u, v}``. At most one edge (of either kind, in either
    orientation) is stored between any pair of nodes.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph. Nodes are addressed by the integer
        indices ``0, 1, ..., num_nodes - 1``.

    Attributes
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Notes
    -----
    Directed edges are stored via per-node ``_parents`` and ``_children`` sets;
    undirected edges are stored via a per-node ``_undirected`` list of symmetric
    neighbour sets. A boolean adjacency matrix is cached lazily and invalidated
    on any mutation. Unlike :class:`DAG`, ``add_directed`` performs no cycle
    checking -- inconsistency (a directed cycle) is surfaced only when
    :meth:`to_dag` is asked for a consistent extension.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self._parents: list[set[int]] = [set() for _ in range(num_nodes)]
        self._children: list[set[int]] = [set() for _ in range(num_nodes)]
        self._undirected: list[set[int]] = [set() for _ in range(num_nodes)]
        self._amat: np.ndarray | None = None
        # Lazy CSR of semi-directed successors (children ∪ undirected), built by
        # `successor_csr` for the optional numba traversal; invalidated on mutation.
        self._succ_csr: tuple[np.ndarray, np.ndarray] | None = None

    def parents(self, v: int) -> frozenset[int]:
        """Return the parents of node `v`."""
        return frozenset(self._parents[v])

    def children(self, v: int) -> frozenset[int]:
        """Return the children of node `v`."""
        return frozenset(self._children[v])

    def undirected_neighbors(self, v: int) -> frozenset[int]:
        """Return the undirected neighbours of node `v`."""
        return frozenset(self._undirected[v])

    def adjacent(self, v: int) -> frozenset[int]:
        """Return the nodes adjacent to `v` (parents, children, and undirected)."""
        return frozenset(self._parents[v] | self._children[v] | self._undirected[v])

    # -- Fast read-only accessors (performance-critical inner loops) ---------
    #
    # The public ``parents``/``children``/``undirected_neighbors``/``adjacent``
    # above each allocate a fresh ``frozenset`` on every call. In the GES search
    # those are invoked hundreds of thousands of times per network, where the copy
    # dominates. The accessors below
    # hand back the *live* internal set (or, for ``adjacent_set``, a single fresh
    # plain set) so hot, read-only code can iterate and test membership without
    # that per-call allocation.
    #
    # CONTRACT: the returned set of ``children_set``/``parents_set``/
    # ``undirected_set`` is the graph's own storage -- callers MUST treat it as
    # read-only. Mutating it corrupts the graph. Use the frozenset-returning
    # public methods anywhere the result escapes a local, read-only scope.

    def children_set(self, v: int) -> set[int]:
        """Return the live child set of `v` (read-only; do not mutate)."""
        return self._children[v]

    def parents_set(self, v: int) -> set[int]:
        """Return the live parent set of `v` (read-only; do not mutate)."""
        return self._parents[v]

    def undirected_set(self, v: int) -> set[int]:
        """Return the live undirected-neighbour set of `v` (read-only; do not mutate)."""
        return self._undirected[v]

    def adjacent_set(self, v: int) -> set[int]:
        """Return a fresh plain-set union of `v`'s parents, children and undirected neighbours.

        Unlike :meth:`adjacent` this returns a mutable ``set`` (a fresh copy, so
        mutating it is harmless) rather than a ``frozenset``, skipping the
        frozenset construction in membership-test-only hot paths.
        """
        return self._parents[v] | self._children[v] | self._undirected[v]

    def has_directed(self, u: int, v: int) -> bool:
        """Return whether the directed edge `u -> v` is present."""
        return v in self._children[u]

    def has_undirected(self, u: int, v: int) -> bool:
        """Return whether the undirected edge `{u, v}` is present."""
        return v in self._undirected[u]

    def add_directed(self, u: int, v: int) -> None:
        """Add the directed edge `u -> v`.

        Adding an edge that already exists (same orientation) is a no-op. No
        cycle checking is performed; consistency is enforced only by
        :meth:`to_dag`.

        Parameters
        ----------
        u : int
            Source node index.
        v : int
            Target node index.

        Raises
        ------
        ValueError
            If `u == v` (self-loop), if the opposite directed edge `v -> u`
            already exists, or if the undirected edge `{u, v}` already
            exists -- at most one edge is allowed between any pair of nodes.
        """
        if u == v:
            raise ValueError(f"self-loop rejected: {u} -> {v}")
        if v in self._children[u]:
            return
        if u in self._children[v]:
            raise ValueError(f"opposite directed edge already exists: {v} -> {u}")
        if v in self._undirected[u]:
            raise ValueError(f"undirected edge already exists: {u} - {v}")
        self._children[u].add(v)
        self._parents[v].add(u)
        self._amat = None
        self._succ_csr = None

    def add_undirected(self, u: int, v: int) -> None:
        """Add the undirected edge `{u, v}`.

        Adding an edge that already exists is a no-op.

        Parameters
        ----------
        u : int
            One endpoint index.
        v : int
            Other endpoint index.

        Raises
        ------
        ValueError
            If `u == v` (self-loop), or if a directed edge already exists
            between `u` and `v` (in either orientation) -- at most one edge
            is allowed between any pair of nodes.
        """
        if u == v:
            raise ValueError(f"self-loop rejected: {u} - {v}")
        if v in self._undirected[u]:
            return
        if v in self._children[u] or u in self._children[v]:
            raise ValueError(f"directed edge already exists between {u} and {v}")
        self._undirected[u].add(v)
        self._undirected[v].add(u)
        self._amat = None
        self._succ_csr = None

    def orient(self, u: int, v: int) -> None:
        """Orient the undirected edge `{u, v}` as the directed edge `u -> v`.

        Parameters
        ----------
        u : int
            Node the oriented edge points *from*.
        v : int
            Node the oriented edge points *to*.

        Raises
        ------
        KeyError
            If `{u, v}` is not an undirected edge of the graph.
        """
        if v not in self._undirected[u]:
            raise KeyError(f"no such undirected edge: {u} - {v}")
        self._undirected[u].remove(v)
        self._undirected[v].remove(u)
        self._children[u].add(v)
        self._parents[v].add(u)
        self._amat = None
        self._succ_csr = None

    def remove_edge(self, u: int, v: int) -> None:
        """Remove the edge between `u` and `v`, of either kind and orientation.

        Parameters
        ----------
        u : int
            One endpoint index.
        v : int
            Other endpoint index.

        Raises
        ------
        KeyError
            If there is no edge (directed either way, or undirected) between
            `u` and `v`.
        """
        if v in self._children[u]:
            self._children[u].remove(v)
            self._parents[v].remove(u)
        elif u in self._children[v]:
            self._children[v].remove(u)
            self._parents[u].remove(v)
        elif v in self._undirected[u]:
            self._undirected[u].remove(v)
            self._undirected[v].remove(u)
        else:
            raise KeyError(f"no such edge: {u} - {v}")
        self._amat = None
        self._succ_csr = None

    def directed_edges(self) -> list[tuple[int, int]]:
        """Return all directed edges, sorted."""
        return sorted((u, v) for u in range(self.num_nodes) for v in self._children[u])

    def undirected_edges(self) -> list[tuple[int, int]]:
        """Return all undirected edges as `(u, v)` pairs with `u < v`, sorted."""
        return sorted((u, v) for u in range(self.num_nodes) for v in self._undirected[u] if u < v)

    def adjacency(self) -> np.ndarray:
        """Return the boolean adjacency matrix of the graph.

        An undirected edge `{u, v}` sets both `A[u, v]` and `A[v, u]`; a
        directed edge `u -> v` sets only `A[u, v]`.

        Returns
        -------
        np.ndarray
            A copy of the cached boolean adjacency matrix.
        """
        if self._amat is None:
            amat = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
            for u in range(self.num_nodes):
                for v in self._children[u]:
                    amat[u, v] = True
                for v in self._undirected[u]:
                    amat[u, v] = True
            self._amat = amat
        return self._amat.copy()

    def successor_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Return CSR arrays of each node's semi-directed successors (children ∪ undirected).

        ``indices[indptr[v]:indptr[v + 1]]`` are node ``v``'s successors along a
        semi-directed path -- its children (a directed edge followed forward) and
        its undirected neighbours. Both arrays are ``int32``. The per-node
        successor order is unspecified (reachability is order-invariant). Cached
        lazily and invalidated on mutation, exactly like :meth:`adjacency`; it
        backs the optional numba traversal in :mod:`fedcausal.search`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(indptr, indices)`` with ``indptr`` of length ``num_nodes + 1``.
        """
        if self._succ_csr is None:
            n = self.num_nodes
            indptr = np.zeros(n + 1, dtype=np.int32)
            rows: list[int] = []
            for v in range(n):
                rows.extend(self._children[v] | self._undirected[v])
                indptr[v + 1] = len(rows)
            indices = np.array(rows, dtype=np.int32) if rows else np.zeros(0, dtype=np.int32)
            self._succ_csr = (indptr, indices)
        return self._succ_csr

    def copy(self) -> PDAG:
        """Return an independent deep copy of this graph.

        Returns
        -------
        PDAG
            A new `PDAG` with the same nodes and edges, sharing no mutable
            state with the original.
        """
        clone = PDAG(self.num_nodes)
        clone._parents = [set(p) for p in self._parents]
        clone._children = [set(c) for c in self._children]
        clone._undirected = [set(u) for u in self._undirected]
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PDAG):
            return NotImplemented
        return (
            self.num_nodes == other.num_nodes
            and self.directed_edges() == other.directed_edges()
            and self.undirected_edges() == other.undirected_edges()
        )

    def __hash__(self) -> int:
        return hash((self.num_nodes, tuple(self.directed_edges()), tuple(self.undirected_edges())))

    def __repr__(self) -> str:
        n_dir = sum(len(c) for c in self._children)
        n_undir = sum(len(u) for u in self._undirected) // 2
        return f"PDAG(num_nodes={self.num_nodes}, directed={n_dir}, undirected={n_undir})"

    @classmethod
    def from_dag(cls, dag: DAG) -> PDAG:
        """Build a `PDAG` from a `DAG`, keeping every edge directed."""
        pdag = cls(dag.num_nodes)
        for u, v in dag.edges():
            pdag.add_directed(u, v)
        return pdag

    def to_dag(self) -> DAG:
        """Return a consistent DAG extension of this PDAG (Dor & Tarsi, 1992).

        Repeatedly finds a node `x` that is a valid *sink* -- one with (a) no
        outgoing directed edge among the remaining nodes, and (b) every
        undirected neighbour `y` of `x` adjacent to all other neighbours
        (directed or undirected) of `x` -- then orients every undirected edge
        of `x` *into* `x` and removes `x`. Candidates are scanned in ascending
        index order for determinism.

        Returns
        -------
        DAG
            A DAG that keeps every directed edge of this PDAG and gives every
            undirected edge some orientation, without introducing a new
            v-structure (unshielded collider).

        Raises
        ------
        ValueError
            If no consistent extension exists (i.e. no valid sink can be found
            among the remaining nodes at some step).
        """
        from fedcausal.graph.dag import DAG

        # Work on mutable copies restricted to the "remaining" node set.
        parents = [set(p) for p in self._parents]
        children = [set(c) for c in self._children]
        undirected = [set(u) for u in self._undirected]
        remaining = set(range(self.num_nodes))

        # Directed edges are kept as-is; oriented undirected edges are appended
        # during elimination. The DAG is materialised only after a full,
        # successful elimination, so a directed cycle surfaces as the
        # "no consistent extension" error rather than a premature CycleError.
        result_edges: list[tuple[int, int]] = list(self.directed_edges())

        while remaining:
            chosen: int | None = None
            for x in sorted(remaining):
                # (a) no outgoing directed edge among remaining nodes
                if children[x] & remaining:
                    continue
                # (b) every undirected neighbour y is adjacent to all other
                #     neighbours (directed-parent or undirected) of x
                undir_x = undirected[x] & remaining
                neighbors_x = undir_x | (parents[x] & remaining)
                ok = True
                for y in undir_x:
                    adj_y = (parents[y] | children[y] | undirected[y]) & remaining
                    for z in neighbors_x:
                        if z != y and z not in adj_y:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    chosen = x
                    break

            if chosen is None:
                raise ValueError("no consistent extension")

            # Orient every undirected edge of `chosen` into `chosen`.
            for y in list(undirected[chosen]):
                if y in remaining:
                    result_edges.append((y, chosen))
                undirected[y].discard(chosen)
                undirected[chosen].discard(y)
            remaining.discard(chosen)

        return DAG.from_edges(self.num_nodes, result_edges)
