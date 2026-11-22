"""Partially directed acyclic graph: the working representation for CPDAGs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fedcausal.graph._kernels import changed_pairs, consistent_extension

if TYPE_CHECKING:
    import numpy.typing as npt

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
    Unlike :class:`DAG`, ``add_directed`` does not check for cycles. A directed
    cycle surfaces only when :meth:`to_dag` finds no consistent extension.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self._parents: list[set[int]] = [set() for _ in range(num_nodes)]
        self._children: list[set[int]] = [set() for _ in range(num_nodes)]
        self._undirected: list[set[int]] = [set() for _ in range(num_nodes)]
        self._amat: npt.NDArray[np.bool_] | None = None
        self._skeleton: npt.NDArray[np.bool_] | None = None
        self._succ_csr: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]] | None = None

    def _invalidate(self) -> None:
        self._amat = None
        self._skeleton = None
        self._succ_csr = None

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

    # Allocation-free accessors for the search's inner loops, where copying into
    # a frozenset on every call dominates the runtime. The first three return
    # the graph's own storage: never mutate it, and never let it escape.

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

        Cheaper than :meth:`adjacent`, which also builds a frozenset.
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
        self._invalidate()

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
        self._invalidate()

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
        self._invalidate()

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
        self._invalidate()

    def update_to(self, adj: npt.ArrayLike) -> None:
        """Make this PDAG the one `adj` encodes, in place, touching only the pairs that differ.

        ``adj[u, v]`` alone is ``u -> v`` and both entries are ``u - v``, as in
        :meth:`adjacency`.
        """
        new = np.ascontiguousarray(adj, dtype=bool)
        old = self._adjacency()
        for u, v in changed_pairs(old.view(np.uint8), new.view(np.uint8)).tolist():
            if old[u, v] or old[v, u]:
                self.remove_edge(u, v)
            if new[u, v] and new[v, u]:
                self.add_undirected(u, v)
            elif new[u, v]:
                self.add_directed(u, v)
            elif new[v, u]:
                self.add_directed(v, u)
        self._amat = new if new is not adj else new.copy()
        self._skeleton = None

    def directed_edges(self) -> list[tuple[int, int]]:
        """Return all directed edges, sorted."""
        return sorted((u, v) for u in range(self.num_nodes) for v in self._children[u])

    def undirected_edges(self) -> list[tuple[int, int]]:
        """Return all undirected edges as `(u, v)` pairs with `u < v`, sorted."""
        return sorted((u, v) for u in range(self.num_nodes) for v in self._undirected[u] if u < v)

    def adjacency(self) -> npt.NDArray[np.bool_]:
        """Return the boolean adjacency matrix of the graph.

        An undirected edge `{u, v}` sets both `A[u, v]` and `A[v, u]`; a
        directed edge `u -> v` sets only `A[u, v]`.

        Returns
        -------
        np.ndarray
            A copy of the cached boolean adjacency matrix.
        """
        return self._adjacency().copy()

    def _adjacency(self) -> npt.NDArray[np.bool_]:
        if self._amat is None:
            amat = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
            for u in range(self.num_nodes):
                for v in self._children[u]:
                    amat[u, v] = True
                for v in self._undirected[u]:
                    amat[u, v] = True
            self._amat = amat
        return self._amat

    def skeleton_view(self) -> npt.NDArray[np.bool_]:
        """Return the cached symmetric boolean matrix of adjacencies (read-only; do not mutate)."""
        if self._skeleton is None:
            amat = self._adjacency()
            self._skeleton = amat | amat.T
        return self._skeleton

    def successor_csr(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Return CSR arrays of each node's semi-directed successors (children ∪ undirected).

        ``indices[indptr[v]:indptr[v + 1]]`` are the nodes a semi-directed path
        can step to from ``v``. Both arrays are ``int32``; the order within a
        node is unspecified. Cached until the next mutation.

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

    @classmethod
    def from_adjacency(cls, adj: npt.ArrayLike) -> PDAG:
        """Build a PDAG from a square matrix: ``A[u, v]`` alone is ``u -> v``, both is ``u - v``."""
        a = np.asarray(adj, dtype=bool)
        pdag = cls(a.shape[0])
        for u, v in zip(*np.nonzero(a & ~a.T), strict=True):
            pdag._children[u].add(int(v))
            pdag._parents[v].add(int(u))
        for u, v in zip(*np.nonzero(np.triu(a & a.T)), strict=True):
            pdag._undirected[u].add(int(v))
            pdag._undirected[v].add(int(u))
        return pdag

    def to_dag(self) -> DAG:
        """Return a DAG in the equivalence class this PDAG describes (Dor & Tarsi, 1992).

        Returns
        -------
        DAG
            Keeps every directed edge, orients every undirected one, and adds
            no v-structure.

        Raises
        ------
        ValueError
            If no such DAG exists.
        """
        from fedcausal.graph.dag import DAG

        dag, ok = consistent_extension(self._adjacency().view(np.uint8))
        if not ok:
            raise ValueError("this PDAG has no consistent DAG extension")
        return DAG.from_amat(dag)
