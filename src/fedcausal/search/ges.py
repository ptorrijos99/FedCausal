r"""Greedy Equivalence Search: the forward and backward phases, and the ``GES`` entry point.

* *Forward Equivalence Search* (:func:`run_fes`) repeatedly applies the
  highest-scoring valid Insert, re-canonicalising to a CPDAG after each step,
  until no Insert improves the score.
* *Backward Equivalence Search* (:func:`run_bes`) is the mirror image over
  Delete.

Both are full scans: they re-score every candidate pair on every step. They are
the reference, exact by construction. :class:`GES` defaults to
the incremental engine in :mod:`fedcausal.search.incremental`, which is verified
byte-identical against these and much faster as graphs grow; pass
``incremental=False`` to search with these instead.

The operator primitives both phases drive live in :mod:`fedcausal.search.gesops`;
the candidate scanning and tie-breaking they share with the incremental engine
live in :mod:`fedcausal.search._candidates`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.graph.pdag import PDAG
from fedcausal.search import gesops as ops
from fedcausal.search._candidates import (
    EPS,
    H_POOLS,
    HPool,
    adjacent_ordered_pairs,
    best_delete_for_pair,
    best_insert_for_pair,
    invalid_h_pool_error,
    rebuild_cpdag,
)

if TYPE_CHECKING:
    from fedcausal.data import DiscreteDataset
    from fedcausal.graph import DAG
    from fedcausal.scores import Score

__all__ = ["HPool", "run_fes", "run_bes", "GES"]


def _best_insert(
    score: Score,
    pdag: PDAG,
    candidate_edges: frozenset[tuple[int, int]] | None,
    max_t_size: int | None = None,
    empty_t_gate: bool = False,
) -> tuple[float, int, int, frozenset[int]] | None:
    """Return the best-scoring valid Insert as ``(delta, x, y, T)``, or `None`.

    Scans every ordered non-adjacent pair in ascending order, keeping the
    highest-delta improving Insert. Ties go to the first seen.
    """
    best: tuple[float, int, int, frozenset[int]] | None = None
    n = pdag.num_nodes
    # Adjacency is fixed for the whole scan (the graph is only edited between
    # scans), so materialise it once rather than inside the O(n^2) pair loop.
    adj = [pdag.adjacent_set(v) for v in range(n)]
    for x in range(n):
        for y in range(n):
            if x == y or x in adj[y]:
                continue
            if candidate_edges is not None and (x, y) not in candidate_edges:
                continue
            pair = best_insert_for_pair(score, pdag, x, y, max_t_size, empty_t_gate)
            if pair is not None and (best is None or pair[0] > best[0] + EPS):
                best = (pair[0], x, y, pair[1])
    return best


def run_fes(
    score: Score,
    pdag: PDAG,
    candidate_edges: frozenset[tuple[int, int]] | None = None,
    max_t_size: int | None = None,
    empty_t_gate: bool = False,
) -> tuple[PDAG, bool]:
    """Run the forward phase to a local optimum, starting from `pdag`.

    `pdag` is never mutated; the search runs on a copy.

    Parameters
    ----------
    score : Score
        Decomposable local score.
    pdag : PDAG
        Starting graph, typically empty, but any CPDAG is accepted.
    candidate_edges : frozenset[tuple[int, int]] or None, optional
        Restrict every step's scan to these `(x, y)` pairs. `None` scans all.
    max_t_size : int or None, optional
        Cap on ``|T|``. `None` is the exhaustive Chickering (2002) scan.
    empty_t_gate : bool, optional
        Abandon a pair unless it already passes validity and scores positively
        at ``T = {}``; see ``docs/algorithms.md``.

    Returns
    -------
    tuple[PDAG, bool]
        The closed CPDAG, and whether any Insert was applied.
    """
    changed = False
    current = pdag.copy()
    while True:
        best = _best_insert(score, current, candidate_edges, max_t_size, empty_t_gate)
        if best is None:
            break
        _, x, y, t = best
        ops.apply_insert(current, x, y, t)
        current = rebuild_cpdag(current)
        changed = True
    return current, changed


def _best_delete(
    score: Score,
    pdag: PDAG,
    candidate_edges: frozenset[tuple[int, int]] | None,
    h_pool: HPool,
) -> tuple[float, int, int, frozenset[int]] | None:
    """Return the best-scoring valid Delete as ``(delta, x, y, H)``, or `None`.

    Raises
    ------
    ValueError
        If `h_pool` is unrecognised. Checked before the scan, so a typo never
        silently no-ops on a graph with no eligible pairs.
    """
    if h_pool not in H_POOLS:
        raise invalid_h_pool_error(h_pool)

    best: tuple[float, int, int, frozenset[int]] | None = None
    for x, y in adjacent_ordered_pairs(pdag):
        if candidate_edges is not None and (x, y) not in candidate_edges:
            continue
        pair = best_delete_for_pair(score, pdag, x, y, h_pool)
        if pair is not None and (best is None or pair[0] > best[0] + EPS):
            best = (pair[0], x, y, pair[1])
    return best


def run_bes(
    score: Score,
    pdag: PDAG,
    candidate_edges: frozenset[tuple[int, int]] | None = None,
    h_pool: HPool = "nayx",
) -> tuple[PDAG, bool]:
    """Run the backward phase to a local optimum, starting from `pdag`.

    `pdag` is never mutated; the search runs on a copy.

    Parameters
    ----------
    score : Score
        Decomposable local score.
    pdag : PDAG
        Starting graph, typically the output of :func:`run_fes`.
    candidate_edges : frozenset[tuple[int, int]] or None, optional
        Restrict every step's scan to these pairs, matched as *ordered* pairs.
        For an undirected edge both `(x, y)` and `(y, x)` must be present to
        give Delete its full search over it; supplying only the canonical
        `(min, max)` pair silently loses the other labelling's candidates.
    h_pool : HPool, optional
        Which pool `H` is drawn from; see :data:`HPool`.

    Returns
    -------
    tuple[PDAG, bool]
        The closed CPDAG, and whether any Delete was applied.

    Raises
    ------
    ValueError
        If `h_pool` is unrecognised.
    """
    changed = False
    current = pdag.copy()
    while True:
        best = _best_delete(score, current, candidate_edges, h_pool)
        if best is None:
            break
        _, x, y, h = best
        # Safety-relevant coupling, not incidental: ops.apply_delete raises
        # KeyError when some w in h has no edge to x, which is exactly what
        # h_pool="bayesfl" selects for. This call is only safe there because
        # the empty-set-first subset order plus the strict tie rule guarantee
        # h is empty under that pool; "nayx" needs no such guarantee, since
        # na_yx(x, y) only contains nodes already adjacent to x.
        ops.apply_delete(current, x, y, h)
        current = rebuild_cpdag(current)
        changed = True
    return current, changed


class GES:
    """Greedy Equivalence Search: the forward phase, then the backward phase.

    Parameters
    ----------
    score : Score
        Decomposable local score. Every method taking a `data` argument
        requires the score to have been built from that same dataset -- `data`
        itself is consulted only for `n_vars`. Passing a mismatched `data`
        silently produces nonsense; this is a caller contract, not something
        `GES` can check through the `Score` protocol.
    h_pool : HPool, optional
        Which pool the backward phase draws `H` from; see :data:`HPool`.
    max_t_size : int or None, optional
        Cap on ``|T|`` in the forward phase. **Defaults to 2.** Measured
        byte-identical to the exhaustive scan on all nine benchmark networks
        while running up to 20x faster, so it is the default. It is not free in
        theory -- any finite cap forfeits Chickering's optimality guarantee --
        so pass `None` for the exhaustive scan, and say which you used in any
        published result.
    empty_t_gate : bool, optional
        Abandon a candidate pair unless it already passes validity and scores
        positively at ``T = {}``. Defaults to `False`: unlike the cap, this
        does cost the optimum, and it is not reliably faster. Provided for
        compatibility with the original Java implementation, not as a recommended
        setting; see ``docs/algorithms.md``.
    incremental : bool, optional
        **Defaults to `True`.** Use the incremental engine, which re-scores
        only the pairs whose neighbourhood changed after each operator rather
        than rescanning everything. Verified byte-identical to the full scan on
        every benchmark network, and dramatically faster as the graph grows
        (~2x at 37 nodes, ~13x at 223). Pass `False` for the full scan, the
        guaranteed-exact reference.

    Raises
    ------
    ValueError
        If `h_pool` is unrecognised or `max_t_size` is negative. Both checked
        at construction, so a typo fails immediately even on an instance only
        ever used through :meth:`fes_only`.
    """

    def __init__(
        self,
        score: Score,
        h_pool: HPool = "nayx",
        max_t_size: int | None = 2,
        empty_t_gate: bool = False,
        incremental: bool = True,
    ) -> None:
        if h_pool not in H_POOLS:
            raise invalid_h_pool_error(h_pool)
        if max_t_size is not None and max_t_size < 0:
            raise ValueError(f"max_t_size must be non-negative or None, got {max_t_size}")
        self.score = score
        self.h_pool: HPool = h_pool
        self.max_t_size = max_t_size
        self.empty_t_gate = empty_t_gate
        self.incremental = incremental

    def _run_fes(
        self, pdag: PDAG, candidate_edges: frozenset[tuple[int, int]] | None
    ) -> tuple[PDAG, bool]:
        """Dispatch the forward phase per `self.incremental`."""
        if self.incremental:
            from fedcausal.search.incremental import run_fes_incremental

            return run_fes_incremental(
                self.score, pdag, candidate_edges, self.max_t_size, self.empty_t_gate
            )
        return run_fes(self.score, pdag, candidate_edges, self.max_t_size, self.empty_t_gate)

    def _run_bes(
        self, pdag: PDAG, candidate_edges: frozenset[tuple[int, int]] | None
    ) -> tuple[PDAG, bool]:
        """Dispatch the backward phase per `self.incremental`."""
        if self.incremental:
            from fedcausal.search.incremental import run_bes_incremental

            return run_bes_incremental(self.score, pdag, candidate_edges, self.h_pool)
        return run_bes(self.score, pdag, candidate_edges, self.h_pool)

    def _start_cpdag(self, data: DiscreteDataset, initial_graph: DAG | None) -> PDAG:
        """Return the starting CPDAG: empty on `data.n_vars`, or `initial_graph`'s.

        Raises
        ------
        ValueError
            If `initial_graph` is given and its node count disagrees with
            `data.n_vars`.
        """
        if initial_graph is not None and initial_graph.num_nodes != data.n_vars:
            raise ValueError(
                f"initial_graph has {initial_graph.num_nodes} nodes but data has "
                f"{data.n_vars} variables"
            )
        if initial_graph is None:
            return PDAG(data.n_vars)
        return dag_to_cpdag(initial_graph)

    def fit(
        self,
        data: DiscreteDataset,
        initial_graph: DAG | None = None,
        *,
        candidate_edges: frozenset[tuple[int, int]] | None = None,
    ) -> PDAG:
        """Run both phases to a local optimum and return the learned CPDAG.

        Parameters
        ----------
        data : DiscreteDataset
            Dataset the score was built from; only `n_vars` is read here.
        initial_graph : DAG or None, optional
            Starting graph. `None` starts from the empty graph.
        candidate_edges : frozenset[tuple[int, int]] or None, optional
            Restrict both phases to these pairs; see :func:`run_bes` for the
            ordered-pair semantics.

        Returns
        -------
        PDAG
            The CPDAG after both phases reach a local optimum.

        Raises
        ------
        ValueError
            If `initial_graph`'s node count disagrees with `data.n_vars`.
        """
        start = self._start_cpdag(data, initial_graph)
        fes_cpdag, _ = self._run_fes(start, candidate_edges)
        bes_cpdag, _ = self._run_bes(fes_cpdag, candidate_edges)
        return bes_cpdag

    def fes_only(
        self,
        data: DiscreteDataset,
        initial_graph: DAG | None = None,
        *,
        candidate_edges: frozenset[tuple[int, int]] | None = None,
    ) -> PDAG:
        """Run only the forward phase; see :meth:`fit` for the arguments."""
        start = self._start_cpdag(data, initial_graph)
        cpdag, _ = self._run_fes(start, candidate_edges)
        return cpdag

    def bes_only(
        self,
        data: DiscreteDataset,
        initial_graph: DAG,
        *,
        candidate_edges: frozenset[tuple[int, int]] | None = None,
    ) -> PDAG:
        """Run only the backward phase, starting from `initial_graph`'s CPDAG.

        Unlike :meth:`fit` and :meth:`fes_only`, `initial_graph` is required: a
        backward pass from an empty graph is meaningless, since nothing is ever
        adjacent and no Delete is ever valid.
        """
        start = self._start_cpdag(data, initial_graph)
        cpdag, _ = self._run_bes(start, candidate_edges)
        return cpdag
