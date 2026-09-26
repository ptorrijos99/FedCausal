"""Greedy equivalence search (Chickering, 2002).

The forward phase (:func:`run_fes`) applies the best valid Insert until none
improves the score; the backward phase (:func:`run_bes`) does the same with
Delete. These full scans re-score every candidate on every step. :class:`GES`
uses the incremental engine in :mod:`fedcausal.search.incremental` by default,
which gives the same result much faster; ``incremental=False`` selects these.
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
    # The graph does not change during a scan.
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
        If `h_pool` is unrecognised, even when there is nothing to scan.
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
        Restrict every step's scan to these ordered pairs. An undirected edge
        needs both `(x, y)` and `(y, x)` to be fully searched.
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
        # Under h_pool="bayesfl" every H ties with the empty set, which is
        # scanned first and so always kept. A non-empty H from that pool would
        # make apply_delete raise, since its nodes are not adjacent to x.
        ops.apply_delete(current, x, y, h)
        current = rebuild_cpdag(current)
        changed = True
    return current, changed


class GES:
    """Greedy Equivalence Search: the forward phase, then the backward phase.

    Parameters
    ----------
    score : Score
        Decomposable local score, built from the same data later passed to
        :meth:`fit`. Only the data's ``n_vars`` is read, so a mismatch is not
        detected.
    h_pool : HPool, optional
        Which pool the backward phase draws `H` from; see :data:`HPool`.
    max_t_size : int or None, optional
        Cap on ``|T|`` in the forward phase. Default 2, which matched the
        exhaustive scan on every benchmark network while running up to 20x
        faster. Any cap gives up Chickering's optimality guarantee in theory;
        `None` is the exhaustive scan.
    empty_t_gate : bool, optional
        Skip a candidate pair unless it improves the score with ``T = {}``.
        Default `False`. It costs the optimum and exists only to reproduce the
        original Java implementation; see ``docs/algorithms.md``.
    incremental : bool, optional
        Re-score only the pairs each operator can affect. Default `True`; about
        2x faster at 37 nodes and 13x at 223, with the same result. `False`
        uses the full scan.

    Raises
    ------
    ValueError
        If `h_pool` is unrecognised or `max_t_size` is negative.
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
        """Run only the backward phase, starting from `initial_graph`'s CPDAG."""
        start = self._start_cpdag(data, initial_graph)
        cpdag, _ = self._run_bes(start, candidate_edges)
        return cpdag
