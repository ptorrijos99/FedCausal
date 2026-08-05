import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG, PDAG
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.scores import BDeuScore
from fedcausal.search import gesops as ops
from fedcausal.search._candidates import adjacent_ordered_pairs
from fedcausal.search.ges import _best_delete, run_bes


def _dep_dataset(n: int = 4000, seed: int = 0) -> DiscreteDataset:
    # A -> B deterministic-ish: B copies A with 10% noise; C, D independent.
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = np.where(rng.random(n) < 0.9, a, 1 - a)
    c = rng.integers(0, 2, n)
    d = rng.integers(0, 2, n)
    values = np.column_stack([a, b, c, d]).astype(np.int32)
    return DiscreteDataset.from_arrays(
        values, np.full(4, 2, np.int32), list("ABCD"), [["0", "1"]] * 4
    )


def test_bes_prunes_spurious_edges() -> None:
    ds = _dep_dataset()
    score = BDeuScore(ds)
    full = dag_to_cpdag(DAG.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]))
    cpdag, flag = run_bes(score, full)
    assert flag is True
    assert 1 in cpdag.adjacent(0)  # true A-B adjacency survives
    assert 3 not in cpdag.adjacent(2)  # spurious C-D pruned


def test_bes_noop_on_empty() -> None:
    _, flag = run_bes(BDeuScore(_dep_dataset()), PDAG(4))
    assert flag is False


def test_bes_noop_on_minimal_true_edge() -> None:
    # BES has a real candidate here, the true A-B adjacency, and must reject it
    # because deleting it would hurt the score rather than because there is
    # nothing to scan.
    ds = _dep_dataset()
    score = BDeuScore(ds)
    minimal = PDAG(4)
    minimal.add_undirected(0, 1)
    cpdag, changed = run_bes(score, minimal)
    assert changed is False
    assert cpdag.directed_edges() == []
    assert cpdag.undirected_edges() == [(0, 1)]


def test_run_bes_does_not_mutate_input_pdag() -> None:
    ds = _dep_dataset()
    score = BDeuScore(ds)
    full = dag_to_cpdag(DAG.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]))
    snapshot_directed = full.directed_edges()
    snapshot_undirected = full.undirected_edges()
    run_bes(score, full)
    assert full.directed_edges() == snapshot_directed
    assert full.undirected_edges() == snapshot_undirected


def test_run_bes_invalid_h_pool_raises() -> None:
    # A typo'd h_pool must fail loudly rather than silently defaulting to
    # either real mode -- this fixture has no edges at all, so a lazy
    # implementation that only validates h_pool inside the per-pair scan loop
    # would never actually raise here (the loop body never runs).
    with pytest.raises(ValueError):
        run_bes(BDeuScore(_dep_dataset()), PDAG(4), h_pool="bogus")  # type: ignore[arg-type]


def test_run_bes_restricted_candidate_edges_keeps_spurious_edge() -> None:
    # Restricting candidate_edges to every ordered pair except (2, 3) and (3, 2)
    # keeps that edge from ever being scanned, so it survives even though the
    # data would otherwise justify removing it.
    ds = _dep_dataset()
    score = BDeuScore(ds)
    full = dag_to_cpdag(DAG.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]))
    n = 4
    restricted = frozenset(
        (x, y) for x in range(n) for y in range(n) if x != y and (x, y) not in {(2, 3), (3, 2)}
    )
    cpdag, _ = run_bes(score, full, restricted)
    assert 3 in cpdag.adjacent(2)


class _DirectedDeleteScore:
    """A fake score where only Delete(x=0, y=1, H=empty) improves.

    The reverse-labelled Delete(x=1, y=0) never does.
    """

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        table = {
            (1, ()): 5.0,
            (1, (0,)): 0.0,
        }
        return table.get((node, parents), 0.0)


def test_run_bes_candidate_edges_pair_order_is_directional() -> None:
    # candidate_edges holds ordered pairs even for an undirected edge: (0, 1)
    # means "consider Delete(x=0, y=1, ...)", not "this edge is a candidate".
    # Restricting to one ordering is therefore direction-sensitive, and
    # _DirectedDeleteScore only rewards deleting with y=1.
    score = _DirectedDeleteScore()
    start = PDAG(2)
    start.add_undirected(0, 1)

    cpdag_allowed, changed_allowed = run_bes(score, start.copy(), frozenset({(0, 1)}))
    assert changed_allowed is True
    assert cpdag_allowed.directed_edges() == []
    assert cpdag_allowed.undirected_edges() == []

    cpdag_blocked, changed_blocked = run_bes(score, start.copy(), frozenset({(1, 0)}))
    assert changed_blocked is False
    assert cpdag_blocked.undirected_edges() == [(0, 1)]


class _TiedFakeDeleteScore:
    """A hand-built score with three exactly-tied best Deletes.

    The fixture is a 3-node star with undirected edges 0-1, 0-2.
    ``Delete(x=0, y=1, H=empty)``, ``Delete(x=0, y=2, H=empty)``, and
    ``Delete(x=1, y=0, H=empty)`` all have delta 10.0; ``Delete(x=2, y=0,
    H=empty)`` ties at delta 0.0 (not an improvement, correctly excluded).
    Every candidate pair here has an empty `NaYX` (1 and 2 are not adjacent to
    each other), so `H` can only ever be the empty set -- this fixture
    isolates the (x, y)-pair scan order and tie-break rule from any `H`-subset
    interaction. The scan order is ascending `(x, y)`, so among the three tied
    candidates `(x=0, y=1)` is considered strictly first.
    """

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        table = {
            (1, ()): 10.0,
            (1, (0,)): 0.0,
            (2, ()): 10.0,
            (2, (0,)): 0.0,
            (0, ()): 10.0,
            (0, (1,)): 0.0,
            (0, (2,)): 10.0,
        }
        return table.get((node, parents), 0.0)


def test_best_delete_keeps_first_seen_candidate_on_exact_tie() -> None:
    pdag = PDAG(3)
    pdag.add_undirected(0, 1)
    pdag.add_undirected(0, 2)
    fake_score = _TiedFakeDeleteScore()
    best = _best_delete(fake_score, pdag, None, "nayx")
    assert best == (10.0, 0, 1, frozenset())


def _common_neighbour_fixture() -> PDAG:
    # x=0, y=1: the edge under deletion. Node 2 is an undirected neighbour of
    # y adjacent to x (via undirected 0-2) -- in NaYX(0, 1). Node 3 is an
    # undirected neighbour of y NOT adjacent to x -- in T0(0, 1), the BayesFL
    # H pool, but *not* in NaYX(0, 1).
    pdag = PDAG(4)
    pdag.add_undirected(0, 1)
    pdag.add_undirected(1, 2)
    pdag.add_undirected(0, 2)
    pdag.add_undirected(1, 3)
    return pdag


class _CommonNeighbourScore:
    """A fake score built for `_common_neighbour_fixture`.

    Under `h_pool="nayx"`, `H={2}` (a genuinely non-empty subset of
    `NaYX(0, 1) = {2}`) strictly beats `H=empty` for `Delete(x=0, y=1)`
    (delta 5.0 vs 2.0), so `_best_delete` must select the non-empty `H`.

    Under `h_pool="bayesfl"`, `H` is drawn from `T0(0, 1) = {3}` instead,
    which is disjoint from `NaYX(0, 1)` by construction, so `NaYX \\ H` is the
    same set (`{2}`) regardless of `H`: `H=empty` and `H={3}` are an exact
    tie at delta 2.0, and the first-seen rule keeps `H=empty`. That degenerate
    delta (2.0) is strictly less than the real Chickering pool's winning
    delta (5.0), so the two pools disagree both on which `H` is chosen and on
    the resulting score delta at this same edge.
    """

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        table = {
            (1, ()): 5.0,
            (1, (0,)): 0.0,
            (1, (2,)): 3.0,
            (1, (0, 2)): 1.0,
            (1, (3,)): 5.0,
        }
        return table.get((node, parents), 0.0)


def test_best_delete_selects_non_empty_h_under_nayx_pool() -> None:
    pdag = _common_neighbour_fixture()
    score = _CommonNeighbourScore()
    best = _best_delete(score, pdag, None, "nayx")
    assert best == (5.0, 0, 1, frozenset({2}))


def test_best_delete_degenerates_to_empty_h_under_bayesfl_pool() -> None:
    # On the fixture where the nayx pool selects a non-empty H={2}, the bayesfl
    # pool keeps H=empty despite having a non-empty T0={3} to draw from, and at
    # a strictly worse delta (2.0 against 5.0).
    pdag = _common_neighbour_fixture()
    score = _CommonNeighbourScore()
    best = _best_delete(score, pdag, None, "bayesfl")
    assert best == (2.0, 0, 1, frozenset())


def test_bayesfl_pool_h_empty_invariant_keeps_apply_delete_from_crashing() -> None:
    # ops.apply_delete raises KeyError when some w in H has no edge to x, which
    # is by construction what t0_neighbors (the "bayesfl" H pool) selects for.
    # run_bes calls apply_delete unconditionally, so it is only safe under
    # h_pool="bayesfl" because _best_delete always returns H=empty there. The
    # fixture has a genuinely non-empty t0_neighbors, so the invariant being
    # pinned is not vacuous.
    pdag = _common_neighbour_fixture()
    score = _CommonNeighbourScore()
    assert ops.t0_neighbors(pdag, 0, 1) == frozenset({3})  # genuinely non-empty pool

    best = _best_delete(score, pdag, None, "bayesfl")
    assert best is not None
    _, x, y, h = best
    assert h == frozenset()

    # 3 is a t0_neighbors member and non-adjacent to x=0, so handing a
    # non-empty H drawn from that pool to apply_delete raises the KeyError.
    ops.apply_delete(pdag.copy(), x, y, h)
    with pytest.raises(KeyError):
        ops.apply_delete(pdag.copy(), x, y, frozenset({3}))


def test_run_bes_pools_diverge_on_common_neighbourhood() -> None:
    # The two pools diverge over a whole run, not just at one _best_delete
    # call. Under "nayx", Delete(0, 1, H={2}) goes first and force-flips 0->2
    # and 1->2, after which nothing further improves. Under "bayesfl",
    # Delete(0, 1, H=empty) goes first, leaves node 2 untouched, and BES then
    # removes the now-spurious 1-2 edge in a second step.
    score = _CommonNeighbourScore()

    cpdag_nayx, changed_nayx = run_bes(score, _common_neighbour_fixture(), h_pool="nayx")
    cpdag_bayesfl, changed_bayesfl = run_bes(score, _common_neighbour_fixture(), h_pool="bayesfl")

    assert changed_nayx is True
    assert cpdag_nayx.directed_edges() == [(0, 2), (1, 2)]
    assert cpdag_nayx.undirected_edges() == [(1, 3)]

    assert changed_bayesfl is True
    assert cpdag_bayesfl.directed_edges() == []
    assert cpdag_bayesfl.undirected_edges() == [(0, 2), (1, 3)]


def testadjacent_ordered_pairs_mixed_directed_and_undirected_edges() -> None:
    # A regression that emitted the reverse orientation of directed edges too
    # would pass every other test here: it would still remove the same edge,
    # via a Delete(y, x, ...) that scores node x rather than node y. Only this
    # pin catches it. Directed 0->1 contributes just (0, 1); each undirected
    # edge contributes both orderings.
    pdag = PDAG(4)
    pdag.add_directed(0, 1)  # directed: contributes only (0, 1)
    pdag.add_undirected(1, 2)  # undirected: contributes (1, 2) and (2, 1)
    pdag.add_undirected(0, 3)  # undirected: contributes (0, 3) and (3, 0)
    assert adjacent_ordered_pairs(pdag) == [
        (0, 1),
        (0, 3),
        (1, 2),
        (2, 1),
        (3, 0),
    ]


def test_run_bes_deletes_a_genuinely_directed_edge() -> None:
    # Every other end-to-end run_bes test in this file starts from a fully
    # undirected graph, so apply_delete's removal of a directed x -> y edge
    # (as opposed to an undirected x - y edge) is never otherwise exercised
    # end-to-end. Reuses _DirectedDeleteScore (only Delete(x=0, y=1, H=empty)
    # improves), but this time the starting 0-1 edge is directed the way that
    # score rewards deleting, not undirected.
    score = _DirectedDeleteScore()
    start = PDAG(2)
    start.add_directed(0, 1)
    cpdag, changed = run_bes(score, start)
    assert changed is True
    assert cpdag.directed_edges() == []
    assert cpdag.undirected_edges() == []
