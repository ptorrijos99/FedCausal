import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG, PDAG
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.scores import BDeuScore
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
    # The true edge is a real candidate, and deleting it would lower the score.
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
    # Raised even with nothing to scan.
    with pytest.raises(ValueError):
        run_bes(BDeuScore(_dep_dataset()), PDAG(4), h_pool="bogus")  # type: ignore[arg-type]


def test_run_bes_restricted_candidate_edges_keeps_spurious_edge() -> None:
    # The data justify deleting C - D, but it is not a candidate.
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
    # Candidates are ordered pairs: only (0, 1) allows the delete this score rewards.
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


def _common_neighbour_fixture() -> PDAG:
    # x=0, y=1. Node 2 is in NaYX(0, 1); node 3 is a neighbour of y not adjacent to
    # x, so it is in T0(0, 1) but not in NaYX(0, 1).
    pdag = PDAG(4)
    pdag.add_undirected(0, 1)
    pdag.add_undirected(1, 2)
    pdag.add_undirected(0, 2)
    pdag.add_undirected(1, 3)
    return pdag


class _CommonNeighbourScore:
    """A score on which the two H pools choose differently for Delete(0, 1).

    With ``"nayx"``, H = {2} beats the empty set (5.0 against 2.0). With
    ``"bayesfl"`` the pool is {3}, disjoint from NaYX, so every H ties at 2.0 and
    the empty set, scanned first, is kept.
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
    pdag = _common_neighbour_fixture()
    score = _CommonNeighbourScore()
    best = _best_delete(score, pdag, None, "bayesfl")
    assert best == (2.0, 0, 1, frozenset())


def test_run_bes_pools_diverge_on_common_neighbourhood() -> None:
    # The two pools lead to different CPDAGs, not just different first steps.
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
    # A directed edge gives only its forward pair; an undirected one gives both.
    pdag = PDAG(4)
    pdag.add_directed(0, 1)  # directed: contributes only (0, 1)
    pdag.add_undirected(1, 2)  # undirected: contributes (1, 2) and (2, 1)
    pdag.add_undirected(0, 3)  # undirected: contributes (0, 3) and (3, 0)
    assert set(adjacent_ordered_pairs(pdag)) == {(0, 1), (0, 3), (1, 2), (2, 1), (3, 0)}


def test_run_bes_deletes_a_genuinely_directed_edge() -> None:
    score = _DirectedDeleteScore()
    start = PDAG(2)
    start.add_directed(0, 1)
    cpdag, changed = run_bes(score, start)
    assert changed is True
    assert cpdag.directed_edges() == []
    assert cpdag.undirected_edges() == []
