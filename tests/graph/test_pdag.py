import numpy as np
import pytest

from fedcausal.graph import DAG, PDAG
from fedcausal.graph.cpdag import dag_to_cpdag


def test_empty_pdag() -> None:
    p = PDAG(4)
    assert p.num_nodes == 4
    assert p.directed_edges() == []
    assert p.undirected_edges() == []
    for v in range(4):
        assert p.parents(v) == frozenset()
        assert p.children(v) == frozenset()
        assert p.undirected_neighbors(v) == frozenset()
        assert p.adjacent(v) == frozenset()


def test_add_directed() -> None:
    p = PDAG(3)
    p.add_directed(0, 1)
    assert p.has_directed(0, 1)
    assert not p.has_directed(1, 0)
    assert not p.has_undirected(0, 1)
    assert p.parents(1) == frozenset({0})
    assert p.children(0) == frozenset({1})
    assert p.adjacent(0) == frozenset({1})
    assert p.adjacent(1) == frozenset({0})
    assert p.directed_edges() == [(0, 1)]
    assert p.undirected_edges() == []


def test_add_undirected() -> None:
    p = PDAG(3)
    p.add_undirected(2, 1)
    assert p.has_undirected(1, 2)
    assert p.has_undirected(2, 1)
    assert not p.has_directed(1, 2)
    assert not p.has_directed(2, 1)
    assert p.undirected_neighbors(1) == frozenset({2})
    assert p.undirected_neighbors(2) == frozenset({1})
    assert p.adjacent(1) == frozenset({2})
    # undirected edges are reported with u < v, sorted
    assert p.undirected_edges() == [(1, 2)]
    assert p.directed_edges() == []


def test_orient_undirected_to_directed() -> None:
    p = PDAG(3)
    p.add_undirected(1, 2)
    p.orient(2, 1)
    assert p.has_directed(2, 1)
    assert not p.has_undirected(1, 2)
    assert p.parents(1) == frozenset({2})
    assert p.children(2) == frozenset({1})


def test_orient_requires_undirected_edge() -> None:
    p = PDAG(3)
    p.add_directed(0, 1)
    with pytest.raises(KeyError):
        p.orient(0, 1)


def test_remove_edge_either_kind() -> None:
    p = PDAG(3)
    p.add_directed(0, 1)
    p.add_undirected(1, 2)
    p.remove_edge(0, 1)
    p.remove_edge(2, 1)
    assert not p.has_directed(0, 1)
    assert not p.has_undirected(1, 2)
    assert p.directed_edges() == []
    assert p.undirected_edges() == []


def test_remove_missing_edge_raises() -> None:
    p = PDAG(3)
    with pytest.raises(KeyError):
        p.remove_edge(0, 1)


def test_copy_is_independent() -> None:
    p = PDAG(3)
    p.add_directed(0, 1)
    p.add_undirected(1, 2)
    q = p.copy()
    q.remove_edge(0, 1)
    q.add_directed(2, 0)
    assert p.has_directed(0, 1)
    assert not p.has_directed(2, 0)
    assert p == p.copy()


def test_equality_by_edge_sets() -> None:
    p = PDAG(3)
    p.add_directed(0, 1)
    p.add_undirected(1, 2)
    q = PDAG(3)
    q.add_undirected(2, 1)
    q.add_directed(0, 1)
    assert p == q
    q.orient(1, 2)
    assert p != q


def test_from_dag_all_directed() -> None:
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    p = PDAG.from_dag(dag)
    assert p.has_directed(0, 1)
    assert p.has_directed(1, 2)
    assert p.undirected_edges() == []
    assert p.directed_edges() == [(0, 1), (1, 2)]


def test_to_dag_all_directed_roundtrip() -> None:
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    p = PDAG.from_dag(dag)
    out = p.to_dag()
    assert out == dag


def test_to_dag_orients_undirected_chain() -> None:
    # 0 - 1 - 2 chain: any acyclic orientation without a new v-structure works.
    p = PDAG(3)
    p.add_undirected(0, 1)
    p.add_undirected(1, 2)
    out = p.to_dag()
    assert out.num_edges == 2
    # no new v-structure at 1: 1 must not have two non-adjacent parents
    assert not (0 in out.parents(1) and 2 in out.parents(1))


def test_to_dag_no_extension_raises() -> None:
    # A directed cycle among "remaining" nodes admits no consistent extension.
    p = PDAG(3)
    p.add_directed(0, 1)
    p.add_directed(1, 2)
    p.add_directed(2, 0)
    with pytest.raises(ValueError, match="no consistent"):
        p.to_dag()


def test_add_undirected_rejects_existing_directed() -> None:
    p = PDAG(2)
    p.add_directed(0, 1)
    with pytest.raises(ValueError):
        p.add_undirected(0, 1)


def test_add_directed_rejects_existing_undirected() -> None:
    p = PDAG(2)
    p.add_undirected(0, 1)
    with pytest.raises(ValueError):
        p.add_directed(0, 1)


def test_add_directed_rejects_opposite_orientation() -> None:
    p = PDAG(2)
    p.add_directed(0, 1)
    with pytest.raises(ValueError):
        p.add_directed(1, 0)


def test_add_same_edge_twice_is_noop() -> None:
    p = PDAG(2)
    p.add_directed(0, 1)
    p.add_directed(0, 1)
    assert p.directed_edges() == [(0, 1)]


def _fresh(pdag: PDAG) -> PDAG:
    """The same graph with no cached arrays."""
    return pdag.copy()


def _neighbours(pdag: PDAG) -> list[list[int]]:
    nbr, deg = pdag.neighbour_lists()
    return [sorted(nbr[v, : deg[v]].tolist()) for v in range(pdag.num_nodes)]


@pytest.mark.parametrize("seed", range(10))
def test_cached_arrays_follow_every_mutation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 12
    pdag = PDAG(n)
    pdag.adjacency_view()
    pdag.neighbour_lists()
    for _ in range(300):
        u, v = (int(i) for i in rng.choice(n, size=2, replace=False))
        if v in pdag.adjacent(u):
            if pdag.has_undirected(u, v) and rng.random() < 0.5:
                pdag.orient(u, v)
            else:
                pdag.remove_edge(u, v)
        elif rng.random() < 0.5:
            pdag.add_directed(u, v)
        else:
            pdag.add_undirected(u, v)
        fresh = _fresh(pdag)
        assert np.array_equal(pdag.adjacency_view(), fresh.adjacency())
        assert np.array_equal(pdag.skeleton_view(), fresh.skeleton_view())
        assert _neighbours(pdag) == [sorted(fresh.adjacent(w)) for w in range(n)]


def test_neighbour_lists_grow_past_their_first_width() -> None:
    pdag = PDAG(30)
    pdag.neighbour_lists()
    for v in range(1, 30):
        pdag.add_undirected(0, v)
    assert _neighbours(pdag)[0] == list(range(1, 30))
    assert _neighbours(pdag)[29] == [0]


@pytest.mark.parametrize("seed", range(20))
def test_complete_in_place_gives_the_cpdag_and_its_changes(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 25))
    order = rng.permutation(n)
    dag = DAG(n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.2:
                dag.add_edge(int(order[i]), int(order[j]))
    pdag = PDAG.from_dag(dag)
    before = pdag.adjacency()
    pairs = pdag.complete_in_place()
    want = dag_to_cpdag(dag)
    assert pdag == want
    assert np.array_equal(pdag.adjacency_view(), want.adjacency())
    changed = before != want.adjacency()
    assert sorted(map(tuple, pairs.tolist())) == sorted(
        {(min(u, v), max(u, v)) for u, v in zip(*np.nonzero(changed), strict=True)}
    )
    assert len(pdag.complete_in_place()) == 0


def test_complete_in_place_without_extension_raises_and_keeps_the_graph() -> None:
    pdag = PDAG(4)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        pdag.add_undirected(u, v)
    before = pdag.copy()
    with pytest.raises(ValueError, match="no consistent extension"):
        pdag.complete_in_place()
    assert pdag == before
