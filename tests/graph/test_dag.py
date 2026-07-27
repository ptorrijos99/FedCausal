import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fedcausal.graph import DAG, CycleError


def test_empty_graph_has_no_edges_and_correct_node_count() -> None:
    dag = DAG(4)
    assert dag.num_nodes == 4
    assert dag.num_edges == 0
    assert dag.edges() == []
    for v in range(4):
        assert dag.parents(v) == frozenset()
        assert dag.children(v) == frozenset()
        assert dag.adjacent(v) == frozenset()


def test_add_edge_then_query() -> None:
    dag = DAG(3)
    dag.add_edge(0, 1)
    assert dag.has_edge(0, 1)
    assert not dag.has_edge(1, 0)
    assert dag.num_edges == 1
    assert dag.parents(1) == frozenset({0})
    assert dag.children(0) == frozenset({1})
    assert dag.adjacent(0) == frozenset({1})
    assert dag.adjacent(1) == frozenset({0})
    assert dag.edges() == [(0, 1)]


def test_remove_edge() -> None:
    dag = DAG(3)
    dag.add_edge(0, 1)
    dag.remove_edge(0, 1)
    assert not dag.has_edge(0, 1)
    assert dag.num_edges == 0
    assert dag.parents(1) == frozenset()
    assert dag.children(0) == frozenset()


def test_remove_absent_edge_raises_key_error() -> None:
    dag = DAG(3)
    with pytest.raises(KeyError):
        dag.remove_edge(0, 1)


def test_add_edge_rejects_cycle_and_leaves_graph_unchanged() -> None:
    dag = DAG(3)
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    with pytest.raises(CycleError):
        dag.add_edge(2, 0)
    assert not dag.has_edge(2, 0)
    assert dag.num_edges == 2
    assert dag.edges() == [(0, 1), (1, 2)]


def test_add_edge_is_idempotent_for_existing_edge() -> None:
    dag = DAG(3)
    dag.add_edge(0, 1)
    dag.add_edge(0, 1)
    assert dag.num_edges == 1
    assert dag.edges() == [(0, 1)]


def test_add_edge_rejects_self_loop() -> None:
    dag = DAG(3)
    with pytest.raises(CycleError):
        dag.add_edge(0, 0)
    assert dag.num_edges == 0


def test_topological_order_respects_all_edges() -> None:
    dag = DAG(5)
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    dag.add_edge(0, 3)
    dag.add_edge(3, 4)
    order = dag.topological_order()
    assert sorted(order) == list(range(5))
    pos = {node: i for i, node in enumerate(order)}
    assert all(pos[a] < pos[b] for a, b in dag.edges())


def test_topological_order_of_empty_graph_is_deterministic() -> None:
    dag = DAG(3)
    assert dag.topological_order() == [0, 1, 2]


def test_ancestors() -> None:
    dag = DAG(4)
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    dag.add_edge(3, 2)
    assert dag.ancestors(2) == frozenset({0, 1, 3})
    assert dag.ancestors(0) == frozenset()


def test_has_path() -> None:
    dag = DAG(4)
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    assert dag.has_path(0, 2)
    assert dag.has_path(0, 1)
    assert not dag.has_path(2, 0)
    assert not dag.has_path(0, 3)
    assert not dag.has_path(0, 0)


def test_from_edges_and_adjacency_round_trip() -> None:
    dag = DAG.from_edges(4, [(0, 1), (1, 2), (0, 2)])
    assert dag.num_nodes == 4
    assert dag.edges() == [(0, 1), (0, 2), (1, 2)]
    amat = dag.adjacency()
    assert amat.dtype == bool
    expected = np.zeros((4, 4), dtype=bool)
    expected[0, 1] = True
    expected[0, 2] = True
    expected[1, 2] = True
    assert np.array_equal(amat, expected)


def test_adjacency_returns_copy_not_view() -> None:
    dag = DAG.from_edges(2, [(0, 1)])
    amat = dag.adjacency()
    amat[1, 0] = True
    assert not dag.has_edge(1, 0)


def test_from_amat_round_trip() -> None:
    A = np.zeros((3, 3), dtype=bool)
    A[0, 1] = True
    A[1, 2] = True
    dag = DAG.from_amat(A)
    assert dag.num_nodes == 3
    assert dag.edges() == [(0, 1), (1, 2)]
    assert np.array_equal(dag.adjacency(), A)


def test_from_amat_rejects_cyclic_matrix() -> None:
    A = np.zeros((3, 3), dtype=bool)
    A[0, 1] = True
    A[1, 2] = True
    A[2, 0] = True
    with pytest.raises(CycleError):
        DAG.from_amat(A)


def test_equality_by_num_nodes_and_edge_set() -> None:
    dag1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    dag2 = DAG.from_edges(3, [(1, 2), (0, 1)])
    dag3 = DAG.from_edges(4, [(0, 1), (1, 2)])
    dag4 = DAG.from_edges(3, [(0, 1)])
    assert dag1 == dag2
    assert dag1 != dag3
    assert dag1 != dag4
    assert dag1 != "not a dag"


def test_copy_is_independent() -> None:
    dag = DAG.from_edges(3, [(0, 1)])
    clone = dag.copy()
    assert clone == dag
    clone.add_edge(1, 2)
    assert not dag.has_edge(1, 2)
    assert dag.num_edges == 1


def test_adjacency_cache_invalidated_on_mutation() -> None:
    dag = DAG(3)
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)

    # Warm the cache
    amat1 = dag.adjacency()
    expected1 = np.zeros((3, 3), dtype=bool)
    expected1[0, 1] = True
    expected1[1, 2] = True
    assert np.array_equal(amat1, expected1)

    # Add edge and verify cache was invalidated
    dag.add_edge(0, 2)
    amat2 = dag.adjacency()
    expected2 = np.zeros((3, 3), dtype=bool)
    expected2[0, 1] = True
    expected2[1, 2] = True
    expected2[0, 2] = True
    assert np.array_equal(amat2, expected2)

    # Remove edge and verify cache was invalidated again
    dag.remove_edge(0, 2)
    amat3 = dag.adjacency()
    assert np.array_equal(amat3, expected1)


@given(st.integers(2, 12), st.data())
def test_random_edge_insertion_never_creates_cycle(n: int, data) -> None:
    dag = DAG(n)
    for _ in range(3 * n):
        u = data.draw(st.integers(0, n - 1))
        v = data.draw(st.integers(0, n - 1))
        if u == v or dag.has_edge(u, v):
            continue
        try:
            dag.add_edge(u, v)
        except CycleError:
            assert dag.has_path(v, u)  # rejection was justified
    order = dag.topological_order()
    pos = {x: i for i, x in enumerate(order)}
    assert all(pos[a] < pos[b] for a, b in dag.edges())
