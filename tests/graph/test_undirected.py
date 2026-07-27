import numpy as np

from fedcausal.graph import DAG
from fedcausal.graph.undirected import (
    fill_in,
    maximal_cliques,
    mcs_order,
    moralize,
    treewidth,
    treewidth_java,
)


def _symmetric_and_hollow(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    if not np.array_equal(adj, adj.T):
        return False
    return not adj.diagonal().any() if n else True


def test_moralize_collider_adds_edge_between_parents() -> None:
    # 0 -> 2, 1 -> 2: parents of the collider get an edge added.
    dag = DAG.from_edges(3, [(0, 2), (1, 2)])
    moral = moralize(dag)
    assert _symmetric_and_hollow(moral)
    assert moral[0, 1] and moral[1, 0]
    assert moral[0, 2] and moral[2, 0]
    assert moral[1, 2] and moral[2, 1]


def test_moralize_chain_is_just_the_skeleton() -> None:
    # 0 -> 1 -> 2: no node has more than one parent, so no edges are added.
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    moral = moralize(dag)
    expected = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ]
    )
    np.testing.assert_array_equal(moral, expected)


def test_moralize_diamond_adds_edge_between_co_parents() -> None:
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3: 3's parents {1, 2} get an edge.
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    moral = moralize(dag)
    assert moral[1, 2] and moral[2, 1]
    # Skeleton edges are still present.
    assert moral[0, 1] and moral[0, 2] and moral[1, 3] and moral[2, 3]
    # No spurious edges.
    assert not moral[0, 3]


def test_mcs_order_ties_break_to_lowest_index() -> None:
    # A 4-cycle 0-1-2-3-0: every node starts at cardinality 0, so the first
    # pick must be node 0 (lowest index among ties).
    adj = np.array(
        [
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
        ]
    )
    order = mcs_order(adj)
    assert order[0] == 0
    assert set(order) == {0, 1, 2, 3}
    assert len(order) == 4


def test_mcs_order_empty_graph() -> None:
    adj = np.zeros((0, 0), dtype=bool)
    assert mcs_order(adj) == []


def test_fill_in_diamond_moral_graph_already_chordal() -> None:
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    moral = moralize(dag)
    order = mcs_order(moral)
    triangulated = fill_in(moral, order)
    # The diamond's moral graph is already chordal (1-2 chord exists), so
    # fill-in should not add any edges.
    np.testing.assert_array_equal(triangulated, moral)
    assert _symmetric_and_hollow(triangulated)


def test_fill_in_produces_chordal_graph_via_perfect_elimination_check() -> None:
    # A 4-cycle 0-1-2-3-0 is NOT chordal; fill-in must add a chord.
    adj = np.array(
        [
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
        ]
    )
    order = mcs_order(adj)
    triangulated = fill_in(adj, order)
    assert _symmetric_and_hollow(triangulated)

    # A graph is chordal iff running MCS again and checking the "earlier
    # neighbours of v form a clique" condition holds for every node
    # (perfect elimination ordering test).
    check_order = mcs_order(triangulated)
    position = {node: idx for idx, node in enumerate(check_order)}
    for i in range(len(check_order) - 1, -1, -1):
        v = check_order[i]
        earlier_neighbors = [
            w for w in range(triangulated.shape[0]) if triangulated[v, w] and position[w] < i
        ]
        for a in range(len(earlier_neighbors)):
            for b in range(a + 1, len(earlier_neighbors)):
                assert triangulated[earlier_neighbors[a], earlier_neighbors[b]]

    # Original edges preserved, and at least one chord added.
    assert np.array_equal(triangulated & adj, adj)
    assert triangulated.sum() > adj.sum()


def test_maximal_cliques_no_subset_cliques() -> None:
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    moral = moralize(dag)
    order = mcs_order(moral)
    triangulated = fill_in(moral, order)
    order2 = mcs_order(triangulated)
    cliques = maximal_cliques(triangulated, order2)

    # No clique is a subset of another.
    for i, c1 in enumerate(cliques):
        for j, c2 in enumerate(cliques):
            if i != j:
                assert not c1 <= c2

    node_sets = set(cliques)
    assert frozenset({0, 1, 2}) in node_sets
    assert frozenset({1, 2, 3}) in node_sets


def test_treewidth_java_diamond_is_three() -> None:
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert treewidth_java(dag) == 3


def test_treewidth_diamond_is_two() -> None:
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert treewidth(dag) == 2


def test_treewidth_java_tree_is_two() -> None:
    # A simple out-tree: no node has more than one parent, so moralization
    # adds no edges and the largest clique is a single edge (size 2).
    dag = DAG.from_edges(5, [(0, 1), (0, 2), (1, 3), (1, 4)])
    assert treewidth_java(dag) == 2


def test_treewidth_tree_is_one() -> None:
    dag = DAG.from_edges(5, [(0, 1), (0, 2), (1, 3), (1, 4)])
    assert treewidth(dag) == 1


def test_treewidth_java_single_node_is_one() -> None:
    dag = DAG(1)
    assert treewidth_java(dag) == 1


def test_treewidth_java_empty_graph_is_zero() -> None:
    dag = DAG(0)
    assert treewidth_java(dag) == 0
