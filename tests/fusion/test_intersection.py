from fedcausal.fusion import intersection
from fedcausal.graph import DAG


def test_keeps_only_edges_present_in_every_dag():
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1)])
    out = intersection([d1, d2])
    assert set(out.edges()) == {(0, 1)}


def test_single_dag_is_identity():
    d = DAG.from_edges(3, [(0, 1), (2, 1)])
    assert set(intersection([d]).edges()) == {(0, 1), (2, 1)}


def test_intersection_is_skeleton_based_and_oriented_by_base():
    # d2's 3->2 matches d1's 2->3 as a pair; the edge keeps d1's orientation.
    d1 = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    d2 = DAG.from_edges(4, [(0, 1), (1, 2), (3, 2)])
    out = intersection([d1, d2])
    assert out.num_nodes == 4
    assert set(out.edges()) == {(0, 1), (1, 2), (2, 3)}


def test_drops_pair_absent_from_a_dag():
    d1 = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    d2 = DAG.from_edges(4, [(0, 1), (1, 2)])  # no {2, 3} pair at all
    out = intersection([d1, d2])
    assert set(out.edges()) == {(0, 1), (1, 2)}
