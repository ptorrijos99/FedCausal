from fedcausal.fusion import Intersection, intersection_fusion
from fedcausal.graph import DAG


def test_keeps_only_edges_present_in_every_dag():
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1)])
    out = intersection_fusion([d1, d2])
    assert set(out.edges()) == {(0, 1)}


def test_single_dag_is_identity():
    d = DAG.from_edges(3, [(0, 1), (2, 1)])
    assert set(intersection_fusion([d]).edges()) == {(0, 1), (2, 1)}


def test_operator_matches_function():
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1), (2, 1)])
    assert set(Intersection().fuse([d1, d2]).edges()) == set(intersection_fusion([d1, d2]).edges())


def test_intersection_is_skeleton_based_and_oriented_by_base():
    # d2 has 3->2; its skeleton pair {2, 3} matches d1's 2->3, so the edge
    # survives, oriented the way d1 (the base) has it. This mirrors Java's
    # reorient-to-base-order-then-intersect.
    d1 = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    d2 = DAG.from_edges(4, [(0, 1), (1, 2), (3, 2)])
    out = intersection_fusion([d1, d2])
    assert out.num_nodes == 4
    assert set(out.edges()) == {(0, 1), (1, 2), (2, 3)}


def test_drops_pair_absent_from_a_dag():
    d1 = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    d2 = DAG.from_edges(4, [(0, 1), (1, 2)])  # no {2, 3} pair at all
    out = intersection_fusion([d1, d2])
    assert set(out.edges()) == {(0, 1), (1, 2)}
