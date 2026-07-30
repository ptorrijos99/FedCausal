from fedcausal.fedges.model import BNModel
from fedcausal.graph import DAG


def _dag(edges, n=3):
    return DAG.from_edges(n, edges)


def test_equality_ignores_score_and_uses_directed_edge_set():
    a = BNModel(_dag([(0, 1), (1, 2)]), score=-10.0)
    b = BNModel(_dag([(1, 2), (0, 1)]), score=999.0)
    assert a == b
    assert hash(a) == hash(b)


def test_edge_direction_matters():
    a = BNModel(_dag([(0, 1)]))
    b = BNModel(_dag([(1, 0)]))
    assert a != b


def test_hashable_in_a_set():
    s = {BNModel(_dag([(0, 1)])), BNModel(_dag([(0, 1)]))}
    assert len(s) == 1
