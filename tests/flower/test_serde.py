import numpy as np
import pytest

pytest.importorskip("flwr")

from fedcausal.flower.serde import (  # noqa: E402
    dag_to_record,
    names_fingerprint,
    record_to_dag,
    record_to_sums,
    sums_to_record,
)
from fedcausal.graph import DAG  # noqa: E402
from fedcausal.lingam.cumulants import moment_sums  # noqa: E402


def test_roundtrip_preserves_edges():
    dag = DAG.from_edges(5, [(0, 1), (1, 2), (3, 4)])
    rec = dag_to_record(dag)
    back = record_to_dag(rec, num_nodes=5)
    assert set(back.edges()) == set(dag.edges())
    assert back.num_nodes == 5


def test_empty_dag_roundtrips():
    dag = DAG(4)
    back = record_to_dag(dag_to_record(dag), num_nodes=4)
    assert back.num_nodes == 4 and back.num_edges == 0


def _ds(names, cards):
    import numpy as np

    from fedcausal.data import DiscreteDataset

    values = np.zeros((4, len(names)), dtype=np.int32)
    return DiscreteDataset(
        values,
        np.array(cards, dtype=np.int32),
        names,
        [[str(i) for i in range(c)] for c in cards],
    )


def test_schema_fingerprint_ignores_the_observations():
    from fedcausal.flower.serde import schema_fingerprint

    a, b = _ds(["X", "Y"], [2, 3]), _ds(["X", "Y"], [2, 3])
    b.values[0, 0] = 1
    assert schema_fingerprint(a) == schema_fingerprint(b)


def test_schema_fingerprint_separates_different_variables():
    from fedcausal.flower.serde import schema_fingerprint

    assert schema_fingerprint(_ds(["X", "Y"], [2, 3])) != schema_fingerprint(
        _ds(["X", "Z"], [2, 3])
    )
    assert schema_fingerprint(_ds(["X", "Y"], [2, 3])) != schema_fingerprint(
        _ds(["X", "Y"], [2, 4])
    )
    assert schema_fingerprint(_ds(["X", "Y"], [2, 2])) != schema_fingerprint(
        _ds(["Y", "X"], [2, 2])
    )


def test_stats_round_trip_is_exact():
    x = np.random.default_rng(0).laplace(0.0, 1.0, size=(500, 4))
    stats = moment_sums(x, order=4)
    restored = record_to_sums(sums_to_record(stats), n=stats["n"])
    assert restored["n"] == stats["n"]
    for key in ("mean", "m11", "m21", "m31", "m22"):
        np.testing.assert_array_equal(restored[key], stats[key])


def test_third_order_stats_travel_without_the_fourth_order_sums():
    x = np.random.default_rng(1).laplace(0.0, 1.0, size=(100, 3))
    stats = moment_sums(x, order=3)
    restored = record_to_sums(sums_to_record(stats), n=stats["n"])
    assert restored.keys() == stats.keys()
    for key in ("mean", "m11", "m21"):
        np.testing.assert_array_equal(restored[key], stats[key])


def test_names_fingerprint_agrees_on_equal_names_and_differs_on_order():
    assert names_fingerprint(["a", "b"]) == names_fingerprint(["a", "b"])
    assert names_fingerprint(["a", "b"]) != names_fingerprint(["b", "a"])
