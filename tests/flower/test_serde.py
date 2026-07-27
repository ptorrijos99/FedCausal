import pytest

pytest.importorskip("flwr")

from fedcausal.flower.serde import control_config, dag_to_record, record_to_dag  # noqa: E402
from fedcausal.graph import DAG  # noqa: E402


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


def test_control_config_carries_scalars():
    cfg = control_config(iteration=3, converged=False, num_nodes=8)
    assert cfg["iteration"] == 3
    assert cfg["converged"] is False
    assert cfg["num_nodes"] == 8


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
