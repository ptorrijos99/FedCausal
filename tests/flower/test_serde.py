import numpy as np
import pytest

pytest.importorskip("flwr")

from fedcausal.flower.serde import (  # noqa: E402
    control_config,
    dag_to_record,
    names_fingerprint,
    record_to_dag,
    record_to_stats,
    stats_to_record,
)
from fedcausal.graph import DAG  # noqa: E402
from fedcausal.lingam.cumulants import compute_local_raw_stats  # noqa: E402


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


def test_stats_round_trip_is_exact():
    x = np.random.default_rng(0).laplace(0.0, 1.0, size=(500, 4))
    stats = compute_local_raw_stats(x, order=4)
    restored = record_to_stats(stats_to_record(stats), n=stats["n"])
    assert restored["n"] == stats["n"]
    for key in (
        "sum_x",
        "sum_xx",
        "sum_x3",
        "sum_x1y2",
        "sum_x2y1",
        "sum_x4",
        "sum_x1y3",
        "sum_x3y1",
        "sum_x2y2",
    ):
        np.testing.assert_array_equal(restored[key], stats[key])


def test_stats_round_trip_rejects_third_order_only():
    """Order-3 stats have no place on the wire: the transport is fourth-order."""
    x = np.random.default_rng(1).laplace(0.0, 1.0, size=(100, 3))
    with pytest.raises(ValueError, match="fourth-order"):
        stats_to_record(compute_local_raw_stats(x, order=3))


def test_names_fingerprint_agrees_on_equal_names_and_differs_on_order():
    assert names_fingerprint(["a", "b"]) == names_fingerprint(["a", "b"])
    assert names_fingerprint(["a", "b"]) != names_fingerprint(["b", "a"])


def test_stats_to_record_packs_arrays_in_the_fixed_wire_order():
    """Pin the wire order positionally and literally.

    `stats_to_record` and `record_to_stats` both derive from `_STATS_KEYS`, so
    a round-trip test alone cannot catch a reorder of that tuple -- packing
    and unpacking would still agree with each other, just not with a client
    or server built against a different version. This test does not read
    `_STATS_KEYS` at all: it names each of the nine keys literally and checks
    the packed record's arrays against the stats dict fetched by that literal
    name, so a reorder of the source tuple makes this test fail.
    """
    x = np.random.default_rng(2).laplace(0.0, 1.0, size=(200, 3))
    stats = compute_local_raw_stats(x, order=4)
    rec = stats_to_record(stats)
    arrays = rec.to_numpy_ndarrays()

    assert len(arrays) == 9
    np.testing.assert_array_equal(arrays[0], stats["sum_x"])
    np.testing.assert_array_equal(arrays[1], stats["sum_xx"])
    np.testing.assert_array_equal(arrays[2], stats["sum_x3"])
    np.testing.assert_array_equal(arrays[3], stats["sum_x1y2"])
    np.testing.assert_array_equal(arrays[4], stats["sum_x2y1"])
    np.testing.assert_array_equal(arrays[5], stats["sum_x4"])
    np.testing.assert_array_equal(arrays[6], stats["sum_x1y3"])
    np.testing.assert_array_equal(arrays[7], stats["sum_x3y1"])
    np.testing.assert_array_equal(arrays[8], stats["sum_x2y2"])
