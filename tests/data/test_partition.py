from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.data.partition import contiguous_split, dirichlet_split

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
TOY_CSV = FIXTURES / "toy.csv"


def _toy_dataset() -> DiscreteDataset:
    return DiscreteDataset.from_csv(TOY_CSV)


def _synthetic_dataset(
    n_rows: int = 100, n_label_levels: int = 4, seed: int = 0
) -> DiscreteDataset:
    """Build a dataset with a bijective ``id`` column for tracing rows.

    Column 0 (``id``) holds the code ``i`` for original row ``i``, so a
    partitioner's output can be checked for an exact row partition by
    collecting the ``id`` column across all clients.
    """
    rng = np.random.default_rng(seed)
    id_values = np.arange(n_rows, dtype=np.int32)
    label_values = rng.integers(0, n_label_levels, size=n_rows).astype(np.int32)
    values = np.column_stack([id_values, label_values]).astype(np.int32)
    cardinalities = np.array([n_rows, n_label_levels], dtype=np.int32)
    names = ["id", "label"]
    categories = [
        [str(i) for i in range(n_rows)],
        [f"L{k}" for k in range(n_label_levels)],
    ]
    return DiscreteDataset.from_arrays(values, cardinalities, names, categories)


# --------------------------------------------------------------------------
# contiguous_split
# --------------------------------------------------------------------------


def test_contiguous_split_sizes() -> None:
    ds = _toy_dataset()
    parts = contiguous_split(ds, 3)
    assert [p.n_rows for p in parts] == [3, 3, 2]


def test_contiguous_split_matches_java_ceil_boundaries() -> None:
    # BN_DataSet.divideDataSet uses ceil boundaries, which for a non-divisible
    # (n, N) differ from numpy.array_split's front-loaded blocks. n=10, N=4:
    # bucketSize=2.5 -> boundaries [0, 3, 5, 8, 10] -> sizes [3, 2, 3, 2]
    # (array_split would give [3, 3, 2, 2]).
    ds = _synthetic_dataset(n_rows=10)
    parts = contiguous_split(ds, 4)
    assert [p.n_rows for p in parts] == [3, 2, 3, 2]
    # still an exact, order-preserving partition (trace via the id column)
    reconstructed = np.concatenate([p.values for p in parts], axis=0)
    assert np.array_equal(reconstructed[:, 0], np.arange(10))


def test_contiguous_split_reproduces_original_rows_in_order() -> None:
    ds = _toy_dataset()
    parts = contiguous_split(ds, 3)
    reconstructed = np.concatenate([p.values for p in parts], axis=0)
    assert np.array_equal(reconstructed, ds.values)


def test_contiguous_split_parts_keep_full_cardinalities() -> None:
    ds = _toy_dataset()
    parts = contiguous_split(ds, 3)
    for part in parts:
        assert np.array_equal(part.cardinalities, ds.cardinalities)
        assert part.categories == ds.categories


def test_contiguous_split_rejects_non_positive_n_clients() -> None:
    ds = _toy_dataset()
    with pytest.raises(ValueError):
        contiguous_split(ds, 0)


# --------------------------------------------------------------------------
# dirichlet_split: quantity skew (by=None)
# --------------------------------------------------------------------------


def test_dirichlet_split_quantity_skew_is_exact_partition() -> None:
    ds = _synthetic_dataset(n_rows=100)
    parts = dirichlet_split(ds, 5, alpha=0.5, seed=123)
    assert len(parts) == 5
    all_ids = np.concatenate([p.values[:, 0] for p in parts])
    assert sorted(all_ids.tolist()) == list(range(100))


def test_dirichlet_split_quantity_skew_all_clients_nonempty() -> None:
    ds = _synthetic_dataset(n_rows=100)
    parts = dirichlet_split(ds, 5, alpha=0.05, seed=99)
    assert all(p.n_rows >= 1 for p in parts)


def test_dirichlet_split_quantity_skew_reproducible_with_same_seed() -> None:
    ds = _synthetic_dataset(n_rows=100)
    parts1 = dirichlet_split(ds, 5, alpha=0.5, seed=123)
    parts2 = dirichlet_split(ds, 5, alpha=0.5, seed=123)
    for p1, p2 in zip(parts1, parts2, strict=True):
        assert np.array_equal(p1.values, p2.values)


def test_dirichlet_split_different_seeds_differ() -> None:
    ds = _synthetic_dataset(n_rows=100)
    parts1 = dirichlet_split(ds, 5, alpha=0.3, seed=1)
    parts2 = dirichlet_split(ds, 5, alpha=0.3, seed=2)
    sizes1 = [p.n_rows for p in parts1]
    sizes2 = [p.n_rows for p in parts2]
    assert sizes1 != sizes2


def test_dirichlet_split_large_alpha_is_approximately_balanced() -> None:
    ds = _synthetic_dataset(n_rows=100)
    parts = dirichlet_split(ds, 5, alpha=1e6, seed=1)
    sizes = [p.n_rows for p in parts]
    assert sum(sizes) == 100
    assert max(sizes) - min(sizes) <= 2


def test_dirichlet_split_keeps_full_cardinalities() -> None:
    ds = _synthetic_dataset(n_rows=100)
    parts = dirichlet_split(ds, 5, alpha=0.5, seed=1)
    for part in parts:
        assert np.array_equal(part.cardinalities, ds.cardinalities)


def test_dirichlet_split_n_clients_equal_n_rows_forces_singletons() -> None:
    ds = _toy_dataset()  # 8 rows
    parts = dirichlet_split(ds, 8, alpha=0.01, seed=0)
    assert all(p.n_rows == 1 for p in parts)


def test_dirichlet_split_rejects_too_many_clients() -> None:
    ds = _toy_dataset()  # 8 rows
    with pytest.raises(ValueError):
        dirichlet_split(ds, 9, alpha=1.0, seed=0)


def test_dirichlet_split_rejects_non_positive_n_clients() -> None:
    ds = _toy_dataset()
    with pytest.raises(ValueError):
        dirichlet_split(ds, 0, alpha=1.0, seed=0)


def test_dirichlet_split_rejects_non_positive_alpha() -> None:
    ds = _toy_dataset()
    with pytest.raises(ValueError):
        dirichlet_split(ds, 3, alpha=0.0, seed=0)


# --------------------------------------------------------------------------
# dirichlet_split: value skew (by=j)
# --------------------------------------------------------------------------


def test_dirichlet_split_value_skew_is_exact_partition() -> None:
    ds = _synthetic_dataset(n_rows=100, n_label_levels=4)
    parts = dirichlet_split(ds, 4, alpha=0.5, seed=7, by=1)
    all_ids = np.concatenate([p.values[:, 0] for p in parts])
    assert sorted(all_ids.tolist()) == list(range(100))


def test_dirichlet_split_value_skew_all_clients_nonempty() -> None:
    ds = _synthetic_dataset(n_rows=100, n_label_levels=4)
    parts = dirichlet_split(ds, 4, alpha=0.05, seed=7, by=1)
    assert all(p.n_rows >= 1 for p in parts)


def test_dirichlet_split_value_skew_reproducible_with_same_seed() -> None:
    ds = _synthetic_dataset(n_rows=100, n_label_levels=4)
    parts1 = dirichlet_split(ds, 4, alpha=0.3, seed=7, by=1)
    parts2 = dirichlet_split(ds, 4, alpha=0.3, seed=7, by=1)
    for p1, p2 in zip(parts1, parts2, strict=True):
        assert np.array_equal(p1.values, p2.values)


def test_dirichlet_split_value_skew_concentrates_levels() -> None:
    # With a tiny alpha, each label level should be concentrated onto very
    # few clients rather than spread near-evenly across all of them.
    ds = _synthetic_dataset(n_rows=200, n_label_levels=4, seed=0)
    parts = dirichlet_split(ds, 4, alpha=0.01, seed=42, by=1)
    for level in range(4):
        counts_per_client = [int(np.sum(p.values[:, 1] == level)) for p in parts]
        nonzero_clients = sum(1 for c in counts_per_client if c > 0)
        assert nonzero_clients < 4  # not spread across every client


def test_dirichlet_split_rejects_out_of_range_by() -> None:
    ds = _synthetic_dataset(n_rows=100)
    with pytest.raises(ValueError):
        dirichlet_split(ds, 3, alpha=1.0, seed=0, by=5)
