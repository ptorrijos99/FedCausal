from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedcausal.data import ContinuousDataset, DiscreteDataset
from fedcausal.data.partition import contiguous_split, dirichlet_split

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
TOY_CSV = FIXTURES / "toy.csv"


def _toy_dataset() -> DiscreteDataset:
    return DiscreteDataset.from_csv(TOY_CSV)


def _synthetic_dataset(
    n_rows: int = 100, n_label_levels: int = 4, seed: int = 0
) -> DiscreteDataset:
    """Build a dataset whose ``id`` column gives each row's original index."""
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


def test_contiguous_split_spreads_the_remainder() -> None:
    ds = _synthetic_dataset(n_rows=10)
    parts = contiguous_split(ds, 4)
    assert [p.n_rows for p in parts] == [3, 2, 3, 2]


class _RowCounter:
    """A dataset stand-in that records which rows each block asked for."""

    def __init__(self, n_rows: int) -> None:
        self.n_rows = n_rows
        self.n_vars = 1

    def subset(self, rows: np.ndarray) -> tuple[int, int]:
        return (int(rows[0]), int(rows[-1]) + 1) if rows.size else (0, 0)


@pytest.mark.parametrize("n_rows", [7, 5000, 16_777_217, 50_000_003])
@pytest.mark.parametrize("n_clients", [3, 7, 10, 12])
def test_contiguous_split_covers_every_row_once(n_rows: int, n_clients: int) -> None:
    blocks = [b for b in contiguous_split(_RowCounter(n_rows), n_clients) if b != (0, 0)]  # type: ignore[type-var]
    assert blocks[0][0] == 0 and blocks[-1][1] == n_rows
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:], strict=False))


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


# --------------------------------------------------------------------------
# ContinuousDataset support
# --------------------------------------------------------------------------


def test_contiguous_split_accepts_continuous():
    ds = ContinuousDataset.from_arrays(np.arange(30, dtype=float).reshape(10, 3), ["a", "b", "c"])
    parts = contiguous_split(ds, 3)
    assert len(parts) == 3
    assert all(isinstance(p, ContinuousDataset) for p in parts)
    assert sum(p.n_rows for p in parts) == 10
    assert all(p.names == ds.names for p in parts)


def test_dirichlet_split_accepts_continuous():
    ds = ContinuousDataset.from_arrays(np.arange(60, dtype=float).reshape(20, 3), ["a", "b", "c"])
    parts = dirichlet_split(ds, 4, alpha=1.0, seed=0)
    assert len(parts) == 4
    assert all(isinstance(p, ContinuousDataset) for p in parts)
    assert sum(p.n_rows for p in parts) == 20


def test_dirichlet_split_rejects_by_with_continuous():
    ds = ContinuousDataset.from_arrays(np.arange(60, dtype=float).reshape(20, 3), ["a", "b", "c"])
    with pytest.raises(
        TypeError, match=r"by is only supported for a DiscreteDataset, got ContinuousDataset"
    ):
        dirichlet_split(ds, 4, alpha=1.0, seed=0, by=1)
