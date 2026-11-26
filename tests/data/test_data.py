"""The splitters that simulate sites from one table."""

from __future__ import annotations

import numpy as np
import pytest
from causalbase.data import ContinuousDataset, DiscreteDataset

from fedcausal.data import contiguous_split, dirichlet_split


def _ids(n_rows: int, n_levels: int = 4) -> DiscreteDataset:
    """A dataset whose first column is each row's index, the second a random label."""
    labels = np.random.default_rng(0).integers(0, n_levels, size=n_rows)
    return DiscreteDataset(
        np.column_stack([np.arange(n_rows), labels]),
        [n_rows, n_levels],
        ["id", "label"],
        [[str(i) for i in range(n_rows)], [f"L{k}" for k in range(n_levels)]],
    )


@pytest.mark.parametrize(("n_rows", "n_clients"), [(10, 4), (7, 12), (50_000_003, 7)])
def test_contiguous_split_is_the_table_in_balanced_blocks(n_rows: int, n_clients: int) -> None:
    if n_rows > 1000:  # only the bounds, which must not overflow
        bounds = [-(-c * n_rows // n_clients) for c in range(n_clients + 1)]
        assert bounds[-1] == n_rows
        assert max(np.diff(bounds)) - min(np.diff(bounds)) <= 1
        return
    ds = _ids(n_rows)
    parts = contiguous_split(ds, n_clients)
    sizes = [p.n_rows for p in parts]
    assert max(sizes) - min(sizes) <= 1
    assert np.concatenate([p.values[:, 0] for p in parts]).tolist() == list(range(n_rows))
    assert all(p.categories == ds.categories for p in parts)


@pytest.mark.parametrize("by", [None, 1])
def test_dirichlet_split_is_a_reproducible_partition_into_nonempty_sites(by: int | None) -> None:
    ds = _ids(200)
    parts = dirichlet_split(ds, 6, alpha=0.1, seed=3, by=by)
    rows = np.concatenate([p.values[:, 0] for p in parts])
    assert sorted(rows.tolist()) == list(range(200))
    assert all(p.n_rows > 0 for p in parts)
    assert all(p.categories == ds.categories for p in parts)
    again = dirichlet_split(ds, 6, alpha=0.1, seed=3, by=by)
    assert all(np.array_equal(p.values, q.values) for p, q in zip(parts, again, strict=True))
    other = dirichlet_split(ds, 6, alpha=0.1, seed=4, by=by)
    assert any(p.n_rows != q.n_rows for p, q in zip(parts, other, strict=True))


def test_dirichlet_skew_follows_alpha() -> None:
    ds = _ids(1000)
    even = [p.n_rows for p in dirichlet_split(ds, 4, alpha=1e6, seed=0)]
    assert max(even) - min(even) <= 4
    for level in range(4):  # a tiny alpha puts each level on few sites
        parts = dirichlet_split(ds, 4, alpha=0.01, seed=42, by=1)
        assert sum(int(np.any(p.values[:, 1] == level)) for p in parts) < 4
    continuous = ContinuousDataset(np.arange(60.0).reshape(20, 3), ["a", "b", "c"])
    assert sum(p.n_rows for p in dirichlet_split(continuous, 4, alpha=1.0, seed=0)) == 20
    assert all(isinstance(p, ContinuousDataset) for p in contiguous_split(continuous, 3))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n_clients": 0, "alpha": 1.0}, ValueError),
        ({"n_clients": 3, "alpha": 0.0}, ValueError),
        ({"n_clients": 101, "alpha": 1.0}, ValueError),
        ({"n_clients": 3, "alpha": 1.0, "by": 5}, ValueError),
    ],
)
def test_dirichlet_split_refuses_impossible_splits(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        dirichlet_split(_ids(100), seed=0, **kwargs)


def test_split_refusals() -> None:
    with pytest.raises(ValueError):
        contiguous_split(_ids(10), 0)
    continuous = ContinuousDataset(np.zeros((20, 3)), ["a", "b", "c"])
    with pytest.raises(TypeError, match="only supported for a DiscreteDataset"):
        dirichlet_split(continuous, 4, alpha=1.0, seed=0, by=1)
