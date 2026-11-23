"""Datasets: how tables are read and encoded, what is refused, and the site splitters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedcausal.data import (
    ContinuousDataset,
    DiscreteDataset,
    as_discrete_sites,
    contiguous_split,
    dirichlet_split,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_csv_levels_are_coded_in_sorted_order_and_none_is_a_level() -> None:
    ds = DiscreteDataset.from_csv(FIXTURES / "toy.csv")
    assert ds.names == ("A", "B", "C")
    assert ds.categories == (("a", "b", "c"), ("hi", "lo"), ("p", "q", "r"))
    assert ds.values[:, 0].tolist() == [1, 0, 2, 0, 1, 0, 2, 1]
    assert ds.cardinalities.tolist() == [3, 2, 3]
    with pytest.raises(ValueError, match="read-only"):
        ds.cardinalities[0] = 9
    # The same table through pandas, and back.
    frame = pd.read_csv(FIXTURES / "toy.csv", dtype=str)
    assert np.array_equal(DiscreteDataset.from_pandas(frame).values, ds.values)
    assert ds.to_pandas().equals(frame)
    # "None" is a real level in some networks (child's DuctFlow, insurance's ThisCarDam).
    none = DiscreteDataset.from_csv(FIXTURES / "none_category.csv")
    assert none.categories[0].count("None") == 1


def test_a_subset_keeps_every_level_and_its_own_values() -> None:
    ds = DiscreteDataset.from_csv(FIXTURES / "toy.csv")
    sub = ds.subset(ds.values[:, 0] != 2)  # no row with A == "c"
    assert sub.n_rows == 6
    assert sub.categories == ds.categories
    assert np.array_equal(sub.cardinalities, ds.cardinalities)
    sub.values[0, 0] = 2
    assert ds.values[0, 0] == 1


def test_sites_share_one_encoding_whatever_levels_each_has() -> None:
    a = pd.DataFrame({"x": ["p", "q"], "y": ["1", "1"]})
    b = pd.DataFrame({"x": ["r", "p"], "y": ["2", "1"]})
    sites = as_discrete_sites([a, b])
    assert sites[0].categories == sites[1].categories == (("p", "q", "r"), ("1", "2"))
    with pytest.raises(ValueError, match="columns"):
        as_discrete_sites([a, b.rename(columns={"y": "z"})])


MALFORMED = [
    (lambda: DiscreteDataset(np.zeros(3), [1], ["a"], [["x"]]), "2-D"),
    (lambda: DiscreteDataset(np.zeros((3, 2)), [1], ["a", "b"], [["x"], ["x"]]), "shape"),
    (lambda: DiscreteDataset(np.zeros((3, 1)), [2], ["a"], [["x"]]), "levels"),
    (lambda: DiscreteDataset(np.zeros((3, 2)), [1, 1], ["a"], [["x"], ["x"]]), "names"),
    (lambda: DiscreteDataset([[0], [2]], [2], ["a"], [["x", "y"]]), "outside"),
    (lambda: DiscreteDataset(np.zeros((2, 2)), [1, 1], ["a", "a"], [["x"], ["x"]]), "repeat"),
    # 1 and "1" are one name once read as strings.
    (lambda: DiscreteDataset.from_pandas(pd.DataFrame([["x", "y"]], columns=[1, "1"])), "rep"),
    (lambda: DiscreteDataset.from_pandas(pd.DataFrame({"a": ["x", None]})), "missing"),
    (lambda: ContinuousDataset(np.zeros(5), ["a"]), "2-D"),
    (lambda: ContinuousDataset(np.zeros((4, 3)), ["a", "b"]), "names"),
    (lambda: ContinuousDataset([[1.0, np.nan]], ["a", "b"]), "non-finite"),
    (lambda: ContinuousDataset([[1.0, -np.inf]], ["a", "b"]), "non-finite"),
    (lambda: ContinuousDataset(np.zeros((2, 2)), ["a", "a"]), "repeated"),
    (lambda: ContinuousDataset.from_pandas(pd.DataFrame({"x": [1.0], "l": ["a"]})), "numeric"),
]


def test_malformed_tables_are_refused() -> None:
    for build, match in MALFORMED:
        with pytest.raises(ValueError, match=match):
            build()


def test_a_blank_csv_field_is_missing_data(tmp_path: Path) -> None:
    for text in ("A,B\nx,y\n,z\n", "A,B\nx,y\n  ,z\n"):
        (tmp_path / "bad.csv").write_text(text)
        with pytest.raises(ValueError, match="missing"):
            DiscreteDataset.from_csv(tmp_path / "bad.csv")
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [0.5, 0.25]})
    frame.to_csv(tmp_path / "c.csv", index=False)
    assert np.array_equal(ContinuousDataset.from_csv(tmp_path / "c.csv").values, frame.values)


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
