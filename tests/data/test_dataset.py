from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedcausal.data import DiscreteDataset

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
TOY_CSV = FIXTURES / "toy.csv"
NONE_CATEGORY_CSV = FIXTURES / "none_category.csv"

# Expected sorted-lexicographic encoding of tests/fixtures/toy.csv:
#   A: b,a,c,a,b,a,c,b -> sorted levels [a,b,c] -> codes [1,0,2,0,1,0,2,1]
#   B: lo,hi,lo,hi,lo,hi,hi,lo -> sorted levels [hi,lo] -> codes [1,0,1,0,1,0,0,1]
#   C: q,p,r,q,p,r,q,p -> sorted levels [p,q,r] -> codes [1,0,2,1,0,2,1,0]
EXPECTED_VALUES = np.array(
    [
        [1, 1, 1],
        [0, 0, 0],
        [2, 1, 2],
        [0, 0, 1],
        [1, 1, 0],
        [0, 0, 2],
        [2, 0, 1],
        [1, 1, 0],
    ],
    dtype=np.int32,
)
EXPECTED_CARDINALITIES = np.array([3, 2, 3], dtype=np.int32)
EXPECTED_CATEGORIES = (("a", "b", "c"), ("hi", "lo"), ("p", "q", "r"))


def test_from_csv_shapes_and_names() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    assert ds.n_rows == 8
    assert ds.n_vars == 3
    assert ds.names == ("A", "B", "C")
    assert ds.values.shape == (8, 3)


def test_from_csv_accepts_path_as_str() -> None:
    ds = DiscreteDataset.from_csv(str(TOY_CSV))
    assert ds.n_rows == 8


def test_from_csv_sorted_lexicographic_encoding() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    assert ds.categories == EXPECTED_CATEGORIES
    assert np.array_equal(ds.cardinalities, EXPECTED_CARDINALITIES)
    assert np.array_equal(ds.values, EXPECTED_VALUES)


def test_from_csv_dtypes_are_int32() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    assert ds.values.dtype == np.int32
    assert ds.cardinalities.dtype == np.int32


def test_from_csv_rejects_missing_values(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("A,B\nx,y\n,z\n")
    with pytest.raises(ValueError, match="missing"):
        DiscreteDataset.from_csv(bad_csv)


def test_from_csv_rejects_whitespace_only_field(tmp_path: Path) -> None:
    # A field that is present but blank/whitespace-only is still genuinely
    # missing data, even though pandas' default NA-token sniffing is
    # disabled (which is what lets the literal string "None" survive as an
    # ordinary category -- see test_from_csv_preserves_none_as_category).
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("A,B\nx,y\n  ,z\n")
    with pytest.raises(ValueError, match="missing"):
        DiscreteDataset.from_csv(bad_csv)


def test_from_csv_preserves_none_as_category() -> None:
    # Regression test: pandas' default `read_csv` NA tokens include the
    # literal string "None", which used to be silently coerced to NaN and
    # then rejected by the missing-value guard. Some reference networks
    # (e.g. child's DuctFlow, insurance's ThisCarDam) use "None" as a
    # genuine category label, exactly as Weka/Tetrad read it.
    ds = DiscreteDataset.from_csv(NONE_CATEGORY_CSV)
    assert ds.n_rows == 4
    a_idx = ds.names.index("A")
    assert "None" in ds.categories[a_idx]
    none_code = ds.categories[a_idx].index("None")
    assert int(np.sum(ds.values[:, a_idx] == none_code)) == 2


def test_from_pandas_matches_from_csv() -> None:
    df = pd.read_csv(TOY_CSV, dtype=str)
    ds = DiscreteDataset.from_pandas(df)
    assert ds.categories == EXPECTED_CATEGORIES
    assert np.array_equal(ds.values, EXPECTED_VALUES)


def test_from_pandas_rejects_nan() -> None:
    df = pd.DataFrame({"A": ["x", None], "B": ["y", "z"]})
    with pytest.raises(ValueError, match="missing"):
        DiscreteDataset.from_pandas(df)


def test_from_arrays_round_trip() -> None:
    values = np.array([[0, 1], [1, 0]])
    cardinalities = np.array([2, 2])
    names = ["X", "Y"]
    categories = [["lo", "hi"], ["no", "yes"]]
    ds = DiscreteDataset.from_arrays(values, cardinalities, names, categories)
    assert ds.n_rows == 2
    assert ds.n_vars == 2
    assert ds.values.dtype == np.int32
    assert ds.cardinalities.dtype == np.int32
    assert ds.names == tuple(names)
    assert ds.categories == tuple(tuple(cats) for cats in categories)


def test_from_arrays_rejects_non_2d_values() -> None:
    with pytest.raises(ValueError):
        DiscreteDataset.from_arrays(
            np.array([0, 1, 2]),
            np.array([3]),
            ["A"],
            [["a", "b", "c"]],
        )


def test_from_arrays_rejects_cardinality_length_mismatch() -> None:
    with pytest.raises(ValueError):
        DiscreteDataset.from_arrays(
            np.zeros((3, 2), dtype=np.int32),
            np.array([2]),  # should have length 2
            ["A", "B"],
            [["a", "b"], ["c", "d"]],
        )


def test_from_arrays_rejects_categories_length_mismatch_with_cardinalities() -> None:
    with pytest.raises(ValueError):
        DiscreteDataset.from_arrays(
            np.zeros((3, 1), dtype=np.int32),
            np.array([3]),
            ["A"],
            [["a", "b"]],  # only 2 levels, cardinality says 3
        )


def test_from_arrays_rejects_names_length_mismatch() -> None:
    with pytest.raises(ValueError):
        DiscreteDataset.from_arrays(
            np.zeros((3, 2), dtype=np.int32),
            np.array([2, 2]),
            ["A"],  # should have length 2
            [["a", "b"], ["c", "d"]],
        )


def test_from_arrays_rejects_out_of_range_code() -> None:
    with pytest.raises(ValueError):
        DiscreteDataset.from_arrays(
            np.array([[0], [2]], dtype=np.int32),  # code 2 invalid for cardinality 2
            np.array([2]),
            ["A"],
            [["a", "b"]],
        )


def test_subset_keeps_full_cardinalities_when_level_absent() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    # Rows 2 and 6 (0-indexed) are the only rows where A == "c" (code 2).
    keep_rows = np.array([0, 1, 3, 4, 5, 7])
    sub = ds.subset(keep_rows)

    assert sub.n_rows == 6
    # Level "c" of A is absent from the subset's actual values...
    assert 2 not in sub.values[:, 0]
    # ...but the full parent cardinalities and categories are still reported.
    assert np.array_equal(sub.cardinalities, ds.cardinalities)
    assert sub.categories == ds.categories
    assert sub.names == ds.names
    assert np.array_equal(sub.values, ds.values[keep_rows])


def test_subset_shares_metadata_objects_with_parent() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    sub = ds.subset(np.array([0, 1, 2]))
    # Metadata is shared by reference (not recomputed/copied) so that
    # federated clients built from the same parent agree on variable
    # domains bit-for-bit. `names` and `categories` are tuples (immutable),
    # so this sharing can never let an in-place edit on one subset corrupt
    # a sibling or the parent.
    assert sub.cardinalities is ds.cardinalities
    assert sub.categories is ds.categories
    assert sub.names is ds.names
    assert isinstance(ds.names, tuple)
    assert isinstance(ds.categories, tuple)
    assert all(isinstance(cats, tuple) for cats in ds.categories)


def test_subset_does_not_mutate_parent_values() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    sub = ds.subset(np.array([0, 1]))
    sub.values[0, 0] = 99
    assert ds.values[0, 0] != 99


def test_cardinalities_is_read_only() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    with pytest.raises(ValueError, match="read-only"):
        ds.cardinalities[0] = 99


def test_from_arrays_copies_caller_cardinalities_array() -> None:
    values = np.array([[0, 1], [1, 0]])
    caller_cardinalities = np.array([2, 2])
    ds = DiscreteDataset.from_arrays(
        values, caller_cardinalities, ["X", "Y"], [["a", "b"], ["c", "d"]]
    )
    caller_cardinalities[0] = 99  # mutating the caller's own array afterwards...
    assert ds.cardinalities[0] == 2  # ...must not affect the stored dataset


def test_subset_with_boolean_mask() -> None:
    ds = DiscreteDataset.from_csv(TOY_CSV)
    mask = ds.values[:, 0] != 2  # exclude rows where A == "c"
    sub = ds.subset(mask)
    assert sub.n_rows == 6
    assert np.array_equal(sub.cardinalities, ds.cardinalities)


def test_values_is_fortran_ordered() -> None:
    """values is stored column-major so the BDeu kernel's column gathers are contiguous."""
    ds = DiscreteDataset.from_arrays(
        np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        ["a", "b"],
        [["x", "y"], ["p", "q"]],
    )
    assert ds.values.flags["F_CONTIGUOUS"]
    assert ds.values.shape == (3, 2)
    assert ds.values.dtype == np.int32


def test_subset_preserves_fortran_order() -> None:
    """subset() re-imposes column-major order (fancy row-indexing returns C-order)."""
    ds = DiscreteDataset.from_arrays(
        np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        ["a", "b"],
        [["x", "y"], ["p", "q"]],
    )
    sub = ds.subset(np.array([0, 2, 3]))
    assert sub.values.flags["F_CONTIGUOUS"]
    # And the data is still correct after the layout round-trip.
    assert sub.values.tolist() == [[0, 1], [0, 0], [1, 1]]
