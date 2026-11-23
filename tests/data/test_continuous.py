import numpy as np
import pandas as pd
import pytest

from fedcausal.data import ContinuousDataset


def test_constructor_shape_and_names():
    values = np.arange(12, dtype=float).reshape(4, 3)
    ds = ContinuousDataset(values, ["a", "b", "c"])
    assert ds.n_rows == 4
    assert ds.n_vars == 3
    assert ds.names == ("a", "b", "c")
    assert ds.values.dtype == np.float64


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        ContinuousDataset(np.zeros(5), ["a"])


def test_rejects_name_count_mismatch():
    with pytest.raises(ValueError, match="names must have 3 entries"):
        ContinuousDataset(np.zeros((4, 3)), ["a", "b"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite(bad):
    values = np.array([[1.0, 2.0], [bad, 4.0]])
    with pytest.raises(ValueError, match="non-finite"):
        ContinuousDataset(values, ["a", "b"])


def test_from_pandas_rejects_a_non_numeric_column():
    df = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
    with pytest.raises(ValueError, match="'label' is not numeric"):
        ContinuousDataset.from_pandas(df)


def test_subset_shares_names_and_selects_rows():
    values = np.arange(12, dtype=float).reshape(4, 3)
    ds = ContinuousDataset(values, ["a", "b", "c"])
    sub = ds.subset(np.array([0, 2]))
    assert sub.n_rows == 2
    assert sub.names is ds.names
    np.testing.assert_array_equal(sub.values, values[[0, 2]])


def test_from_pandas_and_from_csv_agree(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.5, 0.25, 0.125]})
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    from_df = ContinuousDataset.from_pandas(df)
    from_csv = ContinuousDataset.from_csv(path)
    np.testing.assert_array_equal(from_df.values, from_csv.values)
    assert from_df.names == from_csv.names == ("x", "y")


def test_repeated_names_are_rejected():
    with pytest.raises(ValueError, match="repeated"):
        ContinuousDataset(np.zeros((2, 2)), ["a", "a"])
    with pytest.raises(ValueError, match="repeated"):
        ContinuousDataset.from_pandas(pd.DataFrame(np.zeros((2, 2)), columns=["a", "a"]))
