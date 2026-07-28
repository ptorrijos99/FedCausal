"""Continuous dataset representation for causal structure learning.

This module provides :class:`ContinuousDataset`, a fully-observed dataset over
real-valued variables. It is the input format consumed by the cumulant-based
methods in :mod:`fedcausal.lingam`, and is the continuous counterpart of
:class:`~fedcausal.data.DiscreteDataset`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["ContinuousDataset"]


class ContinuousDataset:
    """A fully-observed dataset over continuous (real-valued) variables.

    Parameters
    ----------
    values : numpy.ndarray
        Real-valued data, shape ``(n_rows, n_vars)``. Cast to ``float64``.
    names : Sequence[str]
        Variable names, length ``n_vars``. Coerced to an immutable
        ``tuple[str, ...]``.

    Attributes
    ----------
    values : numpy.ndarray
        See Parameters.
    names : tuple[str, ...]
        See Parameters.
    n_rows : int
        Number of rows (observations).
    n_vars : int
        Number of variables (columns).

    Raises
    ------
    ValueError
        If `values` is not 2-D, if `names` does not have exactly `n_vars`
        entries, or if `values` contains any non-finite entry (``NaN`` or
        infinity).

    Notes
    -----
    **Memory layout.** Unlike :class:`~fedcausal.data.DiscreteDataset`, which is
    stored column-major for the BDeu kernel's per-column gathers, this class
    keeps the default C order. The cumulant estimators consume whole-matrix
    products (``X.T @ X``) rather than per-column slices, so the discrete side's
    layout argument does not apply here.

    **No missing values.** Cumulant estimators have no notion of a missing
    entry, so non-finite values are rejected at construction rather than
    producing silently meaningless statistics downstream.

    **Shared, immutable metadata.** `subset` shares `names` by reference with
    its parent, so every subset of the same dataset agrees exactly on variable
    order. That matters in a federation, where edges and statistics travel as
    bare column indices.
    """

    values: npt.NDArray[np.float64]
    names: tuple[str, ...]

    def __init__(self, values: npt.NDArray[np.float64], names: Sequence[str]) -> None:
        arr = np.asarray(values, dtype=np.float64)

        if arr.ndim != 2:
            raise ValueError(f"values must be 2-D (n_rows, n_vars), got shape {arr.shape}")
        n_vars = arr.shape[1]
        if len(names) != n_vars:
            raise ValueError(f"names must have {n_vars} entries, got {len(names)}")
        if arr.size and not bool(np.isfinite(arr).all()):
            raise ValueError(
                "values contains a non-finite entry (NaN or infinity); "
                "ContinuousDataset requires fully-observed finite data"
            )

        self.values = np.ascontiguousarray(arr)
        self.names = tuple(names)

    @property
    def n_rows(self) -> int:
        """int: Number of rows (observations) in the dataset."""
        return int(self.values.shape[0])

    @property
    def n_vars(self) -> int:
        """int: Number of variables (columns) in the dataset."""
        return int(self.values.shape[1])

    def subset(self, rows: npt.NDArray[np.intp] | npt.NDArray[np.bool_]) -> ContinuousDataset:
        """Select a subset of rows, keeping the parent's variable names.

        Parameters
        ----------
        rows : numpy.ndarray
            Row indices (integer array) or a boolean mask of length `n_rows`.

        Returns
        -------
        ContinuousDataset
            A new dataset over the selected rows only. `names` is shared by
            reference with the parent, which is safe because it is a tuple.
        """
        subset_ds: ContinuousDataset = object.__new__(ContinuousDataset)
        subset_ds.values = np.ascontiguousarray(self.values[np.asarray(rows)])
        subset_ds.names = self.names
        return subset_ds

    @classmethod
    def from_arrays(
        cls, values: npt.NDArray[np.float64], names: Sequence[str]
    ) -> ContinuousDataset:
        """Build a dataset directly from an array and names, with full validation.

        Parameters
        ----------
        values : numpy.ndarray
            Real-valued data, shape ``(n_rows, n_vars)``.
        names : Sequence[str]
            Variable names, length ``n_vars``.

        Returns
        -------
        ContinuousDataset
            The validated dataset.

        Raises
        ------
        ValueError
            See :class:`ContinuousDataset`.
        """
        return cls(values, names)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> ContinuousDataset:
        """Build a dataset from a pandas DataFrame of numeric columns.

        Parameters
        ----------
        df : pandas.DataFrame
            One column per variable, one row per observation. Column names
            become `names`.

        Returns
        -------
        ContinuousDataset
            The dataset.

        Raises
        ------
        ValueError
            If any column is non-numeric, or contains a non-finite value.
        """
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"column {col!r} is not numeric; ContinuousDataset requires real-valued columns"
                )
        return cls(df.to_numpy(dtype=np.float64), [str(c) for c in df.columns])

    @classmethod
    def from_csv(cls, path: str | Path) -> ContinuousDataset:
        """Build a dataset from a CSV file of numeric columns.

        The file must have a header row of variable names followed by
        comma-separated numeric values.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the CSV file.

        Returns
        -------
        ContinuousDataset
            The dataset.

        Raises
        ------
        ValueError
            If any column is non-numeric, or contains a non-finite value.
        """
        return cls.from_pandas(pd.read_csv(path))

    def __repr__(self) -> str:
        return (
            f"ContinuousDataset(n_rows={self.n_rows}, n_vars={self.n_vars}, names={self.names!r})"
        )
