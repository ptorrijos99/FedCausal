"""Discrete dataset representation for causal structure learning.

This module provides :class:`DiscreteDataset`, a fully-observed, purely
categorical dataset over integer-coded variables. It is the input format
consumed by the BDeu score and by the federated client partitioners
elsewhere in ``fedcausal``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["DiscreteDataset"]


class DiscreteDataset:
    """A fully-observed dataset over discrete (categorical) variables.

    Parameters
    ----------
    values : np.ndarray
        Integer-coded data, shape ``(n_rows, n_vars)``. Cast to
        ``int32``. Entry ``values[i, j]`` is the code of row `i`'s level
        of variable `j`, an index into ``categories[j]``.
    cardinalities : np.ndarray
        Number of levels per variable, shape ``(n_vars,)``. Cast to
        ``int32``.
    names : Sequence[str]
        Variable names, length ``n_vars``. Coerced to an immutable
        ``tuple[str, ...]``.
    categories : Sequence[Sequence[str]]
        Level labels per variable; ``categories[j][i]`` is the label of
        level `i` of variable `j`, so ``len(categories[j]) ==
        cardinalities[j]``. Coerced to an immutable
        ``tuple[tuple[str, ...], ...]``.

    Attributes
    ----------
    values : np.ndarray
        See Parameters.
    cardinalities : np.ndarray
        See Parameters.
    names : tuple[str, ...]
        See Parameters.
    categories : tuple[tuple[str, ...], ...]
        See Parameters.
    n_rows : int
        Number of rows (observations).
    n_vars : int
        Number of variables (columns).

    Raises
    ------
    ValueError
        If `values` is not 2-D, if `cardinalities`, `names`, or
        `categories` do not each have exactly `n_vars` entries, if some
        ``categories[j]`` does not have exactly ``cardinalities[j]``
        entries, or if any entry of `values` falls outside
        ``[0, cardinalities[j])`` for its column.

    Notes
    -----
    **Level encoding convention.** Within each column `j`, levels are
    always encoded **sorted lexicographically** (Python's default string
    ordering over ``categories[j]``), independent of the order in which
    levels first appear in the source data. This is a deliberate,
    documented convention chosen for reproducibility across Python
    clients.

    Other tools assign level codes by first appearance instead, so level
    *indices* are not expected to agree across implementations. Only
    quantities computed from them (contingency-table counts, BDeu
    sufficient statistics) need to match, and those are
    invariant to which integer label is assigned to which level.

    **Memory layout.** `values` is stored **column-major** (Fortran order)
    so the BDeu counting kernel's per-column gathers (``values[:, node]``,
    ``values[:, p]``) are contiguous — measured ~1.6x on the kernel at
    ~1000-variable shape. The array is logically unchanged ``(n_rows,
    n_vars)`` int32; do not re-C-order it. `subset` re-imposes the layout,
    since fancy row-indexing returns a C-contiguous copy.

    **Shared, immutable metadata.** `subset` shares `names`,
    `cardinalities`, and `categories` by reference with its parent
    (rather than copying them), so that every subset of the same
    dataset agrees bit-for-bit on the variable domains. Each of these
    is immutable: `cardinalities` is a read-only array (attempting to
    write into it raises `ValueError`), and `names`/`categories` are
    plain tuples, which have no in-place mutation API. This means an
    in-place edit on one subset can never silently corrupt its siblings
    or the parent.
    """

    values: np.ndarray
    cardinalities: np.ndarray
    names: tuple[str, ...]
    categories: tuple[tuple[str, ...], ...]

    def __init__(
        self,
        values: np.ndarray,
        cardinalities: np.ndarray,
        names: Sequence[str],
        categories: Sequence[Sequence[str]],
    ) -> None:
        values = np.asarray(values, dtype=np.int32)
        # Always a fresh, independent copy (never aliases the caller's array):
        # `cardinalities` is small metadata that is shared by reference across
        # every `subset()` of this dataset, so it is made read-only below to
        # guard against accidental in-place mutation corrupting sibling
        # subsets or the parent.
        cardinalities = np.array(cardinalities, dtype=np.int32)

        if values.ndim != 2:
            raise ValueError(f"values must be 2-D (n_rows, n_vars), got shape {values.shape}")
        n_rows, n_vars = values.shape

        if cardinalities.shape != (n_vars,):
            raise ValueError(
                f"cardinalities must have shape ({n_vars},) to match values' "
                f"{n_vars} columns, got shape {cardinalities.shape}"
            )
        if len(names) != n_vars:
            raise ValueError(f"names must have {n_vars} entries, got {len(names)}")
        if len(categories) != n_vars:
            raise ValueError(f"categories must have {n_vars} entries, got {len(categories)}")
        for j, cats in enumerate(categories):
            if len(cats) != int(cardinalities[j]):
                raise ValueError(
                    f"categories[{j}] has {len(cats)} levels but "
                    f"cardinalities[{j}] = {int(cardinalities[j])}"
                )
        if n_rows > 0 and n_vars > 0:
            out_of_range = bool(values.min() < 0 or np.any(values >= cardinalities[np.newaxis, :]))
            if out_of_range:
                raise ValueError(
                    "values contains a category code outside [0, cardinalities[j]) "
                    "for at least one column"
                )

        cardinalities.setflags(write=False)
        # Column-major so the BDeu kernel's per-column gathers are contiguous
        # (see the class Notes). `asfortranarray` copies a C-ordered input and
        # is a no-op when already Fortran-ordered.
        self.values = np.asfortranarray(values)
        self.cardinalities = cardinalities
        self.names = tuple(names)
        self.categories = tuple(tuple(cats) for cats in categories)

    @property
    def n_rows(self) -> int:
        """int: Number of rows (observations) in the dataset."""
        return int(self.values.shape[0])

    @property
    def n_vars(self) -> int:
        """int: Number of variables (columns) in the dataset."""
        return int(self.values.shape[1])

    def subset(self, rows: np.ndarray) -> DiscreteDataset:
        """Select a subset of rows, keeping the full parent variable domains.

        Parameters
        ----------
        rows : np.ndarray
            Row indices (integer array) or a boolean mask of length
            `n_rows`, selecting the rows to keep.

        Returns
        -------
        DiscreteDataset
            A new dataset over the selected rows only. `names`,
            `cardinalities`, and `categories` are shared by reference
            with the parent dataset (safe, since all three are
            immutable) rather than recomputed from the subset's actual
            values, so a level absent from the subset still counts
            toward its variable's cardinality. This is essential for
            federated clients: every client must agree on the same
            variable domains, even if a particular client's local
            partition happens not to contain every level.
        """
        row_index = np.asarray(rows)
        subset_ds: DiscreteDataset = object.__new__(DiscreteDataset)
        # Fancy row-indexing returns a C-contiguous copy; re-impose column-major
        # so subsets keep the kernel's contiguous-gather advantage.
        subset_ds.values = np.asfortranarray(self.values[row_index])
        subset_ds.cardinalities = self.cardinalities
        subset_ds.names = self.names
        subset_ds.categories = self.categories
        return subset_ds

    @classmethod
    def from_arrays(
        cls,
        values: np.ndarray,
        cardinalities: np.ndarray,
        names: Sequence[str],
        categories: Sequence[Sequence[str]],
    ) -> DiscreteDataset:
        """Build a dataset directly from arrays, with full validation.

        Parameters
        ----------
        values : np.ndarray
            Integer-coded data, shape ``(n_rows, n_vars)``.
        cardinalities : np.ndarray
            Number of levels per variable, shape ``(n_vars,)``.
        names : Sequence[str]
            Variable names, length ``n_vars``. Coerced to a tuple.
        categories : Sequence[Sequence[str]]
            Level labels per variable. Coerced to a tuple of tuples.

        Returns
        -------
        DiscreteDataset
            The validated dataset.

        Raises
        ------
        ValueError
            See `DiscreteDataset`.
        """
        return cls(values, cardinalities, names, categories)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> DiscreteDataset:
        """Build a dataset from a pandas DataFrame of categorical columns.

        Each column is encoded independently: its distinct values (read
        as strings) become that variable's levels, sorted
        lexicographically (see the class-level Notes on level encoding).

        Parameters
        ----------
        df : pandas.DataFrame
            One column per variable, one row per observation. Column
            names become `names`. Values are converted to `str` before
            encoding (so, e.g., booleans become ``"True"``/``"False"``).

        Returns
        -------
        DiscreteDataset
            The encoded dataset.

        Raises
        ------
        ValueError
            If any column contains missing values (``NaN``/``None``), or a
            field that is empty or holds only whitespace.
        """
        names = [str(c) for c in df.columns]
        n_rows = len(df)
        n_vars = len(names)
        values = np.empty((n_rows, n_vars), dtype=np.int32)
        cardinalities = np.empty(n_vars, dtype=np.int32)
        categories: list[list[str]] = []
        for j, col in enumerate(df.columns):
            series = df[col]
            if bool(series.isna().any()):
                raise ValueError(
                    f"column {col!r} contains missing values; DiscreteDataset "
                    "requires fully-observed discrete data"
                )
            str_values = series.astype(str)
            # Checked here rather than in `from_csv` so that both entry points
            # enforce it: an empty string that survives becomes a category of
            # its own, silently inflating the column's cardinality.
            if bool((str_values.str.strip() == "").any()):
                raise ValueError(
                    f"column {col!r} contains missing values (an empty or "
                    "whitespace-only field); DiscreteDataset requires "
                    "fully-observed discrete data"
                )
            levels = sorted(str_values.unique().tolist())
            coded = pd.Categorical(str_values, categories=levels)
            values[:, j] = coded.codes.astype(np.int32)
            cardinalities[j] = len(levels)
            categories.append(levels)
        return cls(values, cardinalities, names, categories)

    @classmethod
    def from_csv(cls, path: str | Path) -> DiscreteDataset:
        """Build a dataset from a CSV file of categorical columns.

        The file must have a header row of variable names followed by
        comma-separated categorical string values — the same layout as
        the reference datasets. All values are read as strings (never inferred as
        numeric), so category labels are never silently reinterpreted.

        **NA policy.** pandas' default NA-token sniffing is disabled
        (``keep_default_na=False, na_filter=False``), so literal strings
        such as ``"None"``, ``"NA"``, ``"NULL"``, or ``"NaN"`` are read
        back as ordinary category labels rather than being coerced to a
        missing value. This matters because some benchmark networks use
        ``None`` as a genuine level (``child``'s ``DuctFlow``,
        ``insurance``'s ``ThisCarDam``). A field that is empty or holds
        only whitespace is still treated as missing and rejected with a
        `ValueError`, since fedcausal requires fully-observed data.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the CSV file.

        Returns
        -------
        DiscreteDataset
            The encoded dataset.

        Raises
        ------
        ValueError
            If any column contains an empty or whitespace-only field.
        """
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
        return cls.from_pandas(df)

    def __repr__(self) -> str:
        return f"DiscreteDataset(n_rows={self.n_rows}, n_vars={self.n_vars}, names={self.names!r})"
