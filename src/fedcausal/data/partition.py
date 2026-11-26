"""Split a dataset across simulated sites.

Both splitters work on either dataset type and return the same type they are
given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np
from causalbase.data.dataset import DiscreteDataset

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import Self

__all__ = ["Partitionable", "contiguous_split", "dirichlet_split"]


@runtime_checkable
class Partitionable(Protocol):
    """What the splitters need from a dataset: its size and a way to take rows."""

    @property
    def n_rows(self) -> int:
        """int: Number of rows (observations)."""
        ...

    @property
    def n_vars(self) -> int:
        """int: Number of variables (columns)."""
        ...

    def subset(self, rows: npt.NDArray[np.integer[Any]]) -> Self:
        """Return a new dataset holding only `rows`."""
        ...


DatasetT = TypeVar("DatasetT", bound=Partitionable)


def contiguous_split(ds: DatasetT, n_clients: int) -> list[DatasetT]:
    """Split a dataset into contiguous row blocks, in original order.

    Client ``c`` gets rows ``ceil(c * n / N)`` up to ``ceil((c + 1) * n / N)``,
    so concatenating the blocks in order rebuilds the dataset. The remainder is
    spread across the blocks: ``n=10, N=4`` gives sizes ``[3, 2, 3, 2]``.

    Parameters
    ----------
    ds : DiscreteDataset or ContinuousDataset
        The dataset to split.
    n_clients : int
        Number of clients (blocks) to produce. Must be positive.

    Returns
    -------
    list[DiscreteDataset] or list[ContinuousDataset]
        `n_clients` datasets over the same variables as `ds`. If `n_clients`
        exceeds `ds.n_rows`, some blocks are empty.

    Raises
    ------
    ValueError
        If `n_clients` is not positive.
    """
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive, got {n_clients}")
    bounds = [-(-c * ds.n_rows // n_clients) for c in range(n_clients + 1)]
    return [ds.subset(np.arange(bounds[c], bounds[c + 1])) for c in range(n_clients)]


def dirichlet_split(
    ds: DatasetT,
    n_clients: int,
    alpha: float,
    seed: int,
    by: int | None = None,
) -> list[DatasetT]:
    """Split a dataset across clients via Dirichlet-distribution draws.

    With ``by=None`` the clients' sizes follow one ``Dirichlet(alpha)`` draw
    (quantity skew). With ``by=j`` each level of variable `j` is spread by its
    own draw, so clients see different mixes of that variable (value skew).
    Every client gets at least one row: rows move from the largest client to
    any that would be empty.

    Parameters
    ----------
    ds : DiscreteDataset or ContinuousDataset
        The dataset to split.
    n_clients : int
        Number of clients. Must satisfy ``1 <= n_clients <= ds.n_rows``.
    alpha : float
        Dirichlet concentration, positive. Small values (``0.1``) give very
        uneven splits, large ones (``1e6``) nearly even ones.
    seed : int
        Random seed. The same arguments always give the same split.
    by : int, optional
        Column to skew by. Only for a `DiscreteDataset`. Default ``None``,
        quantity skew.

    Returns
    -------
    list[DiscreteDataset] or list[ContinuousDataset]
        `n_clients` non-empty datasets that together hold every row of `ds`
        exactly once.

    Raises
    ------
    ValueError
        If `n_clients` or `alpha` is not positive, if `n_clients` exceeds
        ``ds.n_rows``, or if `by` is out of range.
    TypeError
        If `by` is given but `ds` is not a `DiscreteDataset`.
    """
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive, got {n_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if n_clients > ds.n_rows:
        raise ValueError(
            f"cannot guarantee >=1 row per client: n_clients={n_clients} > ds.n_rows={ds.n_rows}"
        )
    if by is not None and not (0 <= by < ds.n_vars):
        raise ValueError(f"by={by} is out of range for {ds.n_vars} variables")

    rng = np.random.default_rng(seed)
    client_rows: list[list[int]]
    if by is None:
        client_rows = _quantity_skew_rows(ds.n_rows, n_clients, alpha, rng)
    else:
        if not isinstance(ds, DiscreteDataset):
            raise TypeError(f"by is only supported for a DiscreteDataset, got {type(ds).__name__}")
        client_rows = _value_skew_rows(ds, n_clients, alpha, by, rng)
    _ensure_all_nonempty(client_rows)
    return [ds.subset(np.array(sorted(rows), dtype=np.int64)) for rows in client_rows]


def _round_sizes(total: int, proportions: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Round Dirichlet proportions to integer counts summing to `total`.

    Largest-remainder rounding, with ties going to the lowest index.

    Parameters
    ----------
    total : int
        Target sum of the returned counts.
    proportions : np.ndarray
        Nonnegative weights (need not already sum to 1).

    Returns
    -------
    np.ndarray
        Nonnegative integer counts (``int64``), one per entry of
        `proportions`, summing exactly to `total`.
    """
    weights: npt.NDArray[np.float64] = proportions / proportions.sum()
    raw: npt.NDArray[np.float64] = weights * total
    sizes: npt.NDArray[np.int64] = np.floor(raw).astype(np.int64)
    remainder = int(total - sizes.sum())
    if remainder > 0:
        frac = raw - sizes
        order = np.argsort(-frac, kind="stable")
        sizes[order[:remainder]] += 1
    return sizes


def _quantity_skew_rows(
    n_rows: int, n_clients: int, alpha: float, rng: np.random.Generator
) -> list[list[int]]:
    """Partition ``range(n_rows)`` into `n_clients` groups by quantity skew."""
    shuffled = rng.permutation(n_rows)
    proportions = rng.dirichlet(np.full(n_clients, alpha))
    sizes = _round_sizes(n_rows, proportions)
    client_rows: list[list[int]] = []
    start = 0
    for size in sizes:
        end = start + int(size)
        client_rows.append(shuffled[start:end].tolist())
        start = end
    return client_rows


def _value_skew_rows(
    ds: DiscreteDataset,
    n_clients: int,
    alpha: float,
    by: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Partition rows into `n_clients` groups by value skew on column `by`."""
    client_rows: list[list[int]] = [[] for _ in range(n_clients)]
    column = ds.values[:, by]
    for level in range(int(ds.cardinalities[by])):
        level_rows = np.flatnonzero(column == level)
        if level_rows.size == 0:
            continue
        shuffled = rng.permutation(level_rows)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        sizes = _round_sizes(int(level_rows.size), proportions)
        start = 0
        for c, size in enumerate(sizes):
            end = start + int(size)
            client_rows[c].extend(shuffled[start:end].tolist())
            start = end
    return client_rows


def _ensure_all_nonempty(client_rows: list[list[int]]) -> None:
    """Move one row at a time from the largest client to an empty one, in place.

    Terminates whenever there are at least as many rows as clients: while a
    client is empty, some other client holds two or more rows.
    """
    while True:
        empty = [c for c, rows in enumerate(client_rows) if not rows]
        if not empty:
            return
        target = empty[0]
        donor = max(range(len(client_rows)), key=lambda c: len(client_rows[c]))
        client_rows[target].append(client_rows[donor].pop())
