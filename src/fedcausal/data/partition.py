"""Client partitioning strategies for federated datasets.

This module splits a dataset into per-client datasets under two regimes:

- :func:`contiguous_split` — contiguous row blocks in original order, with
  the block boundaries the original Java implementation uses.
- :func:`dirichlet_split` — a Dirichlet-distribution-based partitioner for
  simulating non-IID federated clients, either by overall row-count skew
  ("quantity skew") or by skewing the distribution of one variable's
  levels across clients ("value skew").

Both functions are generic over :class:`Partitionable`, the minimal
protocol they need from a dataset, so they work on either
:class:`~fedcausal.data.dataset.DiscreteDataset` or
:class:`~fedcausal.data.continuous.ContinuousDataset` without a cast: the
type checker infers ``list[DiscreteDataset]`` or ``list[ContinuousDataset]``
from the argument passed in.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import numpy as np

from fedcausal.data.dataset import DiscreteDataset

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["Partitionable", "contiguous_split", "dirichlet_split"]


@runtime_checkable
class Partitionable(Protocol):
    """A dataset that can be split by rows across federated clients.

    Both :class:`~fedcausal.data.DiscreteDataset` and
    :class:`~fedcausal.data.ContinuousDataset` satisfy this. The partitioners
    need nothing else from a dataset but `n_rows`, `n_vars`, and `subset`, so
    typing them against this protocol is what lets one implementation serve
    both method families.
    """

    @property
    def n_rows(self) -> int:
        """int: Number of rows (observations)."""
        ...

    @property
    def n_vars(self) -> int:
        """int: Number of variables (columns)."""
        ...

    def subset(self, rows: np.ndarray) -> Self:
        """Return a new dataset holding only `rows`."""
        ...


DatasetT = TypeVar("DatasetT", bound=Partitionable)


def contiguous_split(ds: DatasetT, n_clients: int) -> list[DatasetT]:
    """Split a dataset into contiguous row blocks, in original order.

    Each client receives one contiguous block, and row order is preserved
    both within and across blocks, so concatenating the clients' `values` in
    client order rebuilds the original dataset exactly.

    The boundaries deliberately match the original Java implementation rather
    than :func:`numpy.array_split`: with ``bucketSize = n_rows / N`` computed
    in 32-bit float, client ``c`` gets rows
    ``[ceil(c * bucketSize), ceil(c * bucketSize + bucketSize))`` clamped to
    ``n_rows``. The two differ in how they distribute the remainder. For
    ``n_rows=10, N=4`` this gives block sizes ``[3, 2, 3, 2]`` where
    ``array_split`` gives ``[3, 3, 2, 2]``, and matching it is what makes a
    client-for-client comparison against published results meaningful.

    Parameters
    ----------
    ds : DiscreteDataset or ContinuousDataset
        The dataset to split.
    n_clients : int
        Number of clients (blocks) to produce. Must be positive.

    Returns
    -------
    list[DiscreteDataset] or list[ContinuousDataset]
        `n_clients` datasets, in client order, each sharing `ds`'s full
        variable domains (see `DiscreteDataset.subset`). If `n_clients`
        exceeds `ds.n_rows`, the trailing clients receive empty blocks.

    Raises
    ------
    ValueError
        If `n_clients` is not positive.
    """
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive, got {n_clients}")
    n = ds.n_rows
    # 32-bit float bucket size, matching the Java `(float) N` division exactly.
    bucket = np.float32(n) / np.float32(n_clients)
    blocks: list[np.ndarray] = []
    for c in range(n_clients):
        first = math.ceil(float(np.float32(c) * bucket))
        last = min(math.ceil(float(np.float32(c) * bucket + bucket)), n)
        blocks.append(np.arange(first, last))
    return [ds.subset(block) for block in blocks]


def dirichlet_split(
    ds: DatasetT,
    n_clients: int,
    alpha: float,
    seed: int,
    by: int | None = None,
) -> list[DatasetT]:
    """Split a dataset across clients via Dirichlet-distribution draws.

    Two regimes are supported:

    - **Quantity skew** (``by=None``): all rows are shuffled, then sliced
      into `n_clients` contiguous chunks whose sizes are proportional to
      a single draw from a symmetric ``Dirichlet(alpha)`` over the
      clients.
    - **Value skew** (``by=j``): for each level of variable `j`
      independently, that level's rows are distributed across clients
      according to their own draw from a symmetric ``Dirichlet(alpha)``.
      Clients therefore receive different mixes of variable `j`'s
      levels (small `alpha` concentrates each level onto few clients),
      while every row is still assigned to exactly one client.

    In both regimes, every client is guaranteed at least one row. If the
    Dirichlet draw(s) would otherwise leave a client empty, single rows
    are moved deterministically — one at a time, from the currently
    largest client to the lowest-index empty client — until no client is
    empty (see `_ensure_all_nonempty`). This adjustment is only
    satisfiable when ``n_clients <= ds.n_rows``.

    Parameters
    ----------
    ds : DiscreteDataset or ContinuousDataset
        The dataset to split.
    n_clients : int
        Number of clients. Must satisfy ``1 <= n_clients <= ds.n_rows``.
    alpha : float
        Concentration parameter of the symmetric Dirichlet distribution.
        Must be positive. Small values (e.g. ``0.1``) yield highly
        skewed, non-IID splits; large values (e.g. ``1e6``) yield
        near-uniform, near-balanced splits.
    seed : int
        Seed for the random number generator. The split is fully
        reproducible given the same `ds`, `n_clients`, `alpha`, `seed`,
        and `by`.
    by : int, optional
        Column index of the variable to use for value skew. Only
        supported for a `DiscreteDataset`, since it skews the split by a
        variable's discrete levels. If ``None`` (default), quantity skew
        is used instead, which works for either dataset type.

    Returns
    -------
    list[DiscreteDataset] or list[ContinuousDataset]
        `n_clients` datasets forming an exact partition of `ds`'s rows:
        every row is assigned to exactly one client, and every client is
        non-empty. Each shares `ds`'s full variable domains (see
        `DiscreteDataset.subset`).

    Raises
    ------
    ValueError
        If `n_clients` is not positive, if `alpha` is not positive, if
        `n_clients > ds.n_rows` (so the every-client-non-empty guarantee
        is unsatisfiable), or if `by` is out of range.
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


def _round_sizes(total: int, proportions: np.ndarray) -> np.ndarray:
    """Round Dirichlet proportions to integer counts summing to `total`.

    Uses the largest-remainder method: counts are first floored, then
    the largest remaining fractional parts are bumped up by one until
    the counts sum to `total`. Ties in the fractional part are broken
    deterministically in favour of the lowest index (`np.argsort` with a
    stable sort over the negated fractions).

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
    weights: np.ndarray = proportions / proportions.sum()
    raw: np.ndarray = weights * total
    sizes: np.ndarray = np.floor(raw).astype(np.int64)
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
    """Move rows from the largest client(s) to any empty client, in place.

    Repeatedly takes one row from the currently largest client (ties
    broken by lowest index) and gives it to the lowest-index empty
    client, until no client is empty.

    Parameters
    ----------
    client_rows : list[list[int]]
        Per-client row-index lists, modified in place. This always
        terminates provided ``sum(len(r) for r in client_rows) >=
        len(client_rows)``: whenever some client is empty, the total row
        count carried by the remaining, non-empty clients must exceed
        their count, so at least one of them holds 2 or more rows and
        can safely give one up without itself becoming empty.
    """
    while True:
        empty = [c for c, rows in enumerate(client_rows) if not rows]
        if not empty:
            return
        target = empty[0]
        donor = max(range(len(client_rows)), key=lambda c: len(client_rows[c]))
        client_rows[target].append(client_rows[donor].pop())
