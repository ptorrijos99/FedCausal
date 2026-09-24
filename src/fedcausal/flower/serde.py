"""How DAGs, control values and moment sums travel in Flower messages.

A DAG is a ``(2, n_edges)`` int64 array of sources and targets; the node count
travels separately, since isolated nodes leave no trace in the edges. Moment
sums are a fixed list of arrays with the row count sent separately. Everything
is indexed by column position, so the sites must agree on their variables,
which :func:`schema_fingerprint` and :func:`names_fingerprint` let the server
check.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import ArrayRecord, ConfigRecord

from fedcausal.graph import DAG

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.data import DiscreteDataset
    from fedcausal.lingam.cumulants import RawStats

__all__ = [
    "dag_to_record",
    "record_to_dag",
    "control_config",
    "schema_fingerprint",
    "names_fingerprint",
    "stats_to_record",
    "record_to_stats",
]

# The wire format: arrays are matched by position, so only ever append.
_STATS_KEYS = (
    "sum_x",
    "sum_xx",
    "sum_x3",
    "sum_x1y2",
    "sum_x2y1",
    "sum_x4",
    "sum_x1y3",
    "sum_x3y1",
    "sum_x2y2",
)


def names_fingerprint(names: Sequence[str]) -> str:
    r"""Return a short digest of the variable names, in order.

    Names must not contain the separator characters ``\x1e`` or ``\x1f``.
    """
    return hashlib.sha256("\x1f".join(names).encode("utf-8")).hexdigest()[:16]


def schema_fingerprint(data: DiscreteDataset) -> str:
    """Return a short digest of `data`'s variable names and numbers of levels, in order."""
    composite = [
        f"{name}\x1e{card}"
        for name, card in zip(data.names, data.cardinalities.tolist(), strict=True)
    ]
    return names_fingerprint(composite)


def dag_to_record(dag: DAG) -> ArrayRecord:
    """Pack `dag`'s edges as a ``(2, n_edges)`` int64 ``ArrayRecord``."""
    edges = dag.edges()
    arr = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return ArrayRecord([arr])


def record_to_dag(rec: ArrayRecord, num_nodes: int) -> DAG:
    """Rebuild a `DAG` on `num_nodes` nodes from a :func:`dag_to_record` record."""
    arr = rec.to_numpy_ndarrays()[0]
    edges = [(int(u), int(v)) for u, v in zip(arr[0], arr[1], strict=True)]
    return DAG.from_edges(num_nodes, edges)


def control_config(**scalars: Any) -> ConfigRecord:
    """Wrap control scalars (round, converged, num_nodes, ...) in a ``ConfigRecord``."""
    return ConfigRecord(dict(scalars))


def stats_to_record(stats: RawStats) -> ArrayRecord:
    """Pack fourth-order raw moment sums into an ``ArrayRecord``.

    Parameters
    ----------
    stats : RawStats
        Output of :func:`fedcausal.lingam.cumulants.compute_local_raw_stats`
        with ``order=4``.

    Returns
    -------
    ArrayRecord
        The nine moment arrays in a fixed order; see :func:`record_to_stats`.

    Raises
    ------
    ValueError
        If `stats` lacks the fourth-order entries.
    """
    missing = [key for key in _STATS_KEYS if key not in stats]
    if missing:
        raise ValueError(
            f"stats is missing fourth-order entries {missing}; the transport "
            "requires compute_local_raw_stats(..., order=4)"
        )
    return ArrayRecord([np.ascontiguousarray(stats[key], dtype=np.float64) for key in _STATS_KEYS])


def record_to_stats(rec: ArrayRecord, n: int) -> RawStats:
    """Rebuild raw moment sums from a :func:`stats_to_record` record."""
    arrays = rec.to_numpy_ndarrays()
    stats: RawStats = {"n": int(n)}
    stats.update(dict(zip(_STATS_KEYS, arrays, strict=True)))
    return stats
