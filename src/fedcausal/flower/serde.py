"""Serialization of DAGs and control scalars for the Flower transport.

A DAG travels as a single ``(2, n_edges)`` int64 array in an ``ArrayRecord``
(row 0 = sources, row 1 = targets). The node count is *not* recoverable from
that array -- a DAG with isolated nodes is ambiguous -- so it is carried
alongside, as are the other control scalars, in a ``ConfigRecord``. This is the
one place graphs are packed; there are no per-algorithm serializers.

Because edges travel as bare indices, every client must agree on variable
order. :func:`schema_fingerprint` reduces a dataset's names and cardinalities
to one string so the server can check that up front; :func:`names_fingerprint`
is its continuous counterpart, used where there are no cardinalities to fold
in, and `schema_fingerprint` is built on top of it.

Fourth-order raw moment sums (the sufficient statistics of the FedRCD family,
see :mod:`fedcausal.lingam.cumulants`) travel the same way a DAG does: a fixed
list of arrays in an ``ArrayRecord``, with the sample count carried separately
as a control scalar. :func:`stats_to_record` and :func:`record_to_stats` pack
and unpack that list.
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

# This tuple IS the wire format: append-only, never reorder. `stats_to_record`
# and `record_to_stats` agree on which array is which purely by position, so
# reordering these names would silently swap the meaning of every field
# between a client and server built against mismatched versions -- and
# nothing would catch it, since both functions derive from the same tuple and
# a round-trip test is structurally blind to a reorder.
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
    r"""Return a short stable digest of a variable-name sequence.

    Two clients produce the same fingerprint exactly when they agree on the
    variables they are learning over, in the same order -- the condition under
    which exchanging column indices between them is meaningful.

    Notes
    -----
    Names are joined with ``\x1f`` and not otherwise escaped, so a name that
    itself contains ``\x1e`` or ``\x1f`` can collide with a different split of
    the same characters across adjacent names (e.g. ``["A\x1e2\x1f2"]`` and
    ``["A\x1e2", "2"]``). Vanishingly unlikely for CSV-derived variable names,
    but variable names must not contain those two characters.
    """
    return hashlib.sha256("\x1f".join(names).encode("utf-8")).hexdigest()[:16]


def schema_fingerprint(data: DiscreteDataset) -> str:
    """Return a short stable digest of `data`'s variable names and cardinalities.

    Two clients produce the same fingerprint exactly when they agree on the
    variables they are learning over, in the same order, with the same numbers
    of levels -- the condition under which exchanging edges as bare node
    indices is meaningful.

    Row counts and the data itself are deliberately excluded: clients are
    supposed to hold different observations of the same variables.
    """
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

    Notes
    -----
    The sample count is a scalar and travels in the message's ``ConfigRecord``
    rather than here, so :func:`record_to_stats` takes it as an argument.
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
