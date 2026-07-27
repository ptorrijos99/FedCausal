"""Serialization of DAGs and control scalars for the Flower transport.

A DAG travels as a single ``(2, n_edges)`` int64 array in an ``ArrayRecord``
(row 0 = sources, row 1 = targets). The node count is *not* recoverable from
that array -- a DAG with isolated nodes is ambiguous -- so it is carried
alongside, as are the other control scalars, in a ``ConfigRecord``. This is the
one place graphs are packed; there are no per-algorithm serializers.

Because edges travel as bare indices, every client must agree on variable
order. :func:`schema_fingerprint` reduces a dataset's names and cardinalities
to one string so the server can check that up front.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import ArrayRecord, ConfigRecord

from fedcausal.graph import DAG

if TYPE_CHECKING:
    from fedcausal.data import DiscreteDataset

__all__ = ["dag_to_record", "record_to_dag", "control_config", "schema_fingerprint"]


def schema_fingerprint(data: DiscreteDataset) -> str:
    """Return a short stable digest of `data`'s variable names and cardinalities.

    Two clients produce the same fingerprint exactly when they agree on the
    variables they are learning over, in the same order, with the same numbers
    of levels -- the condition under which exchanging edges as bare node
    indices is meaningful.

    Row counts and the data itself are deliberately excluded: clients are
    supposed to hold different observations of the same variables.
    """
    payload = "\x1f".join(
        f"{name}\x1e{card}"
        for name, card in zip(data.names, data.cardinalities.tolist(), strict=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
