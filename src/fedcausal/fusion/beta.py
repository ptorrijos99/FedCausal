"""Transform a DAG into a minimal I-map consistent with a target order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fedcausal.fusion._kernels import beta_to_alpha_kernel
from fedcausal.graph import DAG

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["beta_to_alpha"]


def beta_to_alpha(dag: DAG, order: Sequence[int]) -> DAG:
    """Return `dag` re-expressed as a minimal I-map whose arcs all follow `order`.

    Nodes are bubbled into `order` by adjacent swaps; a swap across an arc
    reverses it and adds the moral edges that keep every conditional
    independence of `dag`, so the result implies no independence `dag` does
    not.

    Parameters
    ----------
    dag : DAG
        Not modified.
    order : sequence of int
        The target topological order, e.g. from
        :func:`~fedcausal.fusion.alpha_order`.

    Returns
    -------
    DAG
    """
    rank = np.empty(dag.num_nodes, dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(dag.num_nodes)
    return DAG.from_amat(beta_to_alpha_kernel(dag.adjacency().view(np.uint8), rank))
