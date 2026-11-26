"""Consensus topological order by sink peeling (Puerta et al., 2021)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fedcausal.fusion._kernels import alpha_order_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causalbase.graph import DAG

__all__ = ["alpha_order"]


def alpha_order(dags: Sequence[DAG]) -> list[int]:
    """Return a topological order that the DAGs in `dags` agree on as much as possible.

    The order is built from the sinks up. Each step picks the node whose
    removal costs the fewest edge changes summed over every DAG (arc reversals
    plus the moral edges they force), places it before the nodes already
    ordered, and peels it from every DAG. Ties go to the lowest index.

    Parameters
    ----------
    dags : sequence of DAG
        DAGs over the same nodes. At least one.

    Returns
    -------
    list of int
        A permutation of the nodes, sinks last.
    """
    if not dags:
        raise ValueError("alpha_order needs at least one DAG")
    stack = np.stack([d.adjacency().view(np.uint8) for d in dags])
    return [int(v) for v in alpha_order_kernel(stack)]
