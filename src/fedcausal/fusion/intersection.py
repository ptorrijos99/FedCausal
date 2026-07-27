"""Intersection fusion: keep only edges present, consistently, in every DAG.

The strictest fusion operator: keep only what every client agrees on.
The the reference implementation code fixes the topological order of the first DAG, reorients the
other
DAGs' edges to agree with that order, then removes from the first DAG every edge
absent in any reoriented DAG. Because fedcausal DAGs already share the integer
node set ``0..n-1``, this reduces to: keep edge ``(a, b)`` of ``dags[0]`` iff the
skeleton pair ``{a, b}`` is present in every other DAG. The orientation is taken
from ``dags[0]`` (the "randomly first graph", per the reference implementation comment).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.graph import DAG

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["intersection_fusion", "Intersection"]


def intersection_fusion(dags: Sequence[DAG]) -> DAG:
    """Return the skeleton intersection of `dags`, oriented by ``dags[0]``.

    Parameters
    ----------
    dags : Sequence[DAG]
        Input DAGs, all over the same node set ``0..n-1``. Must be non-empty.

    Returns
    -------
    DAG
        A DAG keeping each edge ``(a, b)`` of ``dags[0]`` whose skeleton pair
        ``{a, b}`` appears in every other DAG.

    Raises
    ------
    ValueError
        If `dags` is empty.
    """
    if not dags:
        raise ValueError("intersection_fusion requires at least one DAG")
    base = dags[0]
    skeletons = [{frozenset((u, v)) for (u, v) in d.edges()} for d in dags[1:]]
    out = DAG(base.num_nodes)
    for u, v in base.edges():
        pair = frozenset((u, v))
        if all(pair in sk for sk in skeletons):
            out.add_edge(u, v)
    return out


class Intersection:
    """Skeleton-intersection fusion as a :class:`FusionOperator`."""

    def fuse(self, dags: Sequence[DAG]) -> DAG:
        """Return :func:`intersection_fusion` of `dags`."""
        return intersection_fusion(dags)
