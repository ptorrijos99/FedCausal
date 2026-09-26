"""Intersection fusion: keep only the adjacencies every DAG agrees on.

An edge of ``dags[0]`` survives if its node pair is adjacent, in either
direction, in every other DAG. Orientations come from ``dags[0]``.
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
