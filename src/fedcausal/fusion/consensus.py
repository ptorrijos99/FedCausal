"""Consensus-union fusion and its budgeted variants.

The union transforms every input into one consensus order and unions the
parent sets, so the result is acyclic by construction. The budgeted variants
rank the transformed edges by how many inputs contain them and add them under
a cap. Ties in that ranking break by ``(child, parent)``, so the result never
depends on dictionary order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedcausal.fusion.alpha import alpha_order
from fedcausal.fusion.beta import beta_to_alpha
from fedcausal.graph import DAG

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "consensus_union",
    "ConsensusUnion",
    "edge_frequency",
    "max_edges",
    "max_frequency",
    "consensus_pct",
    "max_parents",
    "MaxEdges",
    "MaxFrequency",
    "Consensus",
    "MaxParents",
]


def _transform_all(dags: Sequence[DAG]) -> tuple[list[int], list[DAG]]:
    """Return the consensus order and every input transformed into it.

    Parameters
    ----------
    dags : Sequence[DAG]
        Input DAGs over the same node set. Must be non-empty.

    Returns
    -------
    tuple[list[int], list[DAG]]
        ``(alpha, transformed)`` where ``alpha`` is the consensus order and
        ``transformed[i] = beta_to_alpha(dags[i], alpha)``.

    Raises
    ------
    ValueError
        If `dags` is empty.
    """
    if not dags:
        raise ValueError("consensus fusion needs at least one DAG")
    alpha = alpha_order(dags)
    return alpha, [beta_to_alpha(d, alpha) for d in dags]


def consensus_union(dags: Sequence[DAG]) -> DAG:
    """Return the full consensus union of `dags` (Puerta et al. 2021).

    The result is acyclic, since every transformed input shares one order.

    Parameters
    ----------
    dags : Sequence[DAG]
        Input DAGs over the same node set. Must be non-empty.

    Returns
    -------
    DAG
        The consensus-union DAG.

    Raises
    ------
    ValueError
        If `dags` is empty.
    """
    _alpha, transformed = _transform_all(dags)
    out = DAG(dags[0].num_nodes)
    for child in range(out.num_nodes):
        for d in transformed:
            for parent in sorted(d.parents(child)):
                out.add_edge(parent, child)
    return out


class ConsensusUnion:
    """`FusionOperator` for the full consensus union (no edge budget)."""

    def fuse(self, dags: Sequence[DAG]) -> DAG:
        """Return :func:`consensus_union` of `dags`."""
        return consensus_union(dags)


def edge_frequency(transformed: Sequence[DAG]) -> list[tuple[tuple[int, int], int]]:
    """Return ``((parent, child), count)`` over transformed DAGs, most frequent first."""
    counts: dict[tuple[int, int], int] = {}
    for d in transformed:
        for parent, child in d.edges():
            counts[parent, child] = counts.get((parent, child), 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0]))


def _int_limit(limit: int | str, edge_counts: list[int]) -> int:
    """Resolve a `max_edges` budget, allowing ``"Mean"``/``"Max"``/``"Min"``."""
    if isinstance(limit, int):
        return limit
    if limit == "Mean":
        return sum(edge_counts) // len(edge_counts)
    if limit == "Max":
        return max(edge_counts)
    if limit == "Min":
        return min(edge_counts)
    raise ValueError(f"unknown edge limit {limit!r}; expected int or Mean/Max/Min")


def max_edges(dags: Sequence[DAG], limit: int | str = "Max") -> DAG:
    """Union the highest-frequency edges up to a total-edge budget.

    Parameters
    ----------
    dags : Sequence[DAG]
        Input DAGs over the same node set.
    limit : int | str, optional
        The budget: an int, or the ``"Mean"``, ``"Max"`` or ``"Min"`` of the
        transformed inputs' edge counts. Default ``"Max"``.

    Returns
    -------
    DAG
        The budgeted consensus DAG.
    """
    _alpha, transformed = _transform_all(dags)
    lim = _int_limit(limit, [d.num_edges for d in transformed])
    out = DAG(dags[0].num_nodes)
    for (parent, child), _count in edge_frequency(transformed):
        if out.num_edges >= lim:
            break
        out.add_edge(parent, child)
    return out


def max_frequency(dags: Sequence[DAG], limit: int) -> DAG:
    """Union every edge appearing in at least `limit` of the inputs."""
    _alpha, transformed = _transform_all(dags)
    out = DAG(dags[0].num_nodes)
    for (parent, child), count in edge_frequency(transformed):
        if count >= limit:
            out.add_edge(parent, child)
    return out


def consensus_pct(dags: Sequence[DAG], percent: int) -> DAG:
    """Union every edge appearing in at least `percent`% of inputs.

    The threshold is ``percent * len(dags) // 100``, truncated.

    Parameters
    ----------
    dags : Sequence[DAG]
        Input DAGs over the same node set.
    percent : int
        Percentage threshold (e.g. ``50``).

    Returns
    -------
    DAG
        The consensus DAG.
    """
    lim = percent * len(dags) // 100
    return max_frequency(dags, lim)


def max_parents(dags: Sequence[DAG], limit: int) -> DAG:
    """Union edges in frequency order, giving no node more than `limit` parents."""
    _alpha, transformed = _transform_all(dags)
    out = DAG(dags[0].num_nodes)
    for (parent, child), _count in edge_frequency(transformed):
        if len(out.parents(child)) < limit:
            out.add_edge(parent, child)
    return out


class MaxEdges:
    """`FusionOperator` for :func:`max_edges` with a stored budget."""

    def __init__(self, limit: int | str = "Max") -> None:
        self.limit = limit

    def fuse(self, dags: Sequence[DAG]) -> DAG:
        """Return :func:`max_edges` of `dags` at the stored budget."""
        return max_edges(dags, self.limit)


class MaxFrequency:
    """`FusionOperator` for :func:`max_frequency` with a stored threshold."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def fuse(self, dags: Sequence[DAG]) -> DAG:
        """Return :func:`max_frequency` of `dags` at the stored threshold."""
        return max_frequency(dags, self.limit)


class Consensus:
    """`FusionOperator` for :func:`consensus_pct` with a stored percentage."""

    def __init__(self, percent: int) -> None:
        self.percent = percent

    def fuse(self, dags: Sequence[DAG]) -> DAG:
        """Return :func:`consensus_pct` of `dags` at the stored percentage."""
        return consensus_pct(dags, self.percent)


class MaxParents:
    """`FusionOperator` for :func:`max_parents` with a stored cap."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def fuse(self, dags: Sequence[DAG]) -> DAG:
        """Return :func:`max_parents` of `dags` at the stored cap."""
        return max_parents(dags, self.limit)
