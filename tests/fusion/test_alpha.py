"""Tests for the consensus-ordering heuristic (AlphaOrder port)."""

from __future__ import annotations

from fedcausal.fusion.alpha import alpha_order
from fedcausal.graph import DAG


def test_alpha_order_is_a_permutation() -> None:
    """The order is a permutation of every node index."""
    d = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    order = alpha_order([d])
    assert sorted(order) == [0, 1, 2, 3]


def test_alpha_order_single_chain_is_topological() -> None:
    """A single chain's alpha order keeps every parent before its child."""
    d = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    order = alpha_order([d])
    pos = {n: i for i, n in enumerate(order)}
    for u, v in d.edges():
        assert pos[u] < pos[v]
