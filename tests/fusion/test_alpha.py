"""Tests for the consensus-ordering heuristic (AlphaOrder port)."""

from __future__ import annotations

from fedcausal.fusion.alpha import _compute_next, _peel, alpha_order
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


def test_alpha_order_agreeing_dags() -> None:
    """Two clients agreeing on ``0 -> 1 -> 2`` yield an order respecting it."""
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1), (1, 2)])
    order = alpha_order([d1, d2])
    pos = {n: i for i, n in enumerate(order)}
    assert pos[0] < pos[1] < pos[2]


def test_peel_reverses_and_moralizes_then_isolates() -> None:
    """Peeling the source of a fork reverses its arcs and adds the moral edge.

    ``0 -> 1, 0 -> 2``: peeling ``0`` reverses both out-arcs (making ``0`` a
    sink of ``1`` and ``2``), moralizes its former children into ``1 -> 2``,
    then strips every arc incident to ``0``, leaving ``0`` isolated and only
    the moral edge behind. Hand-derived; see the module docstring.
    """
    g = DAG.from_edges(3, [(0, 1), (0, 2)])
    _peel(g, 0)
    assert g.children(0) == frozenset()
    assert g.parents(0) == frozenset()
    assert set(g.edges()) == {(1, 2)}


def test_compute_next_prefers_cheapest_sink() -> None:
    """On a single chain, the unique sink is the cheapest node to peel."""
    d = DAG.from_edges(3, [(0, 1), (1, 2)])
    # Node 2 is the sink (no children): peeling it costs nothing.
    assert _compute_next([d], [0, 1, 2]) == 2
