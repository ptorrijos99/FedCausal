"""Tests for the minimal-I-map transform into a target order (BetaToAlpha port)."""

from __future__ import annotations

from fedcausal.fusion.beta import beta_to_alpha
from fedcausal.graph import DAG


def _consistent(dag: DAG, order: list[int]) -> bool:
    pos = {n: i for i, n in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in dag.edges())


def test_identity_when_already_consistent() -> None:
    """A DAG already consistent with the order is returned unchanged."""
    g = DAG.from_edges(3, [(0, 1), (1, 2)])
    out = beta_to_alpha(g, [0, 1, 2])
    assert set(out.edges()) == {(0, 1), (1, 2)}
    assert _consistent(out, [0, 1, 2])


def test_reversed_order_is_still_consistent_and_imap() -> None:
    """Reversing a chain keeps its skeleton and respects the new order."""
    g = DAG.from_edges(3, [(0, 1), (1, 2)])
    out = beta_to_alpha(g, [2, 1, 0])
    assert _consistent(out, [2, 1, 0])
    skel = {frozenset(e) for e in out.edges()}
    assert skel == {frozenset({0, 1}), frozenset({1, 2})}


def test_vstructure_moralizes_under_reordering() -> None:
    """Forcing a collider's child earliest introduces the moral edge."""
    g = DAG.from_edges(3, [(0, 2), (1, 2)])
    out = beta_to_alpha(g, [2, 0, 1])
    assert _consistent(out, [2, 0, 1])
    assert 1 in out.adjacent(0)  # moral edge 0-1 introduced


def test_input_is_not_mutated() -> None:
    """The transform returns a new DAG and leaves the input untouched."""
    g = DAG.from_edges(3, [(0, 1), (1, 2)])
    before = set(g.edges())
    beta_to_alpha(g, [2, 1, 0])
    assert set(g.edges()) == before


def test_the_transform_adds_only_the_edges_the_order_needs() -> None:
    """For the order 1, 0, 3, 2, node 3 needs parents {0, 1} and node 2 needs {0, 1, 3}."""
    dag = DAG.from_edges(4, [(0, 3), (2, 1), (2, 3)])
    assert beta_to_alpha(dag, [1, 0, 3, 2]).num_edges == 5
