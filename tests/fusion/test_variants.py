"""Tests for the limited fusion variants (ConsensusUnion switch-branch ports)."""

from __future__ import annotations

from fedcausal.fusion import (
    consensus_pct,
    max_edges,
    max_frequency,
    max_parents,
)
from fedcausal.graph import DAG


def test_max_frequency_keeps_only_unanimous() -> None:
    """``limit=2`` keeps only edges present in both clients."""
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1)])
    out = max_frequency([d1, d2], 2)
    skel = {frozenset(e) for e in out.edges()}
    assert frozenset({0, 1}) in skel
    assert frozenset({1, 2}) not in skel


def test_consensus_pct_needs_at_least_the_given_share_of_votes() -> None:
    """50% of four inputs is two votes, and of three inputs also two."""
    d01 = DAG.from_edges(3, [(0, 1)])
    d12 = DAG.from_edges(3, [(1, 2)])
    four = {frozenset(e) for e in consensus_pct([d01, d01.copy(), d12, DAG(3)], 50).edges()}
    three = {frozenset(e) for e in consensus_pct([d01, d01.copy(), d12], 50).edges()}
    assert four == three == {frozenset({0, 1})}


def test_max_edges_int_budget() -> None:
    """An integer budget adds the highest-frequency edges up to the cap."""
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1)])
    out = max_edges([d1, d2], 1)
    assert out.num_edges == 1
    assert (0, 1) in out.edges()  # highest frequency added first


def test_max_parents_caps_each_nodes_in_degree() -> None:
    """With cap 1, a collider keeps one of its two parents and a chain keeps both edges."""
    collider = DAG.from_edges(3, [(0, 2), (1, 2)])
    assert max_parents([collider, collider.copy()], 1).num_edges == 1

    chain = DAG.from_edges(3, [(0, 1), (1, 2)])
    assert set(max_parents([chain, chain.copy()], 1).edges()) == {(0, 1), (1, 2)}
