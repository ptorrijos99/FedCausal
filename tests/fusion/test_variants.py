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


def test_consensus_pct_threshold_with_four_clients() -> None:
    """At 50% of 4 clients (lim=2), a 2/4 edge is kept and a 1/4 edge dropped.

    Java's ``lim = percent * n / 100`` truncates: this uses four clients so
    the 50% threshold lands exactly on ``lim = 2`` rather than a truncated 1.
    """
    d01 = DAG.from_edges(3, [(0, 1)])
    dags = [d01, d01.copy(), DAG.from_edges(3, [(1, 2)]), DAG(3)]
    out = consensus_pct(dags, 50)
    skel = {frozenset(e) for e in out.edges()}
    assert frozenset({0, 1}) in skel  # 2/4 = 50% >= lim 2
    assert frozenset({1, 2}) not in skel  # 1/4 < lim 2


def test_max_edges_int_budget() -> None:
    """An integer budget adds the highest-frequency edges up to the cap."""
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1)])
    out = max_edges([d1, d2], 1)
    assert out.num_edges == 1
    assert (0, 1) in out.edges()  # highest frequency added first


def test_max_parents_caps_the_source_not_the_target() -> None:
    """The cap is on the *source's* parent count -- BayesFL's bug, replicated.

    On a shared chain ``0 -> 1 -> 2`` with cap 1, the highest-frequency edge
    ``0 -> 1`` is added first, giving node ``1`` one parent. When ``1 -> 2`` is
    considered, its *source* ``1`` already has one parent, so the (buggy) cap
    blocks it even though target ``2`` has none. See :func:`max_parents`.
    """
    d1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    d2 = DAG.from_edges(3, [(0, 1), (1, 2)])
    out = max_parents([d1, d2], 1)
    assert set(out.edges()) == {(0, 1)}
