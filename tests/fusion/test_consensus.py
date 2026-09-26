"""Tests for the full consensus union (ConsensusUnion.applyUnion port)."""

from __future__ import annotations

import pytest

from fedcausal.fusion import consensus_union
from fedcausal.graph import DAG


def _is_acyclic(dag: DAG) -> bool:
    return len(dag.topological_order()) == dag.num_nodes


def test_union_of_identical_dags_is_that_dag() -> None:
    """Fusing a DAG with a copy of itself recovers its skeleton."""
    d = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    out = consensus_union([d, d.copy()])
    assert {frozenset(e) for e in out.edges()} == {
        frozenset({0, 1}),
        frozenset({1, 2}),
        frozenset({2, 3}),
    }
    assert _is_acyclic(out)


def test_union_contains_every_input_skeleton_edge() -> None:
    """The union's skeleton covers every input adjacency."""
    d1 = DAG.from_edges(3, [(0, 1)])
    d2 = DAG.from_edges(3, [(1, 2)])
    out = consensus_union([d1, d2])
    skel = {frozenset(e) for e in out.edges()}
    assert frozenset({0, 1}) in skel
    assert frozenset({1, 2}) in skel
    assert _is_acyclic(out)


def test_union_is_acyclic_on_conflicting_orientations() -> None:
    """Opposed orientations still fuse to an acyclic graph keeping the edge."""
    d1 = DAG.from_edges(2, [(0, 1)])
    d2 = DAG.from_edges(2, [(1, 0)])
    out = consensus_union([d1, d2])
    assert _is_acyclic(out)
    assert 1 in out.adjacent(0)


def test_empty_input_raises() -> None:
    """Fusing no DAGs is a caller error."""
    with pytest.raises(ValueError, match="at least one"):
        consensus_union([])
