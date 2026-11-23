"""Fusion operators: properties that hold however ties break, and the cases that pin semantics."""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from fedcausal.fusion import (
    alpha_order,
    beta_to_alpha,
    consensus,
    consensus_union,
    intersection,
    max_edges,
    max_frequency,
    max_parents,
)
from fedcausal.graph import DAG


@st.composite
def _dags(draw: st.DrawFn) -> list[DAG]:
    """One to four DAGs over the same nodes, each in its own topological order."""
    p = draw(st.integers(2, 7))
    out = []
    for _ in range(draw(st.integers(1, 4))):
        order = draw(st.permutations(range(p)))
        pairs = [(order[i], order[j]) for i in range(p) for j in range(i + 1, p)]
        out.append(DAG.from_edges(p, [e for e in pairs if draw(st.booleans())]))
    return out


def _skeleton(dag: DAG) -> set[frozenset[int]]:
    return {frozenset(e) for e in dag.edges()}


@given(_dags(), st.integers(0, 3))
@settings(max_examples=100, deadline=None)
def test_fusion_properties(dags: list[DAG], cap: int) -> None:
    p = dags[0].num_nodes
    before = [d.edges() for d in dags]
    order = alpha_order(dags)
    assert sorted(order) == list(range(p))
    position = {v: i for i, v in enumerate(order)}
    for dag in dags:
        moved = beta_to_alpha(dag, order)
        assert all(position[u] < position[v] for u, v in moved.edges())
        assert _skeleton(dag) <= _skeleton(moved)
    union = consensus_union(dags)
    assert len(union.topological_order()) == p
    assert all(_skeleton(d) <= _skeleton(union) for d in dags)
    # Budgets that do not bind give the union; those that do give part of it.
    assert set(max_frequency(dags, 1).edges()) == set(union.edges())
    assert set(max_parents(dags, p).edges()) == set(union.edges())
    capped = max_parents(dags, cap)
    assert set(capped.edges()) <= set(union.edges())
    assert all(len(capped.parents(v)) <= cap for v in range(p))
    assert set(consensus(dags, 100).edges()) == set(max_frequency(dags, len(dags)).edges())
    shared = intersection(dags)
    assert set(shared.edges()) <= set(dags[0].edges())
    assert all(_skeleton(shared) <= _skeleton(d) for d in dags)
    assert [d.edges() for d in dags] == before  # no input is modified


def test_the_transform_adds_only_the_edges_the_order_needs() -> None:
    # Already consistent: unchanged. A collider's child first: its parents get married.
    chain = DAG.from_edges(3, [(0, 1), (1, 2)])
    assert beta_to_alpha(chain, [0, 1, 2]) == chain
    assert 1 in beta_to_alpha(DAG.from_edges(3, [(0, 2), (1, 2)]), [2, 0, 1]).adjacent(0)
    # For the order 1, 0, 3, 2, node 3 needs parents {0, 1} and node 2 needs {0, 1, 3}.
    assert beta_to_alpha(DAG.from_edges(4, [(0, 3), (2, 1), (2, 3)]), [1, 0, 3, 2]).num_edges == 5


def test_votes_and_budgets() -> None:
    d01, d12 = DAG.from_edges(3, [(0, 1)]), DAG.from_edges(3, [(1, 2)])
    # 50% of four inputs is two votes, and of three inputs also two (rounded up).
    for dags in ([d01, d01, d12, DAG(3)], [d01, d01, d12]):
        assert consensus(dags, 50).edges() == [(0, 1)]
    assert max_edges([d01, d01, d12], 1).edges() == [(0, 1)]  # the most agreed first
    dense, sparse = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)]), DAG.from_edges(4, [(0, 1)])
    assert [max_edges([dense, sparse], b).num_edges for b in ("min", "mean", "max")] == [1, 2, 3]
    with pytest.raises(ValueError, match="unknown edge limit"):
        max_edges([dense, sparse], "Max")  # type: ignore[arg-type]
    collider = DAG.from_edges(3, [(0, 2), (1, 2)])
    assert max_parents([collider, collider], 1).num_edges == 1
    with pytest.raises(ValueError, match="at least one"):
        consensus_union([])


def test_intersection_matches_pairs_and_keeps_the_first_orientation() -> None:
    d1 = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert intersection([d1, DAG.from_edges(4, [(0, 1), (3, 2)])]).edges() == [(0, 1), (2, 3)]


def test_fuses_real_ges_output() -> None:
    from fedcausal import GES
    from fedcausal.data import contiguous_split
    from fedcausal.datasets import load_asia

    dags = [GES().fit(site).dag_ for site in contiguous_split(load_asia().data, 3)]
    fused = consensus_union(dags)
    assert len(fused.topological_order()) == fused.num_nodes
    assert all(_skeleton(d) <= _skeleton(fused) for d in dags)
