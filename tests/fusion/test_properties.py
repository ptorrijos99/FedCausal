"""Properties of consensus fusion that hold however ties are broken."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from fedcausal.fusion import consensus_union, max_frequency, max_parents
from fedcausal.graph import DAG


@st.composite
def _dag_over(draw: st.DrawFn, p: int) -> DAG:
    """Draw a random DAG on ``p`` nodes using only forward edges (always acyclic)."""
    edges = [(u, v) for u in range(p) for v in range(u + 1, p) if draw(st.booleans())]
    return DAG.from_edges(p, edges)


@given(st.integers(3, 6), st.data())
@settings(max_examples=50, deadline=None)
def test_union_acyclic_and_imap(p: int, data: st.DataObject) -> None:
    """The consensus union is acyclic and covers every input skeleton edge."""
    dags = [data.draw(_dag_over(p)) for _ in range(data.draw(st.integers(1, 3)))]
    out = consensus_union(dags)
    assert len(out.topological_order()) == p
    union_skel = {frozenset(e) for e in out.edges()}
    for d in dags:
        for e in d.edges():
            assert frozenset(e) in union_skel


@given(st.integers(3, 6), st.data())
@settings(max_examples=50, deadline=None)
def test_maxfreq1_and_maxparents_p_equal_union(p: int, data: st.DataObject) -> None:
    """Non-binding budgets reduce to the full union."""
    dags = [data.draw(_dag_over(p)) for _ in range(data.draw(st.integers(1, 3)))]
    union = set(consensus_union(dags).edges())
    assert set(max_frequency(dags, 1).edges()) == union
    assert set(max_parents(dags, p).edges()) == union


@given(st.integers(3, 6), st.integers(0, 3), st.data())
@settings(max_examples=50, deadline=None)
def test_max_parents_respects_the_cap(p: int, limit: int, data: st.DataObject) -> None:
    """No node gets more than `limit` parents, and nothing outside the union appears."""
    dags = [data.draw(_dag_over(p)) for _ in range(data.draw(st.integers(1, 3)))]
    out = max_parents(dags, limit)
    assert all(len(out.parents(v)) <= limit for v in range(p))
    assert set(out.edges()) <= set(consensus_union(dags).edges())


def test_fuses_real_ges_output_on_asia() -> None:
    """Fusing GES output from three sites gives an acyclic union of their adjacencies."""
    from fedcausal import GES
    from fedcausal.data import contiguous_split
    from fedcausal.datasets import load_asia

    client_dags = [GES().fit(s).dag_ for s in contiguous_split(load_asia().data, 3)]

    fused = consensus_union(client_dags)
    assert len(fused.topological_order()) == fused.num_nodes
    assert fused.num_edges > 0
    fused_skeleton = {frozenset(e) for e in fused.edges()}
    assert all(frozenset(e) in fused_skeleton for d in client_dags for e in d.edges())
