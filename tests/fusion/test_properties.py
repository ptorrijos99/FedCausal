"""Property-based invariants for consensus fusion, robust to tie-breaking.

These hold regardless of how ties are resolved, so they validate the port
independently of the committed reference runs, which pin one specific tie order.
"""

from __future__ import annotations

from pathlib import Path

import hypothesis.strategies as st
from hypothesis import given, settings

from fedcausal.fusion import consensus_union, max_frequency, max_parents
from fedcausal.graph import DAG

NETWORKS = Path(__file__).resolve().parents[1] / "fixtures" / "networks"


@st.composite
def _dag_over(draw: st.DrawFn, p: int) -> DAG:
    """Draw a random DAG on ``p`` nodes using only forward edges (always acyclic)."""
    edges = [(u, v) for u in range(p) for v in range(u + 1, p) if draw(st.booleans())]
    return DAG.from_edges(p, edges)


@given(st.integers(3, 6), st.data())
@settings(max_examples=50)
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
@settings(max_examples=50)
def test_maxfreq1_and_maxparents_p_equal_union(p: int, data: st.DataObject) -> None:
    """Non-binding budgets reduce to the full union."""
    dags = [data.draw(_dag_over(p)) for _ in range(data.draw(st.integers(1, 3)))]
    union = set(consensus_union(dags).edges())
    assert set(max_frequency(dags, 1).edges()) == union
    assert set(max_parents(dags, p).edges()) == union


def test_fuses_real_ges_output_on_asia() -> None:
    """End-to-end: three clients learn asia by GES, then fuse to an acyclic DAG.

    Proves the fusion consumes real GES output: the result is acyclic, covers
    every client's adjacencies, and is not empty.
    """
    xbif = NETWORKS / "asia.xbif"

    from fedcausal.data.partition import contiguous_split
    from fedcausal.io import forward_sample, load_xbif
    from fedcausal.scores import BDeuScore
    from fedcausal.search import GES

    bn = load_xbif(str(xbif))
    ds = forward_sample(bn, 5000, seed=0)
    shards = contiguous_split(ds, 3)
    client_dags = [GES(BDeuScore(s)).fit(s).to_dag() for s in shards]

    fused = consensus_union(client_dags)
    assert len(fused.topological_order()) == fused.num_nodes  # acyclic
    assert fused.num_edges > 0
    fused_skeleton = {frozenset(e) for e in fused.edges()}
    assert all(frozenset(e) in fused_skeleton for d in client_dags for e in d.edges())
