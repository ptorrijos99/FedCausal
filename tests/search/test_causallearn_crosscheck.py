"""fedcausal's GES must not end on a materially worse optimum than causal-learn's.

Two independent implementations of GES can stop at different local optima, so
the graphs are not compared. Both results are scored with fedcausal's own
BDeu, and fedcausal's may be at most 0.1% worse. Each network also has one
strong true edge that both must find, which catches a result that collapsed
to an empty or complete graph.

Needs the optional ``compare`` extra; skipped without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fedcausal.graph import PDAG
from fedcausal.io import forward_sample, load_xbif
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

cl_ges = pytest.importorskip("causallearn.search.ScoreBased.GES")

NETWORKS = Path(__file__).resolve().parents[1] / "fixtures" / "networks"

# An edge whose child depends strongly on its parent in the network's own CPTs.
STRONG_ADJACENCY = {
    "asia": ("smoke", "lung"),
    "cancer": ("Cancer", "Xray"),
    "child": ("LVH", "LVHreport"),
}


def _cl_graph_to_pdag(graph: Any, num_nodes: int) -> PDAG:
    """Translate causal-learn's endpoint matrix (-1 tail, 1 arrowhead) into a `PDAG`.

    Raises `ValueError` on any other endpoint pair, which a CPDAG never has.
    """
    amat = graph.graph
    pdag = PDAG(num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            a, b = int(amat[i, j]), int(amat[j, i])
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                pdag.add_directed(i, j)
            elif a == 1 and b == -1:
                pdag.add_directed(j, i)
            elif a == -1 and b == -1:
                pdag.add_undirected(i, j)
            else:
                raise ValueError(f"endpoints ({a}, {b}) between {i} and {j} are not a CPDAG edge")
    return pdag


@pytest.mark.parametrize("net", list(STRONG_ADJACENCY))
def test_crosscheck_causallearn(net: str) -> None:
    bn = load_xbif(NETWORKS / f"{net}.xbif")
    ds = forward_sample(bn, 5000, seed=0)
    score = BDeuScore(ds)

    ours = GES(score).fit(ds)
    theirs = _cl_graph_to_pdag(
        cl_ges.ges(
            ds.values.astype(np.int64), score_func="local_score_BDeu", node_names=list(ds.names)
        )["G"],
        ds.n_vars,
    )

    a, b = (ds.names.index(name) for name in STRONG_ADJACENCY[net])
    assert b in ours.adjacent(a)
    assert b in theirs.adjacent(a)

    our_bdeu = score.score_dag(ours.to_dag())
    their_bdeu = score.score_dag(theirs.to_dag())
    assert our_bdeu >= their_bdeu - abs(their_bdeu) * 1e-3
