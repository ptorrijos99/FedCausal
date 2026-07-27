"""Cross-check ``fedcausal``'s GES against ``causal-learn``'s independent GES.

This is a *sanity oracle*, not an exactness check. ``causal-learn``'s GES
(``causallearn.search.ScoreBased.GES.ges``) and ``fedcausal``'s ``GES`` are
two independent implementations of Chickering's (2002) algorithm; both only
guarantee convergence to *some* local optimum of the score they drive search
with, and different tie-breaking/candidate-ordering choices can and do lead
them to different (but often equally valid, same-equivalence-class-adjacent)
CPDAGs. Asserting structural equality between the two outputs would therefore
be the wrong test, and is deliberately **not** attempted here.

Instead, both learned CPDAGs are extended to a consistent DAG
(:meth:`fedcausal.graph.PDAG.to_dag`) and scored with **our own**
``BDeuScore`` -- the same priors on both sides, so the comparison is
apples-to-apples -- and the single binding assertion is one-sided: fedcausal
must not be *materially* worse than causal-learn under our scorer,
``our_bdeu >= cl_bdeu - TOL`` with ``TOL = abs(cl_bdeu) * 1e-3``. It is never
asserted that fedcausal is better, nor that the two graphs agree.

**Verified empirically** (not merely assumed from the ``ges`` docstring):
``causal-learn``'s ``GeneralGraph.graph`` is an endpoint-encoded adjacency
matrix using Tetrad-style codes -- ``-1`` tail, ``1`` arrowhead, ``0`` no
edge. Checked on tiny synthetic examples before writing this module:

- A pure Markov chain ``A -> B -> C`` (no v-structure) recovers as the fully
  undirected CPDAG ``A - B - C``: ``graph[i, j] == graph[j, i] == -1`` for
  each adjacent pair, confirming the undirected encoding.
- A collider ``A -> C <- B`` (``C`` a noisy OR of two independent parents,
  chosen instead of an XOR collider because a noisy XOR's parents are each
  *marginally* independent of the child, which stops greedy GES from ever
  proposing the first edge) recovers with ``A --> C`` and ``B --> C``:
  ``graph[i, j] == -1 and graph[j, i] == 1`` for the edge printed by
  ``causal-learn`` itself as ``i --> j``, confirming ``-1`` is the tail end
  and ``1`` is the arrowhead end, matching the module's own docstring
  (``Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1`` means
  ``i --> j``).

No bidirected (``1``/``1``) or circle-endpoint (PAG-style ``2``) codes were
observed from ``ges``'s CPDAG output, as expected: GES only ever returns a
CPDAG (canonicalized via ``pdag2dag``/``dag2cpdag`` internally), never a PAG.
:func:`_cl_graph_to_pdag` still checks for and rejects any such edge
explicitly (skipping that network with a clear message) rather than silently
misinterpreting or dropping it, since ``fedcausal``'s ``PDAG`` has no
representation for a bidirected edge.

Tested against ``causal-learn`` 0.1.4.8, via
``causallearn.search.ScoreBased.GES.ges(data, score_func="local_score_BDeu",
node_names=...)``, which returns a dict whose ``'G'`` key holds the learned
``GeneralGraph``. ``local_score_BDeu``'s own priors (its defaults:
``sample_prior=1``, ``structure_prior=1``, per-variable cardinalities
inferred from the data actually observed) are used unmodified here -- they
only drive causal-learn's *own* search and are never compared to our score
directly, so a difference in priors from ``fedcausal.scores.BDeuScore``'s
defaults does not affect the (one-sided, same-scorer) assertion this module
makes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fedcausal.graph import PDAG
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.io import forward_sample, load_xbif
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

cl_ges = pytest.importorskip("causallearn.search.ScoreBased.GES")

NETWORKS = Path(__file__).resolve().parents[1] / "fixtures" / "networks"
NETS = ["asia", "cancer", "child"]

# One genuinely-strong, reliably-recoverable ground-truth edge per network, by
# node NAME (resolved against the dataset's `names` below, never by index).
# This is an independent structural guardrail alongside the BDeu comparison:
# a regression that collapses to a degenerate structure (e.g. everything or
# nothing adjacent) could in principle still slip past a BDeu-only check, but
# would fail a direct adjacency assertion on a known-strong dependency.
#
# Each edge was chosen by inspecting the ground-truth network's own CPTs (see
# `<net>.xbif` under `tests/fixtures/networks/`), picking the child variable
# whose conditional probability swings most strongly with its parent, then
# verified empirically (this exact seed/n_rows) that BOTH engines' result
# CPDAGs keep the pair adjacent:
#
# - asia: smoke -> lung. `P(lung=yes | smoke=yes)=0.1` vs
#   `P(lung=yes | smoke=no)=0.01`, a 10x odds swing on a common (~50/50)
#   cause -- the network's textbook strong causal link.
# - cancer: Cancer -> Xray. `P(Xray=positive | Cancer=True)=0.9` vs
#   `P(Xray=positive | Cancer=False)=0.2`, the largest likelihood-ratio edge
#   in the network (diagnostic-test-strength dependency).
# - child: LVH -> LVHreport. `P(LVHreport=yes | LVH=yes)=0.9` vs
#   `P(LVHreport=yes | LVH=no)=0.05`, a near-deterministic "report" node --
#   directly analogous to the synthetic `A -> B` fixture in
#   test_ges.py/test_bes.py/test_fes.py (a variable that is essentially a
#   noisy copy of its parent).
#
# `asia` also has a known weak edge, `asia -> tub` (the `tub` node is
# `yes` in only ~1% of forward-sampled rows at n=5000), which neither engine
# recovers as adjacent at this sample size -- it is intentionally not used
# here; `smoke -> lung` above is used instead as asia's representative edge.
STRONG_ADJACENCY = {
    "asia": ("smoke", "lung"),
    "cancer": ("Cancer", "Xray"),
    "child": ("LVH", "LVHreport"),
}


def _cl_graph_to_pdag(graph: Any, num_nodes: int) -> PDAG:
    """Convert a causal-learn `GeneralGraph`'s endpoint matrix to a `PDAG`.

    Parameters
    ----------
    graph : causallearn.graph.GeneralGraph.GeneralGraph
        The learned graph, as returned in ``ges(...)['G']``.
    num_nodes : int
        Number of nodes, matching ``graph``'s node count and node order.

    Returns
    -------
    PDAG
        The same graph, translated edge-for-edge: ``graph[i, j] == -1 and
        graph[j, i] == 1`` becomes the directed edge ``i -> j``;
        ``graph[i, j] == graph[j, i] == -1`` becomes the undirected edge
        ``{i, j}`` (see the module docstring for how this encoding was
        verified empirically).

    Raises
    ------
    ValueError
        If some node pair's endpoint codes are anything other than a plain
        directed or undirected edge (e.g. a bidirected or circle endpoint) --
        `fedcausal`'s `PDAG` cannot represent such an edge, so it is reported
        rather than silently dropped or misinterpreted.
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
                raise ValueError(
                    f"causal-learn produced endpoint codes graph[{i},{j}]={a}, "
                    f"graph[{j},{i}]={b} between nodes {i} and {j} -- not a plain directed "
                    "or undirected edge (e.g. bidirected or circle endpoints), which "
                    "fedcausal's PDAG has no representation for"
                )
    return pdag


def _run_crosscheck(net: str, n_rows: int) -> None:
    """Run both GES implementations on `net` and cross-check under our BDeuScore.

    See the module docstring for the one-sided assertion this makes and why.
    """
    xbif = NETWORKS / f"{net}.xbif"
    bn = load_xbif(str(xbif))
    ds = forward_sample(bn, n_rows, seed=0)
    assert ds.names == tuple(bn.names)

    score = BDeuScore(ds)

    t0 = time.perf_counter()
    our_pdag = GES(score).fit(ds)
    our_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = cl_ges.ges(
        ds.values.astype(np.int64),
        score_func="local_score_BDeu",
        node_names=list(ds.names),
    )
    cl_seconds = time.perf_counter() - t0

    try:
        cl_pdag = _cl_graph_to_pdag(result["G"], ds.n_vars)
    except ValueError as exc:
        pytest.skip(f"{net}: causal-learn output not convertible to a PDAG: {exc}")
        return

    name_a, name_b = STRONG_ADJACENCY[net]
    idx_a, idx_b = ds.names.index(name_a), ds.names.index(name_b)
    assert idx_b in our_pdag.adjacent(idx_a), (
        f"{net}: fedcausal's result CPDAG does not have {name_a}-{name_b} adjacent, a "
        "known-strong ground-truth dependency -- possible degenerate-structure regression"
    )
    assert idx_b in cl_pdag.adjacent(idx_a), (
        f"{net}: causal-learn's result CPDAG does not have {name_a}-{name_b} adjacent, a "
        "known-strong ground-truth dependency -- unexpected for the oracle itself, worth "
        "investigating the causal-learn call/version rather than fedcausal"
    )

    our_dag = our_pdag.to_dag()
    # cl_pdag.to_dag() is deliberately left unwrapped: an inconsistent CPDAG from
    # causal-learn here would mean our own lossless graph translation produced
    # something PDAG.to_dag() cannot extend, which is a genuine finding worth a
    # loud failure, not routine noise to skip past (unlike the endpoint-code
    # rejection in _cl_graph_to_pdag above, which is a documented, expected
    # possibility for a PAG-capable library and not itself a bug).
    cl_dag = cl_pdag.to_dag()

    our_bdeu = score.score_dag(our_dag)
    cl_bdeu = score.score_dag(cl_dag)
    tol = abs(cl_bdeu) * 1e-3

    true_dag = dag_to_cpdag(bn.dag).to_dag()
    print(
        f"[{net}] our_bdeu={our_bdeu:.4f} cl_bdeu={cl_bdeu:.4f} "
        f"our_shd_vs_true={shd(true_dag, our_dag)} cl_shd_vs_true={shd(true_dag, cl_dag)} "
        f"our_smhd_vs_true={smhd(bn.dag, our_dag)} cl_smhd_vs_true={smhd(bn.dag, cl_dag)} "
        f"t_ours={our_seconds:.4f}s t_causallearn={cl_seconds:.4f}s"
    )

    assert our_bdeu >= cl_bdeu - tol, (
        f"{net}: fedcausal's GES result (BDeu={our_bdeu:.4f}) is materially worse than "
        f"causal-learn's (BDeu={cl_bdeu:.4f}) under our own BDeuScore (tol={tol:.4f}) -- "
        "this is a possible correctness or search-quality regression, not just a "
        "different-local-optimum artifact"
    )


@pytest.mark.parametrize("net", NETS)
def test_crosscheck_causallearn(net: str) -> None:
    _run_crosscheck(net, 5000)
