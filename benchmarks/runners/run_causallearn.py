"""Benchmark runner: causal-learn's GES with the discrete BDeu score.

Usage::

    python benchmarks/runners/run_causallearn.py <csv> <xbif> <net>

**Priors cannot be matched to Tetrad's, and this is a property of causal-learn,
not a choice made here.** Two independent obstacles:

1. ``ges()``'s ``local_score_BDeu`` branch constructs its ``LocalScoreClass``
   with a hardcoded ``parameters=None``, discarding whatever the caller passed
   in ``parameters``. There is no public way to set the priors for this score.
2. Even if there were, ``local_score_BDeu``'s structure-prior term is
   ``|PA|·log(sp/vm) + (vm-|PA|)·log(1 - sp/vm)`` -- a Bernoulli edge prior over
   the ``vm = n_features - 1`` possible parents, not Tetrad's additive
   ``structure_prior``. Tetrad's effective value of ``0.0`` would evaluate
   ``log(0)`` here. The two parameterisations are not commensurable at any
   setting.

causal-learn therefore searches with ``sample_prior=1``, ``structure_prior=1``
and per-column arities inferred by ``np.unique``, while ``fedcausal`` and
BayesFL search with Tetrad's ``1.0``/``0.0``. The three optimise slightly
different objectives. That does not invalidate the comparison, because the
orchestrator re-scores all three learned structures with one scorer -- but the
report must state it, so the runner records it in ``notes``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from _common import canon, emit, parse_args  # noqa: E402

from fedcausal.data import DiscreteDataset  # noqa: E402

PRIOR_NOTE = (
    "priors not settable: ges() hardcodes parameters=None for local_score_BDeu, "
    "so sample_prior=1/structure_prior=1 (Bernoulli edge prior, not Tetrad's) "
    "were used instead of 1.0/0.0"
)


def extract_edges(graph: np.ndarray, names: tuple[str, ...]) -> tuple[list, list]:
    """Split causal-learn's endpoint matrix into directed and undirected edges.

    causal-learn's convention (documented on ``ges``): ``graph[j, i] == 1`` and
    ``graph[i, j] == -1`` means ``i -> j``; ``graph[i, j] == graph[j, i] == -1``
    means ``i -- j``.

    Parameters
    ----------
    graph : np.ndarray
        The ``GeneralGraph.graph`` endpoint matrix.
    names : tuple[str, ...]
        Variable names, positionally aligned with `graph`.

    Returns
    -------
    tuple[list, list]
        ``(directed, undirected)`` as sorted lists of name pairs.

    Raises
    ------
    ValueError
        If a node pair carries an endpoint combination GES should never
        produce (a bidirected edge, say) -- silently dropping it would
        understate the learned graph's edge count.
    """
    n = graph.shape[0]
    directed: list[list[str]] = []
    undirected: list[list[str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = graph[i, j], graph[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == -1:
                undirected.append(canon(names[i], names[j]))
            elif a == -1 and b == 1:
                directed.append([names[i], names[j]])
            elif a == 1 and b == -1:
                directed.append([names[j], names[i]])
            else:
                raise ValueError(
                    f"unexpected endpoints between {names[i]} and {names[j]}: "
                    f"graph[i,j]={a}, graph[j,i]={b}"
                )
    return sorted(directed), sorted(undirected)


def main(argv: list[str]) -> int:
    """Run causal-learn's GES once and emit the benchmark record."""
    csv, _xbif, net = parse_args(argv, "run_causallearn.py")

    ds = DiscreteDataset.from_csv(csv)
    x = np.asarray(ds.values, dtype=np.int64)

    # Imported here, not at module scope: the orchestrator must be able to run
    # the other runners on a venv without the optional [compare] extra.
    from causallearn.search.ScoreBased.GES import ges

    start = time.perf_counter()
    record = ges(x, score_func="local_score_BDeu", maxP=None, node_names=list(ds.names))
    seconds = time.perf_counter() - start

    directed, undirected = extract_edges(np.asarray(record["G"].graph), ds.names)

    emit(
        {
            "impl": "causal-learn",
            "net": net,
            "n_vars": ds.n_vars,
            "n_rows": ds.n_rows,
            "seconds_search": seconds,
            "directed_edges": directed,
            "undirected_edges": undirected,
            # On causal-learn's own scale, with its own priors, and treated as
            # a cost by its search. Not comparable with the other two; the
            # orchestrator's re-scored BDeu is the comparable number.
            "self_reported_bdeu": float(record["score"]),
            "notes": PRIOR_NOTE,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
