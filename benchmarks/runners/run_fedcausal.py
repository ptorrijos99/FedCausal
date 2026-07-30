"""Benchmark runner: ``fedcausal``'s own GES.

Runs ``GES(BDeuScore(ds)).fit(ds)`` with ``h_pool="nayx"`` -- Chickering's
correct Delete operator, which is ``fedcausal``'s default and the configuration
any published result should use. (``h_pool="bayesfl"`` exists to replicate
BayesFL's incomplete BES; the two are compared in ``bench_ges.py``, not here.)

Usage::

    python benchmarks/runners/run_fedcausal.py <csv> <xbif> <net> [--structure-prior=X]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import canon, emit, parse_args  # noqa: E402

from fedcausal.data import DiscreteDataset  # noqa: E402
from fedcausal.scores import BDeuScore  # noqa: E402
from fedcausal.search import GES  # noqa: E402


def main(argv: list[str]) -> int:
    """Run GES once on the given CSV and emit the benchmark record."""
    # A trailing --structure-prior overrides the library default, which the
    # pipeline test uses to reproduce a reference produced without one.
    prior: float | None = None
    if len(argv) > 4 and argv[4].startswith("--structure-prior="):
        prior = float(argv[4].split("=", 1)[1])
        argv = argv[:4]
    csv, _xbif, net = parse_args(argv, "run_fedcausal.py")

    ds = DiscreteDataset.from_csv(csv)
    score = BDeuScore(ds) if prior is None else BDeuScore(ds, structure_prior=prior)

    start = time.perf_counter()
    cpdag = GES(score, h_pool="nayx").fit(ds)
    seconds = time.perf_counter() - start

    names = ds.names
    directed = sorted([names[u], names[v]] for u, v in cpdag.directed_edges())
    undirected = sorted(canon(names[u], names[v]) for u, v in cpdag.undirected_edges())

    emit(
        {
            "impl": "fedcausal",
            "net": net,
            "n_vars": ds.n_vars,
            "n_rows": ds.n_rows,
            "seconds_search": seconds,
            "directed_edges": directed,
            "undirected_edges": undirected,
            # The score of a DAG extension of the returned CPDAG: every
            # extension shares it, so this is well defined.
            "self_reported_bdeu": score.score_dag(cpdag.to_dag()),
            "notes": "",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
