"""Benchmark runner: one GES search at one structure-prior setting.

Usage::

    python benchmarks/runners/run_sweep_cell.py <net> <seed> <prior> <vm_mode>

Exists so the structure-prior sweep can put a wall-clock budget on each
individual search. The first version of the sweep ran every search in-process
and hung for over two hours on ``pathfinder`` seed 3 -- the same network and
the same prior whose seed 0 finishes in 22 seconds. FES's ``T`` enumeration is
exponential in neighbourhood width, and width varies between samples of one
network, so any harness that runs these searches without a per-search timeout
can be stopped indefinitely by a single unlucky dataset.

Prints one JSON object to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import emit  # noqa: E402
from compare_ges import load_net  # noqa: E402
from sweep_structure_prior import VarCountPrior  # noqa: E402

from fedcausal.metrics import shd, smhd  # noqa: E402
from fedcausal.scores import BDeuScore  # noqa: E402
from fedcausal.search import GES  # noqa: E402


def main(argv: list[str]) -> int:
    """Run one (net, seed, prior, vm_mode) search and emit its record."""
    if len(argv) != 5:
        print("Usage: run_sweep_cell.py <net> <seed> <prior> <vm_mode>", file=sys.stderr)
        return 2
    net, seed, prior, vm_mode = argv[1], int(argv[2]), float(argv[3]), argv[4]

    ds, truth = load_net(net, seed)[:2]
    score_cls = BDeuScore if vm_mode == "rows" else VarCountPrior
    # Quality is always measured with the prior switched off: scores produced
    # under different priors sit on different scales and cannot be ranked
    # against one another.
    reference = BDeuScore(ds)

    start = time.perf_counter()
    cpdag = GES(score_cls(ds, structure_prior=prior)).fit(ds)
    seconds = time.perf_counter() - start
    dag = cpdag.to_dag()

    emit(
        {
            "net": net,
            "vm_mode": vm_mode,
            "structure_prior": prior,
            "seed": seed,
            "seconds": seconds,
            "bdeu_sp0": reference.score_dag(dag),
            "n_edges": dag.num_edges,
            "shd": shd(truth, dag) if truth is not None else None,
            "smhd": smhd(truth, dag) if truth is not None else None,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
