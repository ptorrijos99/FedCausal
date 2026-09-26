"""Benchmark runner: one GES search under one FES-bounds configuration.

Usage::

    python benchmarks/runners/run_bounds_cell.py <net> <seed> <config>

where ``<config>`` names a row of :data:`CONFIGS` -- a ``max_t_size`` /
``empty_t_gate`` / ``h_pool`` combination. Prints one JSON object to stdout.

Runs in its own process so the orchestrator can put a wall-clock budget on
each search: the unbounded configuration is exactly the one known to blow up
(``pathfinder`` sample 3 exceeded two hours), and a bounds benchmark that
cannot survive its own worst case would be useless.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import emit  # noqa: E402
from compare_ges import load_net  # noqa: E402

from fedcausal.metrics import shd, smhd  # noqa: E402
from fedcausal.scores import BDeuScore  # noqa: E402
from fedcausal.search import GES  # noqa: E402

# The configurations compared. Each sets max_t_size explicitly, because an empty
# dict would mean the default, cap2.
CONFIGS: dict[str, dict[str, object]] = {
    "exhaustive": {"max_t_size": None},
    "cap1": {"max_t_size": 1},
    "cap2": {"max_t_size": 2},
    "cap3": {"max_t_size": 3},
    "gate": {"max_t_size": None, "empty_t_gate": True},
    "java": {"max_t_size": 2, "empty_t_gate": True},
}


def main(argv: list[str]) -> int:
    """Run one (net, seed, config) search and emit its record."""
    if len(argv) != 4:
        print(
            f"Usage: run_bounds_cell.py <net> <seed> <{'|'.join(CONFIGS)}>",
            file=sys.stderr,
        )
        return 2
    net, seed, config = argv[1], int(argv[2]), argv[3]
    if config not in CONFIGS:
        print(f"unknown config {config!r}; expected one of {', '.join(CONFIGS)}", file=sys.stderr)
        return 2

    ds, truth = load_net(net, seed)[:2]
    score = BDeuScore(ds)

    start = time.perf_counter()
    cpdag = GES(score, **CONFIGS[config]).fit(ds)  # type: ignore[arg-type]
    seconds = time.perf_counter() - start
    dag = cpdag.to_dag()

    emit(
        {
            "net": net,
            "config": config,
            "seed": seed,
            "seconds": seconds,
            "bdeu": score.score_dag(dag),
            "n_edges": dag.num_edges,
            "shd": shd(truth, dag) if truth is not None else None,
            "smhd": smhd(truth, dag) if truth is not None else None,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
