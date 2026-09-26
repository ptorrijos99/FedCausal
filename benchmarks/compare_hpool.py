"""Compare the two backward-phase H pools: Chickering's and the Java implementation's.

``h_pool="nayx"`` draws the Delete subset ``H`` from ``NaYX``, as Chickering
(2002) defines it; ``h_pool="bayesfl"`` draws it from ``T0``, as BayesFL does,
which in practice only ever deletes with ``H = {}``. Prints BDeu, structural
error and runtime for both. Asserts nothing.

Usage::

    python benchmarks/compare_hpool.py
    python benchmarks/compare_hpool.py --net mildew --net hailfinder
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_ges import load_net  # noqa: E402

from fedcausal.metrics import shd, smhd  # noqa: E402
from fedcausal.scores import BDeuScore  # noqa: E402
from fedcausal.search import GES  # noqa: E402

# mildew and hailfinder form large common neighbourhoods, where a non-empty H
# should matter most.
NETS: tuple[str, ...] = (
    "asia",
    "cancer",
    "child",
    "insurance",
    "water",
    "mildew",
    "alarm",
    "hailfinder",
    "hepar2",
)


def main(argv: list[str] | None = None) -> int:
    """Print one row per network comparing the two pools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", action="append", dest="nets", help="restrict to this net")
    args = parser.parse_args(argv)

    print(f"{'net':<12}{'pool':>9}{'bdeu':>16}{'edges':>7}{'shd':>6}{'smhd':>6}{'secs':>9}")
    for net in args.nets or NETS:
        try:
            ds, truth, _, _ = load_net(net)
        except FileNotFoundError as exc:
            print(f"{net:<12}  skipped: {exc}")
            continue
        for pool in ("nayx", "bayesfl"):
            score = BDeuScore(ds)
            start = time.perf_counter()
            dag = GES(score, h_pool=pool).fit(ds).to_dag()  # type: ignore[arg-type]
            seconds = time.perf_counter() - start
            errors = ("-", "-") if truth is None else (shd(truth, dag), smhd(truth, dag))
            print(
                f"{net:<12}{pool:>9}{score.score_dag(dag):>16.2f}{dag.num_edges:>7}"
                f"{errors[0]:>6}{errors[1]:>6}{seconds:>9.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
