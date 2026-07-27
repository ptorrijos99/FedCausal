"""Scaling bench for the incremental GES engine. Asserts nothing.

Times GES across the net-size ladder and, with ``--profile <net>``, dumps a
cProfile top-cumulative breakdown for one network. Reads BayesFL's sampled CSVs
(``FEDCAUSAL_NETWORKS``, default ``../BayesFL``). Consistent with ``bench_ges.py`` /
``compare_ges.py``: a measurement tool, not a test.

Usage::

    python benchmarks/scale_ges.py                       # full ladder
    python benchmarks/scale_ges.py --nets hepar2 pigs    # a subset
    python benchmarks/scale_ges.py --profile link        # profile one net
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import time
from pathlib import Path

from fedcausal.data import DiscreteDataset
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

NETWORKS = Path(os.environ.get("FEDCAUSAL_NETWORKS", "../networks"))
LADDER = [
    "asia",
    "child",
    "alarm",
    "insurance",
    "hepar2",
    "andes",
    "diabetes",
    "pigs",
    "link",
    "munin",
]


def _load(net: str) -> DiscreteDataset | None:
    """Load ``<net>.0.csv`` from the BayesFL tree, or print+return None if absent."""
    csv = NETWORKS / "res" / "networks" / "BBDD" / net / f"{net}.0.csv"
    if not csv.exists():
        print(f"  {net}: dataset not found ({csv}); skipping")
        return None
    return DiscreteDataset.from_csv(csv)


def _time_ges(ds: DiscreteDataset) -> tuple[float, int]:
    """Return ``(elapsed_seconds, n_edges)`` for one incremental GES fit."""
    t0 = time.perf_counter()
    cpdag = GES(BDeuScore(ds)).fit(ds)
    elapsed = time.perf_counter() - t0
    edges = len(cpdag.directed_edges()) + len(cpdag.undirected_edges())
    return elapsed, edges


def main() -> None:
    """Parse args and either print the ladder timing table or profile one net."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nets", nargs="*", default=None, help="nets to time (default: full ladder)")
    ap.add_argument("--profile", default=None, help="profile one net and print its top costs")
    args = ap.parse_args()

    if args.profile:
        ds = _load(args.profile)
        if ds is None:
            return
        pr = cProfile.Profile()
        pr.enable()
        GES(BDeuScore(ds)).fit(ds)
        pr.disable()
        print(f"cProfile: {args.profile} ({ds.n_vars} vars, {ds.n_rows} rows)")
        pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
        return

    nets = args.nets or LADDER
    print(f"{'net':10s} {'vars':>5s} {'rows':>6s} {'edges':>6s} {'time(s)':>9s}")
    for net in nets:
        ds = _load(net)
        if ds is None:
            continue
        elapsed, edges = _time_ges(ds)
        print(f"{net:10s} {ds.n_vars:5d} {ds.n_rows:6d} {edges:6d} {elapsed:9.2f}")


if __name__ == "__main__":
    main()
