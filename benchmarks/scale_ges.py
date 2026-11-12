"""Time GES on networks of growing size and report how close it gets to the truth.

    FEDCAUSAL_NETWORKS=/path/to/networks python benchmarks/scale_ges.py
    python benchmarks/scale_ges.py --nets alarm,andes --seeds 3

Each network is learned once untimed first, so that numba's compilation and
cache loading are not counted. Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import statistics
import time

from _networks import load

from fedcausal import GES
from fedcausal.metrics import shd, smhd

LADDER = (
    "cancer,asia,child,insurance,water,mildew,alarm,hailfinder,hepar2,win95pts,"
    "pathfinder,andes,diabetes,pigs,link,munin"
)


def main() -> None:
    """Run the ladder and print one row per network."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--nets", default=LADDER)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()

    data, _ = load("asia")
    GES().fit(data)  # compile

    print("| Network | Variables | Seconds | Edges | True edges | SHD | SMHD |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for net in args.nets.split(","):
        rows = []
        for seed in range(args.seeds):
            data, truth = load(net, seed)
            start = time.perf_counter()
            dag = GES().fit(data).dag_
            seconds = time.perf_counter() - start
            rows.append((seconds, dag.num_edges, shd(truth, dag), smhd(truth, dag)))
        mean = [statistics.fmean(col) for col in zip(*rows, strict=True)]
        print(
            f"| {net} | {data.n_vars} | {mean[0]:.2f} | {mean[1]:.0f} | {truth.num_edges} "
            f"| {mean[2]:.1f} | {mean[3]:.1f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
