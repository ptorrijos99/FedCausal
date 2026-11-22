"""How much structure FedGES loses as the same data is split across more sites.

    FEDCAUSAL_NETWORKS=/path/to/networks python benchmarks/fedges_sites.py
    python benchmarks/fedges_sites.py --nets alarm --sites 5,10 --seeds 3

Prints the mean SHD against the true network per number of sites, with
centralised GES on all the data as the first column.
"""

from __future__ import annotations

import argparse
import statistics

from _networks import load

from fedcausal import GES, FedGES
from fedcausal.data import contiguous_split
from fedcausal.metrics import shd


def main() -> None:
    """Run every network at every site count and print a Markdown table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--nets", default="cancer,asia,child,insurance,alarm,hepar2")
    parser.add_argument("--sites", default="5,10,20")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    counts = [int(k) for k in args.sites.split(",")]

    print(
        "| Network | Variables | Centralised | " + " | ".join(f"{k} sites" for k in counts) + " |"
    )
    print("|---|---:|---:|" + "---:|" * len(counts))
    for net in args.nets.split(","):
        cells: list[list[int]] = [[] for _ in range(len(counts) + 1)]
        for seed in range(args.seeds):
            data, truth = load(net, seed)
            cells[0].append(shd(truth, GES().fit(data).dag_))
            for i, k in enumerate(counts, start=1):
                model = FedGES(n_jobs=args.n_jobs).fit(contiguous_split(data, k))
                cells[i].append(shd(truth, model.dag_))
        means = " | ".join(f"{statistics.fmean(c):.1f}" for c in cells)
        print(f"| {net} | {data.n_vars} | {means} |", flush=True)


if __name__ == "__main__":
    main()
