"""Measure what `structure_prior` buys, by sampling from known networks.

With no arguments it uses the five networks bundled with the package, so anyone
can re-run it and reproduce the small-network table in ``docs/algorithms.md``::

    python benchmarks/sweep_bundled_prior.py

Point it at a directory of ``.xbif`` files to cover the larger networks the
package does not ship, which is where the prior earns its keep::

    python benchmarks/sweep_bundled_prior.py --networks /path/to/BNs --out results.json

It needs no CSVs: it forward-samples from the network definition, so a run is
fully determined by the ``.xbif`` and the seed. The paragraph in
``docs/algorithms.md`` is written from its output.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from fedcausal.datasets import list_networks, load_network
from fedcausal.io import forward_sample, load_xbif
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

PRIORS = [0.0, 1.0]
SEEDS = [0, 1, 2, 3, 4]
N_ROWS = 5000


def main() -> None:
    """Sweep `structure_prior` over a set of networks and record the result."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--networks",
        type=Path,
        help="directory of .xbif files; default is the bundled networks",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "default: results/structure_prior_networks.json with --networks, "
            "results/bundled_prior.json without, so a bundled-only run cannot "
            "silently overwrite a full one"
        ),
    )
    parser.add_argument(
        "--max-vars", type=int, default=10_000, help="skip networks larger than this"
    )
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()

    if args.networks:
        sources = [(f.stem, load_xbif(f)) for f in sorted(args.networks.glob("*.xbif"))]
        default_out = Path("benchmarks/results/structure_prior_networks.json")
    else:
        sources = [(name, load_network(name).network) for name in list_networks()]
        default_out = Path("benchmarks/results/bundled_prior.json")
    if args.out is None:
        args.out = default_out
    sources = [(n, bn) for n, bn in sources if len(bn.names) <= args.max_vars]
    sources.sort(key=lambda item: len(item[1].names))

    records = []
    for name, network in sources:
        started = time.perf_counter()
        for seed in SEEDS[: args.seeds]:
            data = forward_sample(network, N_ROWS, seed=seed)
            # forward_sample keeps the network's node order, so the truth needs
            # no re-indexing here.
            truth = network.dag
            for prior in PRIORS:
                learned = GES(BDeuScore(data, structure_prior=prior)).fit(data).to_dag()
                records.append(
                    {
                        "net": name,
                        "n_vars": data.n_vars,
                        "seed": seed,
                        "structure_prior": prior,
                        "edges": len(learned.edges()),
                        "shd": shd(truth, learned),
                        "smhd": smhd(truth, learned),
                    }
                )
        print(f"  {name} ({len(network.names)} vars) done in {time.perf_counter() - started:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_rows": N_ROWS, "runs": records}, indent=1), encoding="utf-8")

    print(f"\nwrote {len(records)} runs to {args.out}\n")
    print(f"{'net':<12}{'vars':>6}" + "".join(f"{p:>10}" for p in PRIORS))
    totals = dict.fromkeys(PRIORS, 0.0)
    for name, network in sources:
        row = f"{name:<12}{len(network.names):>6}"
        for prior in PRIORS:
            vals = [
                r["smhd"] for r in records if r["net"] == name and r["structure_prior"] == prior
            ]
            mean = statistics.mean(vals)
            totals[prior] += mean
            row += f"{mean:>10.1f}"
        print(row)
    print(f"{'SUM':<12}{'':>6}" + "".join(f"{totals[p]:>10.1f}" for p in PRIORS))


if __name__ == "__main__":
    main()
