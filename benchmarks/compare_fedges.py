"""Federated FedGES benchmark: structure quality against the number of clients.

Runs the in-process FedGES loop over ``(net, n_clients, sample)`` cells and
records each final global model's quality (BDeu, SHD, SMHD, edges) plus how many
rounds it took to converge. Sweeping several of a network's sampled datasets
turns each ``(net, n_clients)`` cell into a mean +/- sd rather than a single
number. Results append to ``benchmarks/results/compare_fedges.jsonl`` as each
cell completes, so a killed run resumes where it stopped.

**One metric authority.** Every published number (BDeu, SHD, SMHD) is computed
here with ``BDeuScore(ds, 1.0, 0.0)`` and ``fedcausal.metrics``, against the
ground truth re-indexed onto the dataset's column order -- the same discipline
as ``compare_ges.py``.

Usage::

    python benchmarks/compare_fedges.py
    python benchmarks/compare_fedges.py --net child --net alarm --clients 5 --clients 10
    python benchmarks/compare_fedges.py --n-samples 10   # mean +/- sd over 10 draws
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Reuse the loader from the GES comparator so both reports read one CSV set.
from compare_ges import load_net, networks_dir  # noqa: E402

from fedcausal.federated import run_fedges
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore

if TYPE_CHECKING:
    from collections.abc import Iterable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "compare_fedges.jsonl"

# Small-to-medium nets where the federated loop finishes quickly in-process.
NETS: tuple[str, ...] = ("asia", "cancer", "child", "insurance", "alarm", "hepar2")
CLIENTS: tuple[int, ...] = (5, 10, 20)
N_ITERATIONS = 50
# How many of each network's sampled datasets to sweep. Sample 0 alone gives a
# single number; a handful turns each cell into a mean +/- sd across
# independent draws from the same network.
N_SAMPLES = 5


def _score_row(row: dict[str, Any], ds: Any, truth: Any, dag: Any) -> dict[str, Any]:
    """Fill the shared quality columns for a rebuilt global DAG (one metric authority)."""
    score = BDeuScore(ds)
    row |= {
        "bdeu": score.score_dag(dag),
        "n_edges": dag.num_edges,
        "shd": shd(truth, dag) if truth is not None else None,
        "smhd": smhd(truth, dag) if truth is not None else None,
    }
    return row


def run_fedcausal_cell(net: str, n_clients: int, n_iterations: int, sample: int) -> dict[str, Any]:
    """Run one ``(net, n_clients, sample)`` fedcausal FedGES cell and return its result row."""
    ds, truth, _csv, _xbif = load_net(net, sample)
    row: dict[str, Any] = {
        "impl": "fedcausal",
        "net": net,
        "sample": sample,
        "n_vars": ds.n_vars,
        "n_rows": ds.n_rows,
        "n_clients": n_clients,
        "n_iterations_max": n_iterations,
    }
    start = time.perf_counter()
    try:
        global_model, stats = run_fedges(
            ds, n_clients=n_clients, n_iterations=n_iterations, truth=truth
        )
    except Exception as exc:  # noqa: BLE001 - a failure is a measurement outcome
        row |= {"status": "error", "seconds": time.perf_counter() - start, "notes": repr(exc)}
        return row
    seconds = time.perf_counter() - start
    server_rounds = sum(1 for s in stats if s.stage == "server")
    row |= {
        "status": "ok",
        "seconds": seconds,
        "rounds_run": max((s.iteration for s in stats), default=0),
        "fusions": server_rounds,
        "converged_early": server_rounds < n_iterations,
    }
    return _score_row(row, ds, truth, global_model.dag)


def existing_cells(path: Path) -> set[tuple[str, str, int, int]]:
    """Return the ``(impl, net, n_clients, sample)`` cells already recorded in `path`."""
    if not path.exists():
        return set()
    cells = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                cells.add((r["impl"], r["net"], r["n_clients"], r.get("sample", 0)))
    return cells


def append(path: Path, row: dict[str, Any]) -> None:
    """Append one result row to the JSONL, flushing so a kill loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        json.dump(row, fh)
        fh.write("\n")
        fh.flush()


def summarize(row: dict[str, Any]) -> str:
    """Return a one-line progress summary for stderr."""
    head = f"{row['impl']:<10}{row['net']:<12}c={row['n_clients']:<4}s={row.get('sample', 0):<3}"
    if row["status"] != "ok":
        return f"{head}{row['status'].upper():>8}  {row.get('notes', '')}"
    rounds = f" rounds={row['rounds_run']}" if "rounds_run" in row else ""
    return (
        f"{head}{row['seconds']:>7.1f}s  bdeu={row['bdeu']:.2f} "
        f"shd={row['shd']} smhd={row['smhd']} edges={row['n_edges']}{rounds}"
    )


def main(argv: Iterable[str] | None = None) -> int:
    """Run the requested cells, appending each to the results JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", action="append", dest="nets", help="restrict to this net")
    parser.add_argument(
        "--clients", action="append", dest="clients", type=int, help="restrict to this client count"
    )
    parser.add_argument(
        "--sample", action="append", dest="samples", type=int, help="restrict to this sample index"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help="sweep samples 0..N-1 (ignored if --sample is given)",
    )
    parser.add_argument("--n-iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--force", action="store_true", help="re-run cells already recorded")
    args = parser.parse_args(list(argv) if argv is not None else None)

    nets = tuple(args.nets) if args.nets else NETS
    clients = tuple(args.clients) if args.clients else CLIENTS
    samples = tuple(args.samples) if args.samples else tuple(range(args.n_samples))

    if not networks_dir().exists():
        print(f"networks directory not found: {networks_dir()}", file=sys.stderr)
        return 1

    done = set() if args.force else existing_cells(args.out)
    for net in nets:
        for n_clients in clients:
            for sample in samples:
                head = f"{net:<12}c={n_clients:<4}s={sample:<3}"
                if ("fedcausal", net, n_clients, sample) in done:
                    print(f"{head} (recorded, skip)", file=sys.stderr)
                    continue
                try:
                    row = run_fedcausal_cell(net, n_clients, args.n_iterations, sample)
                except FileNotFoundError as exc:
                    print(f"{head}  SKIP: {exc}", file=sys.stderr)
                    continue
                append(args.out, row)
                print(summarize(row), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
