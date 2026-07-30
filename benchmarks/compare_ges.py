"""Cross-implementation GES benchmark: orchestrator.

Runs ``fedcausal`` and causal-learn over the same benchmark networks, one
subprocess per ``(net, impl)`` cell, appending each finished cell to
``benchmarks/results/compare_ges.jsonl`` as it completes.

Design points that matter for reading the numbers this produces:

**One subprocess per cell.** The largest networks are expected to hang or
exhaust memory in at least one implementation, and a per-cell process means
such a failure costs one cell rather than the run. It also gives a timeout that
works on Windows -- ``subprocess.run(timeout=...)`` does, whereas
``bench_ges.py``'s ``signal.SIGALRM`` budget is a documented no-op there.

**One metric authority.** Runners return edge lists only. Every published
number (BDeu, SHD, SMHD) is computed *here*, with ``fedcausal``'s
``BDeuScore(ds, 1.0, 0.0)`` and ``fedcausal.metrics``. This is required, not
cosmetic: causal-learn's search optimises a BDeu with different priors on a
different scale (see ``runners/run_causallearn.py``), so implementations'
self-reported scores are not comparable with each other.

**Resumable.** Cells already present in the JSONL are skipped unless
``--force`` is given, so a killed run resumes where it stopped.

Usage::

    python benchmarks/compare_ges.py                       # everything
    python benchmarks/compare_ges.py --net asia --net alarm
    python benchmarks/compare_ges.py --impl fedcausal --timeout 7200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG, PDAG
from fedcausal.io import load_xbif
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore

if TYPE_CHECKING:
    from collections.abc import Iterable

_NETWORKS_ENV = "FEDCAUSAL_NETWORKS"

HERE = Path(__file__).resolve().parent
RUNNERS = HERE / "runners"
RESULTS = HERE / "results" / "compare_ges.jsonl"

# The scaling ladder, ordered by node count. Mirrors bench_ges.SCALE_NETS so
# the two notes' tables line up row for row.
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
    "win95pts",
    "pathfinder",
    "andes",
    "diabetes",
    "pigs",
    "link",
    "munin",
)

IMPLS: tuple[str, ...] = ("fedcausal", "causal-learn")


def networks_dir() -> Path:
    """Return the benchmark networks root (``FEDCAUSAL_NETWORKS``).

    Expected layout: ``<root>/BBDD/<net>/<net>.<sample>.csv`` for the sampled
    datasets and ``<root>/<net>.xbif`` for the ground truths. See
    ``docs/benchmarks.md``.

    Raises
    ------
    RuntimeError
        If the variable is unset.
    """
    root = os.environ.get(_NETWORKS_ENV)
    if not root:
        raise RuntimeError(
            f"set {_NETWORKS_ENV} to a benchmark networks directory; see docs/benchmarks.md"
        )
    return Path(root)


def load_net(net: str, sample: int = 0) -> tuple[DiscreteDataset, DAG | None, Path, Path]:
    """Load one of a network's sampled CSVs and its ground-truth DAG, plus both paths.

    The ground truth is re-indexed onto the dataset's column order when the two
    disagree, so that node indices mean the same thing in the returned DAG as
    in `ds`.

    Parameters
    ----------
    net : str
        Network name, e.g. ``"alarm"``.
    sample : int, optional
        Which of the network's sampled datasets to load (``<net>.<sample>.csv``).
        The cross-implementation benchmark uses only sample 0, since GES is
        deterministic and repeating one dataset would measure stopwatch noise;
        sweeps that compare *settings* do vary it, because there the quantity
        of interest is how a setting behaves across datasets from the same
        network.
    """
    root = networks_dir()
    csv = root / "res" / "networks" / "BBDD" / net / f"{net}.{sample}.csv"
    xbif = root / "res" / "networks" / f"{net}.xbif"
    if not csv.exists():
        raise FileNotFoundError(f"dataset not found: {csv}")
    ds = DiscreteDataset.from_csv(csv)

    truth: DAG | None = None
    if xbif.exists():
        bn = load_xbif(str(xbif))
        if tuple(bn.names) == ds.names:
            truth = bn.dag
        else:
            index = {name: i for i, name in enumerate(ds.names)}
            truth = DAG.from_edges(
                ds.n_vars,
                [(index[bn.names[u]], index[bn.names[v]]) for u, v in bn.dag.edges()],
            )
    return ds, truth, csv, xbif


def rebuild_pdag(payload: dict[str, Any], names: tuple[str, ...]) -> PDAG:
    """Rebuild a runner's edge lists as a PDAG over `names`' index order.

    Runners emit edges by variable name precisely so this step can exist: the
    implementations do not agree on column ordering, nor on level encoding.

    Raises
    ------
    KeyError
        If a runner returned a variable name absent from the dataset, which
        would mean the two read different files.
    """
    index = {name: i for i, name in enumerate(names)}
    pdag = PDAG(len(names))
    for u, v in payload["directed_edges"]:
        pdag.add_directed(index[u], index[v])
    for u, v in payload["undirected_edges"]:
        pdag.add_undirected(index[u], index[v])
    return pdag


def command(impl: str, csv: Path, xbif: Path, net: str) -> list[str]:
    """Return the subprocess command line for one ``(impl, net)`` cell."""
    script = "run_fedcausal.py" if impl == "fedcausal" else "run_causallearn.py"
    return [sys.executable, str(RUNNERS / script), str(csv), str(xbif), net]


def run_cell(impl: str, net: str, timeout: int) -> dict[str, Any]:
    """Run one ``(impl, net)`` cell and return its result row.

    Never raises for an implementation's own failure: a timeout or a crash is a
    measurement outcome and is recorded as such, with ``status`` set to
    ``"timeout"`` or ``"error"``.
    """
    ds, truth, csv, xbif = load_net(net)
    row: dict[str, Any] = {
        "impl": impl,
        "net": net,
        "n_vars": ds.n_vars,
        "n_rows": ds.n_rows,
        "timeout_s": timeout,
    }

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command(impl, csv, xbif, net),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        row |= {
            "status": "timeout",
            "seconds_total": float(timeout),
            "seconds_search": None,
            "notes": f"exceeded the {timeout}s budget",
        }
        return row
    total = time.perf_counter() - start

    if proc.returncode != 0 or not proc.stdout.strip():
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        row |= {
            "status": "error",
            "seconds_total": total,
            "seconds_search": None,
            "notes": f"exit {proc.returncode}: " + " | ".join(tail),
        }
        return row

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    pdag = rebuild_pdag(payload, ds.names)
    dag = pdag.to_dag()
    score = BDeuScore(ds)

    row |= {
        "status": "ok",
        "seconds_total": total,
        "seconds_search": payload["seconds_search"],
        # Every implementation's graph re-scored by the same scorer with
        # Tetrad's effective priors -- the only cross-implementation-comparable
        # quality number in this file.
        "bdeu": score.score_dag(dag),
        "self_reported_bdeu": payload.get("self_reported_bdeu"),
        "n_edges": dag.num_edges,
        "n_directed": len(payload["directed_edges"]),
        "n_undirected": len(payload["undirected_edges"]),
        # The learned graph itself, kept so a surprising cell can be diagnosed
        # from the results file instead of by re-running it. Two runs that tie
        # on BDeu need not agree on structure -- hailfinder is exactly that
        # case -- and only the edge lists can tell the two situations apart.
        "directed_edges": payload["directed_edges"],
        "undirected_edges": payload["undirected_edges"],
        "shd": shd(truth, dag) if truth is not None else None,
        "smhd": smhd(truth, dag) if truth is not None else None,
        "notes": payload.get("notes", ""),
    }
    return row


def existing_cells(path: Path) -> set[tuple[str, str]]:
    """Return the ``(impl, net)`` cells already recorded in `path`."""
    if not path.exists():
        return set()
    cells = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                cells.add((row["impl"], row["net"]))
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
    head = f"{row['net']:<12}{row['impl']:<14}"
    if row["status"] != "ok":
        return f"{head}{row['status'].upper():>10}  {row['notes']}"
    return (
        f"{head}{row['seconds_search']:>10.2f}s search "
        f"{row['seconds_total']:>8.2f}s total  "
        f"bdeu={row['bdeu']:.2f} shd={row['shd']} smhd={row['smhd']}"
    )


def main(argv: Iterable[str] | None = None) -> int:
    """Run the requested cells, appending each to the results JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", action="append", dest="nets", help="restrict to this net")
    parser.add_argument("--impl", action="append", dest="impls", choices=IMPLS)
    parser.add_argument(
        "--timeout", type=int, default=3600, help="per-cell wall-clock budget in seconds"
    )
    parser.add_argument(
        "--impl-timeout",
        action="append",
        default=[],
        metavar="IMPL=SECONDS",
        help=(
            "override --timeout for one implementation. Each cell records the budget "
            "it was actually given, and the report prints it, so differing budgets "
            "stay visible rather than hiding behind a single number."
        ),
    )
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--force", action="store_true", help="re-run cells already recorded")
    args = parser.parse_args(list(argv) if argv is not None else None)

    nets = tuple(args.nets) if args.nets else NETS
    impls = tuple(args.impls) if args.impls else IMPLS

    budgets: dict[str, int] = {impl: int(args.timeout) for impl in IMPLS}
    for spec in args.impl_timeout:
        impl, _, seconds = spec.partition("=")
        if impl not in IMPLS or not seconds.isdigit():
            parser.error(f"--impl-timeout expects IMPL=SECONDS with IMPL in {IMPLS}, got {spec!r}")
        budgets[impl] = int(seconds)

    done = set() if args.force else existing_cells(args.out)

    for net in nets:
        for impl in impls:
            if (impl, net) in done:
                print(f"{net:<12}{impl:<14}     (already recorded, skipping)", file=sys.stderr)
                continue
            try:
                row = run_cell(impl, net, budgets[impl])
            except FileNotFoundError as exc:
                print(f"{net:<12}{impl:<14}      SKIP: {exc}", file=sys.stderr)
                continue
            append(args.out, row)
            print(summarize(row), file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
