"""Benchmark the fedcausal GES engine: runtime scaling and H-pool comparison.

This is a reporting script, not a test: it asserts nothing and prints tables.

Two modes:

``--mode scale``
    Time ``GES(BDeuScore(ds)).fit(ds)`` across a ladder of benchmark networks,
    reporting wall-clock, final BDeu, edge count, and structural error against
    the ground-truth network.

``--mode hpool``
    Run each network twice, once with ``h_pool="nayx"`` (Chickering's correct
    Delete operator) and once with ``h_pool="bayesfl"`` (which replicates
    BayesFL's BES, where the ``H`` pool is drawn from ``T0`` and the search
    therefore only ever applies ``Delete(x, y, {})``). Reports the quality and
    runtime difference between the two.

Data is read from a BayesFL checkout: CSVs from
``$FEDCAUSAL_NETWORKS/res/networks/BBDD/<net>/<net>.0.csv`` and ground-truth networks
from ``$FEDCAUSAL_NETWORKS/res/networks/<net>.xbif``. ``FEDCAUSAL_NETWORKS`` defaults to
``../BayesFL`` relative to the repository root.

Usage::

    python benchmarks/bench_ges.py --mode hpool --net mildew --net hailfinder
    python benchmarks/bench_ges.py --mode scale --timeout 900

.. warning::
   ``--timeout`` is enforced with ``signal.SIGALRM`` and is therefore a no-op
   on Windows, where a long run must be bounded externally instead (for
   example ``timeout 900 python benchmarks/bench_ges.py ...`` under Git Bash).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fedcausal.data import DiscreteDataset
from fedcausal.io import load_xbif
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

if TYPE_CHECKING:
    from fedcausal.graph import DAG

# Networks ordered by node count -- the scaling ladder.
SCALE_NETS: tuple[str, ...] = (
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

# Networks for the H-pool comparison. mildew and hailfinder are the cases the
# project owner flagged as forming large common neighbourhoods, which is where
# a non-empty H should matter most.
HPOOL_NETS: tuple[str, ...] = (
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


def networks_dir() -> Path:
    """Return the benchmark networks root (``FEDCAUSAL_NETWORKS``); see docs/benchmarks.md."""
    return Path(os.environ.get("FEDCAUSAL_NETWORKS", "../networks"))


def load_net(net: str) -> tuple[DiscreteDataset, DAG | None]:
    """Load a network's first CSV and, when available, its ground-truth DAG."""
    root = networks_dir()
    csv = root / "res" / "networks" / "BBDD" / net / f"{net}.0.csv"
    if not csv.exists():
        raise FileNotFoundError(f"dataset not found: {csv}")
    ds = DiscreteDataset.from_csv(csv)

    truth: DAG | None = None
    xbif = root / "res" / "networks" / f"{net}.xbif"
    if xbif.exists():
        bn = load_xbif(str(xbif))
        if tuple(bn.names) == ds.names:
            truth = bn.dag
        else:
            index = {name: i for i, name in enumerate(ds.names)}
            from fedcausal.graph import DAG as _DAG

            truth = _DAG.from_edges(
                ds.n_vars,
                [(index[bn.names[u]], index[bn.names[v]]) for u, v in bn.dag.edges()],
            )
    return ds, truth


class _Timeout(Exception):
    """Raised when a single run exceeds its wall-clock budget."""


def _run_one(ds: DiscreteDataset, h_pool: str) -> tuple[float, float, int, object]:
    """Run GES once, returning ``(seconds, bdeu, n_edges, cpdag)``."""
    score = BDeuScore(ds)
    start = time.perf_counter()
    cpdag = GES(score, h_pool=h_pool).fit(ds)  # type: ignore[arg-type]
    seconds = time.perf_counter() - start
    dag = cpdag.to_dag()
    return seconds, score.score_dag(dag), dag.num_edges, cpdag


def _errors(cpdag: object, truth: DAG | None) -> tuple[str, str]:
    """Return ``(shd, smhd)`` against `truth` as strings, or ``("-", "-")``."""
    if truth is None:
        return "-", "-"
    dag = cpdag.to_dag()  # type: ignore[attr-defined]
    return str(shd(truth, dag)), str(smhd(truth, dag))


def mode_scale(nets: tuple[str, ...], timeout: int) -> None:
    """Print the runtime scaling ladder."""
    print(
        f"{'net':<12}{'vars':>6}{'rows':>8}{'secs':>10}{'bdeu':>16}{'edges':>7}{'shd':>6}{'smhd':>6}"
    )
    print("-" * 72)
    for net in nets:
        try:
            ds, truth = load_net(net)
        except FileNotFoundError as exc:
            print(f"{net:<12}  SKIP: {exc}")
            continue
        try:
            with _budget(timeout):
                seconds, bdeu, edges, cpdag = _run_one(ds, "nayx")
        except _Timeout:
            print(f"{net:<12}{ds.n_vars:>6}{ds.n_rows:>8}   TIMEOUT after {timeout}s")
            continue
        s, m = _errors(cpdag, truth)
        print(
            f"{net:<12}{ds.n_vars:>6}{ds.n_rows:>8}{seconds:>10.2f}"
            f"{bdeu:>16.2f}{edges:>7}{s:>6}{m:>6}"
        )


def mode_hpool(nets: tuple[str, ...], timeout: int) -> None:
    """Print the ``nayx`` vs ``bayesfl`` H-pool comparison."""
    header = (
        f"{'net':<12}{'vars':>5}"
        f"{'bdeu(nayx)':>15}{'bdeu(bfl)':>15}{'delta':>10}"
        f"{'shd':>5}{'shd':>5}{'smhd':>6}{'smhd':>6}"
        f"{'s(nayx)':>9}{'s(bfl)':>9}"
    )
    print(header)
    print(f"{'':<12}{'':>5}{'':>15}{'':>15}{'':>10}{'nayx':>5}{'bfl':>5}{'nayx':>6}{'bfl':>6}")
    print("-" * len(header))
    for net in nets:
        try:
            ds, truth = load_net(net)
        except FileNotFoundError as exc:
            print(f"{net:<12}  SKIP: {exc}")
            continue
        row: dict[str, tuple[float, float, int, object]] = {}
        timed_out = False
        for pool in ("nayx", "bayesfl"):
            try:
                with _budget(timeout):
                    row[pool] = _run_one(ds, pool)
            except _Timeout:
                print(f"{net:<12}{ds.n_vars:>5}   TIMEOUT ({pool}) after {timeout}s")
                timed_out = True
                break
        if timed_out:
            continue
        sn, bn_ = row["nayx"], row["bayesfl"]
        shd_n, smhd_n = _errors(sn[3], truth)
        shd_b, smhd_b = _errors(bn_[3], truth)
        print(
            f"{net:<12}{ds.n_vars:>5}{sn[1]:>15.2f}{bn_[1]:>15.2f}{sn[1] - bn_[1]:>10.2f}"
            f"{shd_n:>5}{shd_b:>5}{smhd_n:>6}{smhd_b:>6}{sn[0]:>9.2f}{bn_[0]:>9.2f}"
        )


class _budget:
    """Context manager enforcing a wall-clock budget (POSIX ``SIGALRM`` only)."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.supported = hasattr(signal, "SIGALRM")

    def __enter__(self) -> _budget:
        if self.supported:
            signal.signal(signal.SIGALRM, self._fire)  # type: ignore[attr-defined]
            signal.alarm(self.seconds)  # type: ignore[attr-defined]
        return self

    def _fire(self, *_: object) -> None:
        raise _Timeout

    def __exit__(self, *_: object) -> None:
        if self.supported:
            signal.alarm(0)  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested benchmark mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scale", "hpool"), default="scale")
    parser.add_argument("--net", action="append", dest="nets", help="restrict to this net")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="per-run wall-clock budget in seconds (POSIX only)",
    )
    args = parser.parse_args(argv)

    default = SCALE_NETS if args.mode == "scale" else HPOOL_NETS
    nets = tuple(args.nets) if args.nets else default
    if args.mode == "scale":
        mode_scale(nets, args.timeout)
    else:
        mode_hpool(nets, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
