"""Sweep the BDeu structure prior to test whether it improves recovered structure.

Motivation: in the cross-implementation benchmark, causal-learn recovers
markedly better structures than ``fedcausal`` on the large dense networks
(``win95pts`` SHD 38 vs 71, ``pathfinder`` 203 vs 233) while reaching a *worse*
BDeu. causal-learn is the only one of the three whose search carries a
structure-prior term; ``fedcausal`` and BayesFL both run with Tetrad's
effective ``structure_prior=0.0``, i.e. no complexity penalty at all.

But causal-learn also differs in its search, so that comparison cannot
attribute the gap to the prior. This script removes the confound: one search
(``fedcausal``'s), one dataset, one variable -- ``structure_prior``.

**The prior is already implemented.** ``BDeuScore._prior_term`` is a verbatim
port of Tetrad's ``getPriorForStructure``::

    num_parents*log(e/vm) + (vm - num_parents)*log(1 - e/vm)

which is the same Bernoulli-over-parents form causal-learn uses. The two differ
only in ``vm``: Tetrad passes the **row** count (``n_rows - 1``), causal-learn
the **variable** count (``n_features - 1``). Since the term depends only on
``num_parents``, it acts as a flat per-parent penalty of roughly
``log(e/vm)`` nats, so Tetrad's much larger ``vm`` makes the same ``e`` a
*stronger* penalty. ``--vm-mode`` selects between the two conventions so the
question "does a structure prior help?" is not silently entangled with the
question "which ``vm``?".

**Scoring.** The search uses the swept prior, but the reported BDeu is always
re-computed with ``structure_prior=0.0``. Scores from different priors are on
different scales and cannot be ranked against each other; the fixed scorer
makes the ``bdeu`` column comparable across rows, and SHD/SMHD are unaffected
by the prior anyway.

**Seeds.** Each ``(net, prior)`` cell is run over several of BayesFL's sampled
datasets (``<net>.0.csv`` .. ``<net>.<k-1>.csv``) and reported as mean +/-
standard deviation. GES is deterministic, so this is not repeat-measurement
noise: it is the spread over datasets drawn from the same network, which is the
only thing that can distinguish a prior that genuinely helps from one that
happened to suit a single sample. A difference smaller than the spread is not
a difference.

Raw per-run records go to ``results/structure_prior_sweep.jsonl`` so the
aggregation can be redone without re-running the searches.

Usage::

    python benchmarks/sweep_structure_prior.py --seeds 10
    python benchmarks/sweep_structure_prior.py --prior 0 --prior 1 --prior 10
    python benchmarks/sweep_structure_prior.py --vm-mode vars --net win95pts
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_ges import networks_dir  # noqa: E402

from fedcausal.scores import BDeuScore  # noqa: E402

# The block where all three implementations completed in the cross-comparison,
# plus the two large nets where causal-learn's advantage was largest.
NETS: tuple[str, ...] = (
    "asia",
    "child",
    "insurance",
    "water",
    "mildew",
    "alarm",
    "hepar2",
    "win95pts",
    "pathfinder",
)

PRIORS: tuple[float, ...] = (0.0, 0.001, 0.1, 1.0, 10.0)

RESULTS = Path(__file__).resolve().parent / "results" / "structure_prior_sweep.jsonl"
RUNNER = Path(__file__).resolve().parent / "runners" / "run_sweep_cell.py"


class VarCountPrior(BDeuScore):
    """``BDeuScore`` with causal-learn's ``vm``: the variable count, not rows.

    Tetrad passes the row count as ``getPriorForStructure``'s ``N``, making
    ``vm = n_rows - 1``. causal-learn uses ``vm = n_features - 1``. Both are the
    same functional form; only the implied per-edge probability ``e/vm``
    differs, and with it the penalty's strength. Subclassing keeps the ported
    Tetrad path in :class:`BDeuScore` untouched and verbatim.
    """

    def __init__(self, dataset, sample_prior=1.0, structure_prior=0.0, cache=True):  # noqa: ANN001, ANN204
        super().__init__(dataset, sample_prior, structure_prior, cache)
        # _prior_term reads self._n_rows; rebinding it to the variable count
        # switches the convention without duplicating the formula.
        self._n_rows = dataset.n_vars


def dataset_exists(net: str) -> bool:
    """Return whether `net`'s first sampled CSV is present."""
    return (networks_dir() / "res" / "networks" / "BBDD" / net / f"{net}.0.csv").exists()


@functools.cache
def shape_of(net: str) -> tuple[int, int]:
    """Return ``(n_rows, n_vars)`` of `net`'s first sample, read from its header.

    Only the shape is needed (to bound the prior), so the CSV is counted rather
    than parsed into a dataset.
    """
    path = networks_dir() / "res" / "networks" / "BBDD" / net / f"{net}.0.csv"
    with path.open(encoding="utf-8") as fh:
        header = fh.readline()
        n_rows = sum(1 for _ in fh)
    return n_rows, header.count(",") + 1


def run_cell(net: str, seed: int, prior: float, vm_mode: str, timeout: int) -> dict[str, Any]:
    """Run one search in a subprocess under a wall-clock budget.

    A search is isolated in its own process so that a single pathological
    dataset cannot stall the sweep. This is not hypothetical: an earlier,
    in-process version of this script ran for over two hours on ``pathfinder``
    seed 3, whose seeds 0-2 finish in 22-28 seconds each.

    Returns
    -------
    dict[str, Any]
        The runner's record with ``status="ok"``, or a record with
        ``status="timeout"``/``"error"`` carrying enough identity to be
        aggregated and skipped.
    """
    stub = {
        "net": net,
        "vm_mode": vm_mode,
        "structure_prior": prior,
        "seed": seed,
        "timeout_s": timeout,
    }
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                net,
                str(seed),
                repr(prior),
                vm_mode,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return stub | {"status": "timeout", "seconds": None}

    if proc.returncode != 0 or not proc.stdout.strip():
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return stub | {"status": "error", "seconds": None, "notes": " | ".join(tail)}

    return json.loads(proc.stdout.strip().splitlines()[-1]) | {
        "status": "ok",
        "timeout_s": timeout,
    }


def existing(path: Path, vm_mode: str) -> dict[tuple[str, float, int], dict[str, Any]]:
    """Return already-recorded runs for `vm_mode`, keyed by ``(net, prior, seed)``.

    Makes the sweep resumable: a killed run picks up where it stopped instead
    of repeating searches that can each cost minutes.
    """
    if not path.exists():
        return {}
    out: dict[tuple[str, float, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("vm_mode") != vm_mode:
                continue
            # Records written before per-cell budgets existed carry no status;
            # they completed, so treat them as ok.
            rec.setdefault("status", "ok")
            out[rec["net"], rec["structure_prior"], rec["seed"]] = rec
    return out


def spread(values: list[float], fmt: str = ".1f") -> str:
    """Format ``mean +/- sd`` for `values`, or just the mean for a single run."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return format(mean, fmt)
    return f"{format(mean, fmt)}+-{format(statistics.stdev(values), fmt)}"


def main(argv: list[str] | None = None) -> int:
    """Sweep the structure prior and print one aggregated row per (net, prior)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", action="append", dest="nets")
    parser.add_argument("--prior", action="append", dest="priors", type=float)
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of sampled datasets per cell (<net>.0.csv .. <net>.N-1.csv)",
    )
    parser.add_argument(
        "--vm-mode",
        choices=("rows", "vars"),
        default="rows",
        help="rows: Tetrad's vm = n_rows-1 (default). vars: causal-learn's vm = n_vars-1.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=(
            "per-search wall-clock budget in seconds. FES's T enumeration is "
            "exponential in neighbourhood width, which varies between samples of "
            "one network, so a single dataset can otherwise stall the sweep."
        ),
    )
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    nets = tuple(args.nets) if args.nets else NETS
    priors = tuple(args.priors) if args.priors else PRIORS
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"vm mode: {args.vm_mode}   seeds: {args.seeds}")
    header = (
        f"{'net':<12}{'prior':>8}{'secs':>8}{'bdeu(sp=0)':>18}{'edges':>12}{'shd':>12}{'smhd':>12}"
    )
    print(header)
    print("-" * len(header))

    done = existing(args.out, args.vm_mode)

    for net in nets:
        if not dataset_exists(net):
            print(f"{net:<12}  SKIP: no dataset under {networks_dir()}", flush=True)
            continue

        for prior in priors:
            secs: list[float] = []
            bdeus: list[float] = []
            edges: list[float] = []
            shds: list[float] = []
            smhds: list[float] = []
            killed = 0
            timed_out = 0

            # Tetrad's prior term evaluates log(1 - e/vm), so e must stay below
            # vm. vm is the row count in "rows" mode (4999 here) but the
            # variable count in "vars" mode, where the usable range is far
            # smaller -- e = 100 is fine for a 5000-row dataset and a domain
            # error for a 70-variable one.
            n_rows, n_vars = shape_of(net)
            vm = (n_rows if args.vm_mode == "rows" else n_vars) - 1
            if prior >= vm:
                print(
                    f"{net:<12}{prior:>8g}   SKIP: prior must be < vm = {vm} "
                    f"in {args.vm_mode} mode",
                    flush=True,
                )
                continue

            for seed in range(args.seeds):
                key = (net, prior, seed)
                record = done.get(key)
                if record is None:
                    record = run_cell(net, seed, prior, args.vm_mode, args.timeout)
                    with args.out.open("a", encoding="utf-8") as fh:
                        json.dump(record, fh)
                        fh.write("\n")

                if record.get("status") != "ok":
                    killed += 1
                    timed_out += record.get("status") == "timeout"
                    continue
                secs.append(record["seconds"])
                bdeus.append(record["bdeu_sp0"])
                edges.append(record["n_edges"])
                if record["shd"] is not None:
                    shds.append(record["shd"])
                    smhds.append(record["smhd"])

            if not secs:
                # Distinguish the two failure modes: a budget overrun says the
                # search is too slow, an error says the setting is invalid.
                # Reporting both as "over budget" would have hidden that
                # e >= vm is a domain error, not a performance problem.
                why = "exceeded the budget" if timed_out == killed else "failed"
                print(f"{net:<12}{prior:>8g}   all {killed} seeds {why}", flush=True)
                continue
            # n is printed whenever seeds were lost, so a cell aggregated over
            # fewer (and, being the ones that finished, systematically easier)
            # datasets is never mistaken for a complete one.
            suffix = f"   [n={len(secs)}/{args.seeds}, {killed} over budget]" if killed else ""
            print(
                f"{net:<12}{prior:>8g}{statistics.fmean(secs):>8.2f}"
                f"{spread(bdeus, '.1f'):>18}{spread(edges):>12}"
                f"{spread(shds) if shds else '-':>12}{spread(smhds) if smhds else '-':>12}"
                f"{suffix}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
