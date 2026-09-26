"""Measure what the two optional FES bounds save in time and cost in quality.

The exhaustive forward phase enumerates every subset ``T``, which is
exponential in neighbourhood width. The configurations, in ``run_bounds_cell``:

===============  =========================================================
``exhaustive``   No bound: Chickering (2002).
``cap1/2/3``     ``|T| <= k``. ``cap2`` is the library's default.
``gate``         The ``T = {}`` pre-gate alone.
``java``         Both, as the original Java implementation runs.
===============  =========================================================

Every configuration is scored with the same ``BDeuScore`` on the same data, so
the ``bdeu`` column compares across rows. A bound only removes candidates, so a
bounded row that beats ``exhaustive`` means a bug.

Each network is run over several sampled datasets, because runtimes vary a
lot between them: one ``pathfinder`` sample finishes in 22s, another did not
finish in two hours.

Usage::

    python benchmarks/sweep_fes_bounds.py --seeds 5
    python benchmarks/sweep_fes_bounds.py --net hailfinder --config exhaustive --config cap2
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "runners"))

from compare_ges import networks_dir  # noqa: E402
from run_bounds_cell import CONFIGS  # noqa: E402
from sweep_structure_prior import spread  # noqa: E402

# hailfinder and pathfinder are where the exhaustive search is slowest; the rest
# give a quality baseline.
NETS: tuple[str, ...] = (
    "child",
    "insurance",
    "water",
    "mildew",
    "alarm",
    "hailfinder",
    "hepar2",
    "win95pts",
    "pathfinder",
)

RESULTS = Path(__file__).resolve().parent / "results" / "fes_bounds_sweep.jsonl"
RUNNER = Path(__file__).resolve().parent / "runners" / "run_bounds_cell.py"


def run_cell(net: str, seed: int, config: str, timeout: int) -> dict[str, Any]:
    """Run one search in a subprocess under a wall-clock budget."""
    stub = {"net": net, "config": config, "seed": seed, "timeout_s": timeout}
    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER), net, str(seed), config],
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


def existing(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Return already-recorded runs keyed by ``(net, config, seed)``."""
    if not path.exists():
        return {}
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["net"], rec["config"], rec["seed"]] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the requested cells and print one aggregated row per (net, config)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", action="append", dest="nets")
    parser.add_argument("--config", action="append", dest="configs", choices=tuple(CONFIGS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-search budget; the unbounded config is expected to hit it",
    )
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    nets = tuple(args.nets) if args.nets else NETS
    configs = tuple(args.configs) if args.configs else tuple(CONFIGS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing(args.out)

    header = (
        f"{'net':<12}{'config':<12}{'secs':>10}{'bdeu':>16}{'edges':>11}{'shd':>11}{'smhd':>11}"
    )
    print(f"seeds: {args.seeds}   timeout: {args.timeout}s")
    print(header)
    print("-" * len(header))

    for net in nets:
        if not (networks_dir() / "res" / "networks" / "BBDD" / net / f"{net}.0.csv").exists():
            print(f"{net:<12}  SKIP: no dataset under {networks_dir()}", flush=True)
            continue

        for config in configs:
            secs: list[float] = []
            bdeus: list[float] = []
            edges: list[float] = []
            shds: list[float] = []
            smhds: list[float] = []
            lost = 0

            for seed in range(args.seeds):
                record = done.get((net, config, seed))
                if record is None:
                    record = run_cell(net, seed, config, args.timeout)
                    with args.out.open("a", encoding="utf-8") as fh:
                        json.dump(record, fh)
                        fh.write("\n")
                if record.get("status") != "ok":
                    lost += 1
                    continue
                secs.append(record["seconds"])
                bdeus.append(record["bdeu"])
                edges.append(record["n_edges"])
                if record["shd"] is not None:
                    shds.append(record["shd"])
                    smhds.append(record["smhd"])

            if not secs:
                print(f"{net:<12}{config:<12}   all {lost} seeds lost (budget/error)", flush=True)
                continue
            # A cell that lost seeds is aggregated over the ones that
            # finished, which are systematically the easy samples. Saying so
            # matters most for "exhaustive", where the lost seeds are exactly
            # the evidence for bounding.
            suffix = f"   [n={len(secs)}/{args.seeds}, {lost} lost]" if lost else ""
            print(
                f"{net:<12}{config:<12}{statistics.fmean(secs):>10.2f}"
                f"{statistics.fmean(bdeus):>16.2f}{spread(edges):>11}"
                f"{spread(shds) if shds else '-':>11}{spread(smhds) if smhds else '-':>11}"
                f"{suffix}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
