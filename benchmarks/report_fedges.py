"""Render the FedGES benchmark JSONL as a quality-vs-clients table.

Reads ``benchmarks/results/compare_fedges.jsonl`` (produced by
``compare_fedges.py``) and prints, per ``(net, n_clients)`` cell, the final
global model's BDeu / SHD / SMHD / edges and the round count, aggregated over
however many samples were run: a single number when one sample is present,
``mean +/- sd`` when several are. Asserts nothing.

Usage::

    python benchmarks/report_fedges.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "compare_fedges.jsonl"


def load(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Return ``{(net, n_clients): [rows]}`` from the JSONL.

    A later write for the same ``(net, n_clients, sample)`` replaces the earlier
    one, so a re-run corrects rather than duplicates.
    """
    by_sample: dict[tuple[str, int, int], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                by_sample[(r["net"], r["n_clients"], r.get("sample", 0))] = r
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (net, c, _s), r in by_sample.items():
        grouped[(net, c)].append(r)
    return grouped


def _stats(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float, int]:
    """Return ``(mean, sd, n)`` of `key` over the ``status == "ok"`` rows (sample sd)."""
    vals = [r[key] for r in rows if r.get("status") == "ok" and r.get(key) is not None]
    n = len(vals)
    if n == 0:
        return None, 0.0, 0
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mean, sd, n


def _fmt(mean: float | None, sd: float, n: int, decimals: int) -> str:
    """Format ``mean`` (with ``+/- sd`` when more than one sample) or ``-``."""
    if mean is None:
        return "-"
    if n <= 1:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}+-{sd:.{decimals}f}"


def main() -> int:
    """Print the benchmark table."""
    grouped = load(RESULTS)
    if not grouped:
        print(f"no results at {RESULTS}", file=sys.stderr)
        return 1

    n_samples_seen: set[int] = set()
    print(
        f"{'net':<12}{'cli':>4} | {'BDeu':>18} | {'SHD':>12} | "
        f"{'SMHD':>12} | {'edges':>12} | {'rounds':>10}"
    )
    print("-" * 94)
    for net, c in sorted(grouped):
        rows = grouped[(net, c)]
        bdeu, bdeu_sd, n = _stats(rows, "bdeu")
        n_samples_seen.add(n)
        if n == 0:
            print(f"{net:<12}{c:>4} | (no successful runs)")
            continue
        shd_m, shd_sd, _ = _stats(rows, "shd")
        smhd_m, smhd_sd, _ = _stats(rows, "smhd")
        edges_m, edges_sd, _ = _stats(rows, "n_edges")
        rounds_m, rounds_sd, _ = _stats(rows, "rounds_run")
        print(
            f"{net:<12}{c:>4} | {_fmt(bdeu, bdeu_sd, n, 1):>18} | "
            f"{_fmt(shd_m, shd_sd, n, 1):>12} | {_fmt(smhd_m, smhd_sd, n, 1):>12} | "
            f"{_fmt(edges_m, edges_sd, n, 1):>12} | {_fmt(rounds_m, rounds_sd, n, 1):>10}"
        )

    print("-" * 94)
    n_str = ", ".join(str(n) for n in sorted(n_samples_seen - {0}))
    print(f"samples per cell: {n_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
