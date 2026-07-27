"""Cross-implementation GES benchmark: render ``compare_ges.jsonl`` as markdown.

Emits three tables -- runtime, quality, and a consistency check -- plus the
runner notes. Reads only what ``compare_ges.py`` wrote; it recomputes nothing,
so the report can never disagree with the recorded measurements.

Usage::

    python benchmarks/report_ges.py [--in results/compare_ges.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "compare_ges.jsonl"
IMPLS: tuple[str, ...] = ("fedcausal", "bayesfl", "causal-learn")

# Column width for the leading net/vars columns, so the three tables align.
CELL_MISSING = "--"


def load(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    """Return ``{(net, impl): row}`` and the net order as first recorded."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # A re-run of the same cell overwrites the earlier measurement.
            rows[row["net"], row["impl"]] = row
            if row["net"] not in order:
                order.append(row["net"])
    return rows, order


def cell(row: dict[str, Any] | None, key: str, fmt: str) -> str:
    """Format one measurement, or a marker for missing/failed cells."""
    if row is None:
        return CELL_MISSING
    if row["status"] == "timeout":
        return f"**>{row['timeout_s']:.0f}s**"
    if row["status"] == "error":
        return "**err**"
    value = row.get(key)
    if value is None:
        return CELL_MISSING
    return format(value, fmt)


def table(
    rows: dict[tuple[str, str], dict[str, Any]],
    order: list[str],
    title: str,
    columns: list[tuple[str, str, str]],
) -> str:
    """Render one markdown table.

    Parameters
    ----------
    columns : list[tuple[str, str, str]]
        ``(header, row_key, format_spec)`` triples, repeated once per
        implementation.
    """
    heads = ["net", "vars"]
    for impl in IMPLS:
        heads += [f"{header}<br>{impl}" for header, _, _ in columns]

    lines = [f"### {title}", "", "| " + " | ".join(heads) + " |"]
    lines.append("|" + "|".join(["---"] * len(heads)) + "|")

    for net in order:
        any_row = next((rows.get((net, i)) for i in IMPLS if (net, i) in rows), None)
        cells = [net, str(any_row["n_vars"]) if any_row else CELL_MISSING]
        for impl in IMPLS:
            row = rows.get((net, impl))
            cells += [cell(row, key, fmt) for _, key, fmt in columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def consistency(rows: dict[tuple[str, str], dict[str, Any]], order: list[str]) -> str:
    """Report whether each implementation's self-reported BDeu was reproduced.

    Only meaningful for ``bayesfl``: it and ``fedcausal`` score with the same
    priors on the same scale, so a mismatch is a harness bug. causal-learn's
    self-reported score is on its own scale (different priors, and treated as a
    cost by its search), so it is expected to differ and is excluded.
    """
    lines = ["### Self-reported vs re-scored BDeu (harness check)", ""]
    bad = []
    for net in order:
        for impl in ("fedcausal", "bayesfl"):
            row = rows.get((net, impl))
            if row is None or row["status"] != "ok" or row.get("self_reported_bdeu") is None:
                continue
            delta = abs(row["bdeu"] - row["self_reported_bdeu"])
            if delta > 1e-6:
                bad.append(f"- `{impl}` on `{net}`: delta {delta:.6g}")
    if bad:
        lines.append("**Mismatches found — treat the quality table as suspect:**")
        lines += bad
    else:
        lines.append(
            "All `fedcausal` and `bayesfl` cells reproduce their own reported BDeu to "
            "within 1e-6. causal-learn is excluded: its search optimises a different "
            "objective, so its self-reported score is not on this scale."
        )
    return "\n".join(lines)


def notes_section(rows: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Collect the distinct runner notes, so caveats travel with the numbers."""
    seen: dict[str, set[str]] = {}
    for (_, impl), row in rows.items():
        note = (row.get("notes") or "").strip()
        if note and row["status"] == "ok":
            seen.setdefault(impl, set()).add(note)
    if not seen:
        return ""
    lines = ["### Runner notes", ""]
    for impl in IMPLS:
        for note in sorted(seen.get(impl, ())):
            lines.append(f"- **{impl}**: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Render the report to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="path", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"no results at {args.path}; run compare_ges.py first", file=sys.stderr)
        return 1

    rows, order = load(args.path)

    print(
        table(
            rows,
            order,
            "Runtime",
            [("search s", "seconds_search", ".2f"), ("total s", "seconds_total", ".2f")],
        )
    )
    print()
    print(
        table(
            rows,
            order,
            "Quality (BDeu re-scored by fedcausal, priors 1.0/0.0)",
            [
                ("bdeu", "bdeu", ".2f"),
                ("edges", "n_edges", "d"),
                ("shd", "shd", "d"),
                ("smhd", "smhd", "d"),
            ],
        )
    )
    print()
    print(consistency(rows, order))
    notes = notes_section(rows)
    if notes:
        print()
        print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
