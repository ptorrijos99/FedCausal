"""Shared helpers for the cross-implementation GES benchmark runners.

Every runner in this package obeys one contract: take ``<csv> <xbif> <net>``
on the command line, perform exactly **one** search, and print exactly **one**
JSON object to stdout. Anything diagnostic goes to stderr, so a noisy library
cannot corrupt the payload the orchestrator parses.

The emitted object is deliberately thin -- edge lists plus a stopwatch reading.
Runners do not compute BDeu, SHD, or SMHD for publication: the orchestrator
re-derives all three with one scorer, because the implementations disagree on
score scale and sign (see the design doc, "One metric authority").
"""

from __future__ import annotations

import json
import sys
from typing import Any

__all__ = ["canon", "emit", "parse_args"]


def parse_args(argv: list[str], impl: str) -> tuple[str, str, str]:
    """Return ``(csv, xbif, net)`` from `argv`, or exit 2 with a usage line."""
    if len(argv) != 4:
        print(f"Usage: {impl} <csv> <xbif> <net>", file=sys.stderr)
        raise SystemExit(2)
    return argv[1], argv[2], argv[3]


def canon(a: str, b: str) -> list[str]:
    """Return an undirected edge as a name pair in a canonical order."""
    return [a, b] if a <= b else [b, a]


def emit(payload: dict[str, Any]) -> None:
    """Print the single JSON result object to stdout."""
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()
