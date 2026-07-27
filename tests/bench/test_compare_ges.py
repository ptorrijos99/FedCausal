"""Tests for the cross-implementation GES benchmark harness (``benchmarks/``).

The harness is reporting code and asserts nothing about performance, so these
tests do not check timings. They check the two things that could silently
corrupt a published table:

1. that a runner's name-keyed edge lists rebuild into the graph they described,
   and
2. that the whole ``fedcausal`` path -- runner subprocess, JSON contract,
   graph rebuild, re-scoring -- lands on the numbers already pinned by the
   committed reference run in ``tests/regression/ges_runs.json``.

The second test ties the harness to the committed reference runs: if it ever
starts reporting a different BDeu for a network whose value is pinned, that is
a harness bug, not a search result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmarks"
sys.path.insert(0, str(BENCH))

from compare_ges import rebuild_pdag  # noqa: E402

NETWORKS = REPO_ROOT / "tests" / "fixtures" / "networks"
GOLDEN = json.loads((REPO_ROOT / "tests" / "regression" / "ges_runs.json").read_text())


def test_rebuild_pdag_round_trips_named_edges() -> None:
    """A runner payload rebuilds into exactly the graph its names describe."""
    names = ("alpha", "beta", "gamma", "delta")
    payload = {
        "directed_edges": [["alpha", "gamma"], ["beta", "gamma"]],
        "undirected_edges": [["delta", "gamma"]],
    }

    pdag = rebuild_pdag(payload, names)

    assert sorted(pdag.directed_edges()) == [(0, 2), (1, 2)]
    assert pdag.undirected_edges() == [(2, 3)]


def test_rebuild_pdag_ignores_runner_column_order() -> None:
    """Edges are placed by name, so a runner's own column order cannot leak in.

    The three implementations do not agree on column ordering, which is why the
    contract is name-keyed. This pins that the mapping, not the payload order,
    decides indices.
    """
    names = ("gamma", "alpha", "beta")
    payload = {"directed_edges": [["alpha", "beta"]], "undirected_edges": []}

    pdag = rebuild_pdag(payload, names)

    assert pdag.directed_edges() == [(1, 2)]


def test_rebuild_pdag_rejects_unknown_variable() -> None:
    """A name absent from the dataset is an error, never a dropped edge."""
    with pytest.raises(KeyError):
        rebuild_pdag({"directed_edges": [["a", "zzz"]], "undirected_edges": []}, ("a", "b"))


@pytest.mark.regression
def test_fedcausal_runner_reproduces_reference() -> None:
    """The fedcausal runner on `asia` matches the BDeu pinned by the reference run.

    ``asia`` is one of the networks where fedcausal and the reference agree
    exactly (`child` and `alarm` diverge; see
    ``tests/regression/test_ges_golden.py``), so its pinned BDeu is a valid
    target for the default ``h_pool="nayx"`` the runner uses.
    """
    run = next(r for r in GOLDEN["runs"] if r["net"] == "asia")
    csv = NETWORKS / "asia.0.csv.gz"
    xbif = NETWORKS / "asia.xbif"

    proc = subprocess.run(
        [sys.executable, str(BENCH / "runners" / "run_fedcausal.py"), str(csv), str(xbif), "asia"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    assert payload["impl"] == "fedcausal"
    assert payload["n_vars"] == run["n_vars"]
    assert payload["self_reported_bdeu"] == pytest.approx(run["bdeu"], abs=1e-6)
    assert set(payload) >= {
        "seconds_search",
        "directed_edges",
        "undirected_edges",
        "notes",
    }
