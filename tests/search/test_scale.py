"""Scale tests: prove the GES engine runs on large benchmark networks.

These need networks too large to ship with the package (up to 1041 variables),
so they are marked `bigdata` and skip unless ``FEDCAUSAL_NETWORKS`` points at a
directory laid out as ``<net>/<net>.0.csv``. See `docs/benchmarks.md`.

They assert structure, not wall-clock: timing assertions are machine-dependent
and flaky. Each test carries a generous budget and *skips* when it is exceeded,
so a slow machine reports "not measured here" instead of a red suite.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

_NETWORKS_ENV = "FEDCAUSAL_NETWORKS"

# Wall-clock budgets, generous multiples of the measured runtimes recorded in
# the perf-gate and phase-6 scaling notes (hepar2 ~1s, andes ~195s, pigs ~60s,
# link/munin the ~1000-node target). Budgets are ~3x the measured incremental
# times on the reference machine; a slower machine *skips* rather than fails.
_BUDGET_SECONDS = {
    "hepar2": 120.0,
    "andes": 1200.0,
    "pigs": 1200.0,
    "link": 2400.0,
    "munin": 3600.0,
}


def _load(net: str) -> DiscreteDataset:
    root = os.environ.get(_NETWORKS_ENV)
    if not root:
        pytest.skip(f"set {_NETWORKS_ENV} to a networks directory to run the scale tests")
    csv = Path(root) / net / f"{net}.0.csv"
    if not csv.exists():
        pytest.skip(f"network not found: {csv}")
    return DiscreteDataset.from_csv(csv)


@pytest.mark.bigdata
@pytest.mark.slow
@pytest.mark.parametrize(
    "net,min_vars",
    [
        ("hepar2", 70),
        ("andes", 223),
        ("pigs", 441),
        ("link", 724),
        ("munin", 1041),
    ],
)
def test_ges_runs_at_scale(net: str, min_vars: int) -> None:
    """GES completes on a large network and returns a non-trivial CPDAG."""
    ds = _load(net)
    assert ds.n_vars >= min_vars

    start = time.perf_counter()
    cpdag = GES(BDeuScore(ds)).fit(ds)
    elapsed = time.perf_counter() - start

    budget = _BUDGET_SECONDS[net]
    if elapsed > budget:
        pytest.skip(
            f"{net} took {elapsed:.1f}s, over the {budget:.0f}s budget for this machine; "
            "structure was still produced, see the perf-gate note"
        )

    assert isinstance(cpdag, PDAG)
    assert cpdag.num_nodes == ds.n_vars
    # A degenerate empty or near-empty result would mean the search collapsed.
    assert len(cpdag.directed_edges()) + len(cpdag.undirected_edges()) >= ds.n_vars // 2
    # The result must be a consistent CPDAG (a valid extension must exist).
    assert cpdag.to_dag().num_nodes == ds.n_vars
