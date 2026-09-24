"""Scale tests: prove the GES engine runs on large benchmark networks.

These need networks too large to ship with the package (up to 1041 variables),
so they are marked `bigdata` and skip unless ``FEDCAUSAL_NETWORKS`` points at a
directory laid out as ``<net>/<net>.0.csv``. See `docs/benchmarks.md`.

They assert structure, not wall-clock: timings belong in the benchmarks, where
they are measured properly, not in a pass/fail check.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

_NETWORKS_ENV = "FEDCAUSAL_NETWORKS"


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

    cpdag = GES(BDeuScore(ds)).fit(ds)

    assert isinstance(cpdag, PDAG)
    assert cpdag.num_nodes == ds.n_vars
    # A degenerate empty or near-empty result would mean the search collapsed.
    assert len(cpdag.directed_edges()) + len(cpdag.undirected_edges()) >= ds.n_vars // 2
    # The result must be a consistent CPDAG (a valid extension must exist).
    assert cpdag.to_dag().num_nodes == ds.n_vars
