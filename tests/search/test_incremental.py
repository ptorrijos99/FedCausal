"""Differential test: the incremental FES/BES engine equals the full scan, byte-for-byte.

The incremental engine (:mod:`fedcausal.search.incremental`) is a performance
optimization of the full-scan GES (:mod:`fedcausal.search.ges`): it re-scores
only the candidate pairs whose neighbourhood changed after each operator. Its
score-delta invalidation is provably complete, but operator *validity* is not
fully local (semi-directed-path and clique checks), so equality with the full
scan is not proven for every conceivable graph -- it is *enforced empirically
here*. Every case asserts the two engines produce byte-identical CPDAGs (same
directed and undirected edge sets), so any future divergence fails CI.

Coverage: an always-on synthetic dataset (so CI without the BayesFL tree still
exercises the engine) plus the real benchmark networks when ``$BAYESFL_DIR`` is
available, across both ``max_t_size`` settings, both ``h_pool`` values, and the
``candidate_edges``-restricted (federated) path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph.pdag import PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search import GES
from fedcausal.search.ges import run_bes, run_fes
from fedcausal.search.incremental import run_bes_incremental, run_fes_incremental

REPO_ROOT = Path(__file__).resolve().parents[2]
BAYESFL = Path(os.environ.get("BAYESFL_DIR", str(REPO_ROOT.parent / "BayesFL")))
REAL_NETS = ("asia", "cancer", "child", "insurance", "alarm", "hepar2")


def _canon(pdag: PDAG) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return a hashable, order-independent snapshot of a CPDAG's edges."""
    return (sorted(pdag.directed_edges()), sorted(pdag.undirected_edges()))


def _synthetic_dataset() -> DiscreteDataset:
    """Build a small dataset with genuine dependencies GES can recover.

    A fixed-seed chain/collider structure over 9 binary/ternary variables, so the
    search actually applies inserts and deletes (an all-independent dataset would
    make the comparison vacuous).
    """
    rng = np.random.default_rng(20260725)
    n = 1500
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 3, n)
    c = (a + rng.integers(0, 2, n)) % 2  # c depends on a
    d = (b + rng.integers(0, 3, n)) % 3  # d depends on b
    e = (c + d) % 3  # collider-ish on c, d
    f = (e + rng.integers(0, 2, n)) % 2  # f depends on e
    g = rng.integers(0, 2, n)  # independent
    h = (a + g) % 2  # h depends on a, g
    k = (f + h) % 2  # k depends on f, h
    values = np.column_stack([a, b, c, d, e, f, g, h, k]).astype(np.int32)
    cards = np.array([2, 3, 2, 3, 3, 2, 2, 2, 2], dtype=np.int32)
    names = tuple("abcdefghk")
    cats = tuple(tuple(str(i) for i in range(int(card))) for card in cards)
    return DiscreteDataset.from_arrays(values, cards, names, cats)


def _load_real(net: str) -> DiscreteDataset | None:
    """Load a benchmark net's sample 0, or `None` if the BayesFL tree is absent."""
    csv = BAYESFL / "res" / "networks" / "BBDD" / net / f"{net}.0.csv"
    if not csv.exists():
        return None
    return DiscreteDataset.from_csv(csv)


def _datasets() -> list[tuple[str, DiscreteDataset]]:
    """Return the (label, dataset) pairs to run: synthetic always, real nets if present."""
    out: list[tuple[str, DiscreteDataset]] = [("synthetic", _synthetic_dataset())]
    for net in REAL_NETS:
        ds = _load_real(net)
        if ds is not None:
            out.append((net, ds))
    return out


DATASETS = _datasets()


@pytest.mark.parametrize("label,ds", DATASETS, ids=[d[0] for d in DATASETS])
@pytest.mark.parametrize("max_t_size", [2, None], ids=["cap2", "exhaustive"])
def test_incremental_fes_equals_full_scan(
    label: str, ds: DiscreteDataset, max_t_size: int | None
) -> None:
    """Incremental FES yields the identical CPDAG the full scan does."""
    score = BDeuScore(ds)
    start = PDAG(ds.n_vars)
    full, _ = run_fes(score, start, None, max_t_size, False)
    inc, _ = run_fes_incremental(score, start, None, max_t_size, False)
    assert _canon(inc) == _canon(full)


@pytest.mark.parametrize("label,ds", DATASETS, ids=[d[0] for d in DATASETS])
@pytest.mark.parametrize("h_pool", ["nayx", "bayesfl"])
def test_incremental_bes_equals_full_scan(label: str, ds: DiscreteDataset, h_pool: str) -> None:
    """Incremental BES (from the FES result) yields the identical CPDAG the full scan does."""
    score = BDeuScore(ds)
    fes_start, _ = run_fes(score, PDAG(ds.n_vars), None, 2, False)
    full, _ = run_bes(score, fes_start, None, h_pool)  # type: ignore[arg-type]
    inc, _ = run_bes_incremental(score, fes_start, None, h_pool)  # type: ignore[arg-type]
    assert _canon(inc) == _canon(full)


@pytest.mark.parametrize("label,ds", DATASETS, ids=[d[0] for d in DATASETS])
def test_incremental_ges_fit_equals_full_scan(label: str, ds: DiscreteDataset) -> None:
    """GES(incremental=True).fit equals GES(incremental=False).fit end to end."""
    score = BDeuScore(ds)
    exact = GES(score, incremental=False).fit(ds)
    fast = GES(score, incremental=True).fit(ds)
    assert _canon(fast) == _canon(exact)


@pytest.mark.parametrize("label,ds", DATASETS, ids=[d[0] for d in DATASETS])
def test_incremental_fes_respects_candidate_edges(label: str, ds: DiscreteDataset) -> None:
    """The federated candidate-edge restriction is honoured identically by both FES engines."""
    score = BDeuScore(ds)
    # Restrict to a deterministic subset of ordered pairs (the federated path).
    n = ds.n_vars
    candidates = frozenset(
        (x, y) for x in range(n) for y in range(n) if x != y and (x + y) % 3 == 0
    )
    full, _ = run_fes(score, PDAG(n), candidates, 2, False)
    inc, _ = run_fes_incremental(score, PDAG(n), candidates, 2, False)
    assert _canon(inc) == _canon(full)


@pytest.mark.parametrize("label,ds", DATASETS, ids=[d[0] for d in DATASETS])
def test_incremental_bes_respects_candidate_edges(label: str, ds: DiscreteDataset) -> None:
    """The federated candidate-edge restriction is honoured identically by both BES engines.

    Mirrors ``GESLearner._refine_bes``: run BES from a populated CPDAG, restricted
    to a subset of its own edges' orderings.
    """
    score = BDeuScore(ds)
    fes_cpdag, _ = run_fes(score, PDAG(ds.n_vars), None, 2, False)
    # Candidate deletes: both orderings of a deterministic third of the edges.
    edges = sorted(set(fes_cpdag.directed_edges()) | set(fes_cpdag.undirected_edges()))
    candidates = frozenset(
        pair for i, (u, v) in enumerate(edges) if i % 3 == 0 for pair in ((u, v), (v, u))
    )
    full, _ = run_bes(score, fes_cpdag, candidates, "nayx")
    inc, _ = run_bes_incremental(score, fes_cpdag, candidates, "nayx")
    assert _canon(inc) == _canon(full)
