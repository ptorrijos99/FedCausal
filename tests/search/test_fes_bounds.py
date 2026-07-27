"""Tests for FES's two optional bounds on the `T` enumeration.

``fedcausal``'s FES enumerates every ``T`` subset of ``T0``, which is
exponential in neighbourhood width and is why it can be 39x slower than
BayesFL on hailfinder and fail to finish at all on some pathfinder samples.
BayesFL bounds the same loop with two *separate* mechanisms:

``max_t_size``
    ``FESThread.java:293`` caps forward selection at ``|T| <= 2``.

``empty_t_gate``
    ``FESThread.java:254,259`` scores each candidate pair at ``T = {}`` first
    and abandons the pair entirely unless that passes, so a pair rejected at
    ``T = {}`` is never revisited with a non-empty ``T``.

They are exposed separately, and tested separately here, because the golden
diagnosis in ``tests/golden/test_ges_golden.py`` established that it is the
*pre-gate*, not the cap, that makes Java converge to a worse optimum on
``child``. Bundling them as a single "java mode" would make that
attributable to neither.

Both default to off, so the default search stays Chickering-exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search import GES
from fedcausal.search._candidates import subsets
from fedcausal.search.ges import run_fes


def testsubsets_unbounded_yields_every_subset() -> None:
    """Without a cap, every subset is emitted, smallest first."""
    got = list(subsets([1, 2, 3]))

    assert len(got) == 8
    assert got[0] == frozenset()
    assert [len(s) for s in got] == [0, 1, 1, 1, 2, 2, 2, 3]


@pytest.mark.parametrize(("cap", "expected"), [(0, 1), (1, 4), (2, 7), (3, 8), (99, 8)])
def testsubsets_cap_limits_size(cap: int, expected: int) -> None:
    """A cap truncates the size classes emitted, keeping the same order."""
    got = list(subsets([1, 2, 3], max_size=cap))

    assert len(got) == expected
    assert all(len(s) <= cap for s in got)
    assert got == list(subsets([1, 2, 3]))[:expected]


def testsubsets_cap_of_two_matches_bayesfl() -> None:
    """``max_size=2`` is BayesFL's ``|T| <= 2`` (FESThread.java:293)."""
    got = list(subsets([1, 2, 3, 4], max_size=2))

    assert max(len(s) for s in got) == 2
    assert frozenset({1, 2}) in got
    assert frozenset({1, 2, 3}) not in got


def _chain_dataset(seed: int = 0, n_rows: int = 2000) -> DiscreteDataset:
    """Return data from a 4-node chain ``0 -> 1 -> 2 -> 3`` with strong links."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    noise = rng.random((3, n_rows)) < 0.05
    x1 = x0 ^ noise[0]
    x2 = x1 ^ noise[1]
    x3 = x2 ^ noise[2]
    values = np.stack([x0, x1, x2, x3], axis=1).astype(np.int32)
    return DiscreteDataset(
        values,
        np.array([2, 2, 2, 2], dtype=np.int32),
        ("a", "b", "c", "d"),
        (("0", "1"),) * 4,
    )


def test_default_cap_recovers_the_chain() -> None:
    """The default search (now ``max_t_size=2``) recovers the chain's skeleton.

    The default cap is 2, not unbounded; a chain never needs ``|T| > 0``, so
    the default recovers it exactly, as would ``max_t_size=None``.
    """
    ds = _chain_dataset()
    cpdag = GES(BDeuScore(ds)).fit(ds)
    dag = cpdag.to_dag()

    skeleton = {frozenset(e) for e in dag.edges()}
    assert skeleton == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_cap_zero_still_finds_the_chain() -> None:
    """``max_t_size=0`` (only ``T = {}``) suffices on a chain.

    A chain never needs a non-empty ``T``: each Insert's ``NaYX`` is empty, so
    the harshest possible cap loses nothing here. This pins that the cap
    changes *which* operators are considered, not whether the search runs.
    """
    ds = _chain_dataset()
    cpdag = GES(BDeuScore(ds), max_t_size=0).fit(ds)
    dag = cpdag.to_dag()

    skeleton = {frozenset(e) for e in dag.edges()}
    assert skeleton == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_cap_never_beats_unbounded_score() -> None:
    """Capping ``T`` cannot improve on the unbounded search's optimum.

    The capped operator set is a strict subset of the unbounded one, so its
    reachable optima are a subset too. A cap scoring *better* would mean the
    unbounded scan is missing an operator it should have considered.
    """
    ds = _chain_dataset(seed=7)
    score = BDeuScore(ds)

    unbounded = score.score_dag(GES(score, max_t_size=None).fit(ds).to_dag())
    for cap in (0, 1, 2):
        capped = score.score_dag(GES(score, max_t_size=cap).fit(ds).to_dag())
        assert capped <= unbounded + 1e-9, f"cap={cap} scored above the unbounded search"


def test_empty_t_gate_is_independent_of_the_cap() -> None:
    """The gate and the cap are separate switches, settable independently."""
    ds = _chain_dataset()
    score = BDeuScore(ds)

    default = GES(score)
    gated = GES(score, max_t_size=None, empty_t_gate=True)
    capped = GES(score, max_t_size=2)
    both = GES(score, max_t_size=2, empty_t_gate=True)

    # The default is BayesFL's cap with the gate off, not the exhaustive scan.
    assert default.max_t_size == 2
    assert default.empty_t_gate is False
    assert gated.max_t_size is None
    assert gated.empty_t_gate is True
    assert capped.max_t_size == 2
    assert capped.empty_t_gate is False
    assert both.max_t_size == 2
    assert both.empty_t_gate is True


def test_run_fes_accepts_the_bounds() -> None:
    """``run_fes`` takes the bounds too, for the federated client refinement."""
    ds = _chain_dataset()
    score = BDeuScore(ds)

    bounded, changed = run_fes(score, PDAG(ds.n_vars), max_t_size=1, empty_t_gate=True)

    assert changed
    assert bounded.num_nodes == ds.n_vars


def test_negative_cap_is_rejected() -> None:
    """A negative cap is a caller error, not a synonym for 'unbounded'."""
    ds = _chain_dataset()
    with pytest.raises(ValueError, match="max_t_size"):
        GES(BDeuScore(ds), max_t_size=-1)
