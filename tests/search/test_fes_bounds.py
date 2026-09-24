"""FES's two optional bounds on the ``T`` enumeration.

``max_t_size`` caps the size of ``T``; ``empty_t_gate`` abandons a candidate
pair unless it improves with ``T = {}``. Both come from the reference Java
implementation and are separate switches, because only the gate explains its
result on ``child`` (see ``tests/regression/test_ges_reference.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.scores import BDeuScore
from fedcausal.search import GES


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


def test_negative_cap_is_rejected() -> None:
    """A negative cap is a caller error, not a synonym for 'unbounded'."""
    ds = _chain_dataset()
    with pytest.raises(ValueError, match="max_t_size"):
        GES(BDeuScore(ds), max_t_size=-1)
