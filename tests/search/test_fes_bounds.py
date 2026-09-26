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
    """The default search recovers the chain's skeleton."""
    ds = _chain_dataset()
    cpdag = GES(BDeuScore(ds)).fit(ds)
    dag = cpdag.to_dag()

    skeleton = {frozenset(e) for e in dag.edges()}
    assert skeleton == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_cap_zero_still_finds_the_chain() -> None:
    """``max_t_size=0`` loses nothing on a chain, which never needs a non-empty T."""
    ds = _chain_dataset()
    cpdag = GES(BDeuScore(ds), max_t_size=0).fit(ds)
    dag = cpdag.to_dag()

    skeleton = {frozenset(e) for e in dag.edges()}
    assert skeleton == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_cap_never_beats_unbounded_score() -> None:
    """A cap only removes operators, so it can never score higher than no cap."""
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
