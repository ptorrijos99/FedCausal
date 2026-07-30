"""Rounding residue must not look like a score improvement.

Both operator evaluations subtract two local scores of similar magnitude, so the
difference keeps far fewer significant digits than its inputs. On a large
network the residue reaches tens of ulps, which is above a fixed 1e-12
threshold, and the search was inserting edges that existed only in the
rounding. The tolerance is relative now; these pin that.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search import GES
from fedcausal.search.gesops import _denoise, insert_eval, na_yx


def _dataset(n_rows: int = 4000, seed: int = 0) -> DiscreteDataset:
    """Three dependent binary variables plus one that never varies."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n_rows)
    b = np.where(rng.random(n_rows) < 0.9, a, 1 - a)
    c = np.where(rng.random(n_rows) < 0.9, b, 1 - b)
    values = np.stack([a, b, c, np.zeros(n_rows, dtype=np.int32)], axis=1).astype(np.int32)
    return DiscreteDataset(
        values,
        np.array([2, 2, 2, 1], np.int32),
        ["A", "B", "C", "K"],
        [["0", "1"], ["0", "1"], ["0", "1"], ["0"]],
    )


@pytest.mark.parametrize(
    ("delta", "magnitude", "expected"),
    [
        (7.3e-12, 6944.0, 0.0),  # the residue actually observed on `link`
        (3.6e-12, 3469.0, 0.0),
        (1e-3, 6944.0, 1e-3),  # small but far above the noise floor
        (-7.3e-12, 6944.0, 0.0),
        (5.0, 6944.0, 5.0),
    ],
)
def test_denoise_drops_only_rounding(delta: float, magnitude: float, expected: float) -> None:
    assert _denoise(delta, magnitude, magnitude - delta) == expected


def test_tolerance_scales_with_the_scores() -> None:
    """The same absolute delta is noise against a large score, signal against a small one."""
    assert _denoise(1e-9, 1e5, 1e5) == 0.0
    assert _denoise(1e-9, 1.0, 1.0) == 1e-9


@pytest.mark.parametrize("structure_prior", [0.0, 1.0])
def test_a_constant_variable_earns_no_insert(structure_prior: float) -> None:
    """A variable with one level carries no information, so no edge to it can pay.

    Without a structure prior the delta is exactly zero, which is what the
    relative tolerance exists to keep it at. With one it is negative, since the
    extra parent is penalised and buys nothing. Neither is an improvement.
    """
    data = _dataset()
    score = BDeuScore(data, structure_prior=structure_prior)
    pdag = PDAG(data.n_vars)
    constant = 3

    for other in range(3):
        empty: frozenset[int] = frozenset()
        assert insert_eval(score, pdag, other, constant, empty, na_yx(pdag, other, constant)) <= 0.0
        assert insert_eval(score, pdag, constant, other, empty, na_yx(pdag, constant, other)) <= 0.0


@pytest.mark.parametrize("structure_prior", [0.0, 1.0])
def test_the_search_leaves_a_constant_isolated(structure_prior: float) -> None:
    data = _dataset()
    learned = GES(BDeuScore(data, structure_prior=structure_prior)).fit(data).to_dag()

    constant = 3
    assert not [(u, v) for u, v in learned.edges() if constant in (u, v)]
    assert learned.num_edges > 0  # the rest of the graph was still found
