"""Rounding in a score difference must not count as an improvement."""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.scores import BDeuScore
from fedcausal.search import GES
from fedcausal.search.gesops import _denoise


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


def test_tolerance_scales_with_the_scores() -> None:
    """The same absolute delta is noise against a large score, signal against a small one."""
    assert _denoise(1e-9, 1e5, 1e5) == 0.0
    assert _denoise(1e-9, 1.0, 1.0) == 1e-9


@pytest.mark.parametrize("structure_prior", [0.0, 1.0])
def test_the_search_leaves_a_constant_isolated(structure_prior: float) -> None:
    data = _dataset()
    learned = GES(BDeuScore(data, structure_prior=structure_prior)).fit(data).to_dag()

    constant = 3
    assert not [(u, v) for u, v in learned.edges() if constant in (u, v)]
    assert learned.num_edges > 0  # the rest of the graph was still found
