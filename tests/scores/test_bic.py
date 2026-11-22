"""The BIC scores against direct maximum-likelihood fits."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedcausal.data import ContinuousDataset, DiscreteDataset
from fedcausal.graph import DAG
from fedcausal.scores import BICScore, GaussianBICScore, log_likelihood, make_score


def _discrete(n_rows: int, n_vars: int, seed: int) -> DiscreteDataset:
    rng = np.random.default_rng(seed)
    cards = rng.integers(2, 5, size=n_vars)
    values = np.column_stack([rng.integers(0, c, size=n_rows) for c in cards])
    return DiscreteDataset(
        values, cards, [f"x{i}" for i in range(n_vars)], [[str(k) for k in range(c)] for c in cards]
    )


def _continuous(n_rows: int, seed: int) -> ContinuousDataset:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n_rows)
    x1 = 0.8 * x0 + rng.normal(size=n_rows)
    x2 = -0.5 * x1 + 0.3 * x0 + rng.normal(size=n_rows)
    x3 = rng.normal(size=n_rows)
    return ContinuousDataset(np.column_stack([x0, x1, x2, x3]), ["x0", "x1", "x2", "x3"])


@pytest.mark.parametrize("parents", [(), (1,), (1, 3)])
def test_discrete_bic_is_the_maximised_likelihood_less_the_penalty(parents) -> None:
    data = _discrete(500, 5, seed=0)
    r = int(data.cardinalities[0])
    q = math.prod(int(data.cardinalities[p]) for p in parents)
    counts: dict[tuple[int, ...], np.ndarray] = {}
    for row in data.values:
        counts.setdefault(tuple(row[list(parents)]), np.zeros(r))[row[0]] += 1
    lik = sum(float(c @ np.log(np.where(c > 0, c / c.sum(), 1.0))) for c in counts.values())
    expected = lik - math.log(500) / 2 * q * (r - 1)
    assert BICScore(data).local_score(0, parents) == pytest.approx(expected, rel=1e-10)


def test_gaussian_bic_is_the_regression_likelihood_less_the_penalty() -> None:
    data = _continuous(2000, seed=1)
    x = data.values
    design = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1]])
    residual = x[:, 2] - design @ np.linalg.lstsq(design, x[:, 2], rcond=None)[0]
    s2 = residual @ residual / len(x)
    lik = -len(x) / 2 * (math.log(2 * math.pi * s2) + 1)
    expected = lik - math.log(len(x)) / 2 * 4
    assert GaussianBICScore(data).local_score(2, {0, 1}) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "make", [lambda: BICScore(_discrete(400, 6, 2)), lambda: GaussianBICScore(_continuous(400, 3))]
)
def test_extensions_scored_together_equal_one_at_a_time(make) -> None:
    together = make().local_scores_with(0, frozenset({1}), [2, 3])
    alone = [make().local_score(0, {1, x}) for x in (2, 3)]
    np.testing.assert_allclose(together, alone, rtol=1e-10)


def test_penalty_discount_and_prior_only_ever_penalise_more() -> None:
    data = _continuous(300, seed=4)
    plain = GaussianBICScore(data)
    harsher = GaussianBICScore(data, penalty_discount=2.0)
    prior = GaussianBICScore(data, expected_parents=1.0)
    for score in (harsher, prior):
        assert score.local_score(2, {0, 1}) < plain.local_score(2, {0, 1})


def test_make_score_names_the_three_and_nothing_else() -> None:
    assert isinstance(make_score("bic", _discrete(50, 3, 5)), BICScore)
    assert isinstance(make_score("gaussian_bic", _continuous(50, 5)), GaussianBICScore)
    with pytest.raises(ValueError, match="score must be one of"):
        make_score("aic", _discrete(50, 3, 5))  # type: ignore[arg-type]


def test_held_out_likelihood_prefers_the_true_structure() -> None:
    train, test = _continuous(3000, seed=6), _continuous(1000, seed=7)
    truth = DAG.from_edges(4, [(0, 1), (0, 2), (1, 2)])
    assert log_likelihood(truth, train, test) > log_likelihood(DAG(4), train, test)


def test_discrete_held_out_likelihood_is_the_same_across_an_equivalence_class() -> None:
    data = _discrete(600, 3, seed=8)
    train, test = data.subset(np.arange(400)), data.subset(np.arange(400, 600))
    forward = DAG.from_edges(3, [(0, 1), (1, 2)])
    backward = DAG.from_edges(3, [(2, 1), (1, 0)])
    assert log_likelihood(forward, train, test) == pytest.approx(
        log_likelihood(backward, train, test), rel=1e-10
    )
