"""The BDeu score against an independent count-and-lgamma implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG
from fedcausal.scores import BDeuScore


def _dataset(n_rows: int, n_vars: int, levels: int, seed: int) -> DiscreteDataset:
    values = np.random.default_rng(seed).integers(0, levels, size=(n_rows, n_vars))
    labels = [str(k) for k in range(levels)]
    return DiscreteDataset(
        values.astype(np.int32),
        [levels] * n_vars,
        [f"x{i}" for i in range(n_vars)],
        [labels] * n_vars,
    )


def _reference(data: DiscreteDataset, node: int, parents: tuple[int, ...], ess: float) -> float:
    r = int(data.cardinalities[node])
    q = math.prod(int(data.cardinalities[p]) for p in parents)
    counts: dict[tuple[int, ...], np.ndarray] = {}
    for row in data.values:
        counts.setdefault(tuple(row[list(parents)]), np.zeros(r))[row[node]] += 1
    a_j, a_jk = ess / q, ess / (q * r)
    total = 0.0
    for n_k in counts.values():
        total += math.lgamma(a_j) - math.lgamma(a_j + n_k.sum())
        total += sum(math.lgamma(a_jk + c) - math.lgamma(a_jk) for c in n_k)
    return total


@pytest.mark.parametrize("parents", [(), (1,), (1, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("ess", [1.0, 10.0])
def test_matches_the_reference(parents: tuple[int, ...], ess: float) -> None:
    data = _dataset(300, 5, 3, seed=1)
    assert BDeuScore(data, ess, expected_parents=None).local_score(0, parents) == pytest.approx(
        _reference(data, 0, parents, ess), rel=1e-10
    )


@pytest.mark.parametrize("n_parents", [12, 45])
def test_matches_the_reference_when_parent_configurations_outnumber_rows(n_parents: int) -> None:
    """3**45 configurations overflow any direct encoding; they must be relabelled."""
    data = _dataset(200, 50, 3, seed=2)
    parents = tuple(range(1, n_parents + 1))
    assert BDeuScore(data, expected_parents=None).local_score(0, parents) == pytest.approx(
        _reference(data, 0, parents, 1.0), rel=1e-10
    )


def test_scoring_many_extensions_at_once_matches_one_at_a_time() -> None:
    data = _dataset(400, 8, 4, seed=3)
    together = BDeuScore(data).local_scores_with(0, frozenset({1, 2}), [3, 4, 5, 6, 7])
    alone = [BDeuScore(data).local_score(0, {1, 2, x}) for x in (3, 4, 5, 6, 7)]
    assert together == pytest.approx(alone, rel=1e-12)


def test_the_structure_prior_charges_a_fixed_amount_per_parent() -> None:
    """Six variables and one expected parent: p = 1/5 per candidate parent."""
    data = _dataset(300, 6, 2, seed=4)
    plain, sparse = BDeuScore(data, expected_parents=None), BDeuScore(data)
    p = 1.0 / 5
    per_parent = math.log(p) - math.log(1 - p)
    for parents in [(), (1,), (1, 2, 3)]:
        assert sparse.local_score(0, parents) - plain.local_score(0, parents) == pytest.approx(
            len(parents) * per_parent
        )


def test_parent_order_does_not_matter_and_dags_sum_their_families() -> None:
    data = _dataset(300, 4, 3, seed=5)
    score = BDeuScore(data)
    assert score.local_score(3, (2, 0, 1)) == score.local_score(3, (0, 1, 2))
    dag = DAG.from_edges(4, [(0, 1), (1, 3), (2, 3)])
    expected = sum(score.local_score(v, dag.parents(v)) for v in range(4))
    assert score.score_dag(dag) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"ess": 0}, "ess"), ({"expected_parents": 0}, "expected_parents")]
)
def test_invalid_priors_are_rejected(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BDeuScore(_dataset(10, 4, 2, seed=6), **kwargs)
