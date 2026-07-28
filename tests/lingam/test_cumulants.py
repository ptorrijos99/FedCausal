import numpy as np
import pytest

from fedcausal.lingam.cumulants import (
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
    unbiased_cumulant3,
    unbiased_cumulant4,
)


def _laplace(seed: int, n: int, p: int) -> np.ndarray:
    return np.random.default_rng(seed).laplace(0.0, 1.0, size=(n, p))


def test_raw_stats_are_additive_across_clients():
    """The property the whole federation rests on: sums split across clients."""
    x = _laplace(0, 900, 4)
    whole = compute_local_raw_stats(x, order=4)
    parts = [compute_local_raw_stats(chunk, order=4) for chunk in np.array_split(x, 3)]

    assert whole["n"] == sum(part["n"] for part in parts)
    for key in (
        "sum_x",
        "sum_xx",
        "sum_x3",
        "sum_x1y2",
        "sum_x2y1",
        "sum_x4",
        "sum_x1y3",
        "sum_x3y1",
        "sum_x2y2",
    ):
        np.testing.assert_allclose(
            whole[key], sum(part[key] for part in parts), rtol=1e-10, atol=1e-10
        )


def test_federated_cumulants_match_centralised():
    """Aggregating three clients equals computing on the pooled data."""
    x = _laplace(1, 1200, 5)
    pooled = aggregate_and_compute_cumulants([compute_local_raw_stats(x, order=4)])
    federated = aggregate_and_compute_cumulants(
        [compute_local_raw_stats(chunk, order=4) for chunk in np.array_split(x, 3)]
    )
    for key in ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22"):
        np.testing.assert_allclose(pooled[key], federated[key], rtol=1e-8, atol=1e-8)


def test_order_three_omits_fourth_order_keys():
    stats = compute_local_raw_stats(_laplace(2, 100, 3), order=3)
    assert "sum_x4" not in stats
    cumulants = aggregate_and_compute_cumulants([stats])
    assert "c4" not in cumulants


def test_marginal_cumulants_match_the_matrix_versions():
    x = _laplace(3, 2000, 3)
    cumulants = aggregate_and_compute_cumulants([compute_local_raw_stats(x, order=4)])
    for j in range(3):
        assert unbiased_cumulant3(x[:, j]) == pytest.approx(cumulants["c3"][j], rel=1e-9)
        assert unbiased_cumulant4(x[:, j]) == pytest.approx(cumulants["c4"][j], rel=1e-9)


def test_empty_client_list_returns_empty():
    assert aggregate_and_compute_cumulants([]) == {}
