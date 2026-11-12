import numpy as np
import pytest

from fedcausal.lingam.cumulants import (
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
)


def _laplace(seed: int, n: int, p: int) -> np.ndarray:
    return np.random.default_rng(seed).laplace(0.0, 1.0, size=(n, p))


def test_federated_cumulants_match_centralised():
    """Aggregating three clients equals computing on the pooled data."""
    x = _laplace(1, 1200, 5)
    pooled = aggregate_and_compute_cumulants([compute_local_raw_stats(x, order=4)])
    federated = aggregate_and_compute_cumulants(
        [compute_local_raw_stats(chunk, order=4) for chunk in np.array_split(x, 3)]
    )
    for key in ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22"):
        np.testing.assert_allclose(pooled[key], federated[key], rtol=1e-8, atol=1e-8)


def test_sites_with_different_means_merge_exactly():
    x = _laplace(4, 900, 3)
    parts = [x[:300] + 5.0, x[300:] - 2.0]
    pooled = aggregate_and_compute_cumulants([compute_local_raw_stats(np.vstack(parts))])
    federated = aggregate_and_compute_cumulants([compute_local_raw_stats(p) for p in parts])
    for key in ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22"):
        np.testing.assert_allclose(pooled[key], federated[key], rtol=1e-9, atol=1e-9)


def test_cumulants_do_not_depend_on_where_the_data_are_located():
    x = _laplace(5, 3000, 3)
    near = aggregate_and_compute_cumulants(
        [compute_local_raw_stats(c) for c in np.array_split(x, 3)]
    )
    far = aggregate_and_compute_cumulants(
        [compute_local_raw_stats(c + 1e6) for c in np.array_split(x, 3)]
    )
    for key in ("c4", "c13", "c22"):
        np.testing.assert_allclose(far[key], near[key], rtol=1e-4, atol=1e-6)


def test_order_three_omits_fourth_order_keys():
    stats = compute_local_raw_stats(_laplace(2, 100, 3), order=3)
    assert "m31" not in stats
    cumulants = aggregate_and_compute_cumulants([stats])
    assert "c4" not in cumulants


def test_sites_must_send_sums_of_the_same_order_and_enough_rows():
    x = _laplace(6, 100, 3)
    with pytest.raises(ValueError, match="same order"):
        aggregate_and_compute_cumulants(
            [compute_local_raw_stats(x, order=4), compute_local_raw_stats(x, order=3)]
        )
    with pytest.raises(ValueError, match="rows in total"):
        aggregate_and_compute_cumulants([compute_local_raw_stats(x[:3], order=4)])


def test_marginal_cumulants_are_the_k_statistics():
    x = _laplace(3, 2000, 3)
    cumulants = aggregate_and_compute_cumulants([compute_local_raw_stats(x, order=4)])
    n = len(x)
    c = x - x.mean(axis=0)
    m2, m3, m4 = (c**2).sum(0), (c**3).sum(0), (c**4).sum(0)
    k3 = n * m3 / ((n - 1) * (n - 2))
    k4 = (n * (n + 1) * m4 - 3 * (n - 1) * m2**2) / ((n - 1) * (n - 2) * (n - 3))
    np.testing.assert_allclose(cumulants["c3"], k3, rtol=1e-9)
    np.testing.assert_allclose(cumulants["c4"], k4, rtol=1e-9)


def test_empty_client_list_returns_empty():
    assert aggregate_and_compute_cumulants([]) == {}
