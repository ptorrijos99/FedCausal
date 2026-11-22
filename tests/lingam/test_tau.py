import numpy as np

from fedcausal.lingam.cumulants import aggregate_and_compute_cumulants, compute_local_raw_stats
from fedcausal.lingam.tau import tau_matrix_3rd, tau_matrix_4th


def _chain(seed: int, n: int = 4000) -> np.ndarray:
    """x0 -> x1 -> x2, Laplace noise. The causal order is 0, 1, 2."""
    rng = np.random.default_rng(seed)
    e = rng.laplace(0.0, 1.0, size=(n, 3))
    x0 = e[:, 0]
    x1 = 0.9 * x0 + e[:, 1]
    x2 = 0.9 * x1 + e[:, 2]
    return np.column_stack([x0, x1, x2])


def test_tau_is_square_with_zero_diagonal():
    cum = aggregate_and_compute_cumulants([compute_local_raw_stats(_chain(0), order=4)])
    tau = tau_matrix_4th(cum["c4"], cum["c13"], cum["c31"], cum["c22"])
    assert tau.shape == (3, 3)
    np.testing.assert_array_equal(np.diagonal(tau), np.zeros(3))
    assert np.all(tau >= 0.0)


def test_row_sum_ranks_the_source_lowest():
    """The exogenous variable has the smallest total tau. This is the whole method."""
    cum = aggregate_and_compute_cumulants([compute_local_raw_stats(_chain(1), order=4)])
    tau = tau_matrix_4th(cum["c4"], cum["c13"], cum["c31"], cum["c22"])
    assert int(np.argmin(tau.sum(axis=1))) == 0


def _skewed_chain(seed: int, n: int = 4000) -> np.ndarray:
    """x0 -> x1 -> x2 with skewed noise, which the third-order tau needs."""
    rng = np.random.default_rng(seed)
    e = rng.exponential(1.0, size=(n, 3)) - 1.0
    x0 = e[:, 0]
    x1 = 0.9 * x0 + e[:, 1]
    x2 = 0.9 * x1 + e[:, 2]
    return np.column_stack([x0, x1, x2])


def test_third_order_also_ranks_the_source_lowest():
    cum = aggregate_and_compute_cumulants([compute_local_raw_stats(_skewed_chain(2), order=3)])
    tau = tau_matrix_3rd(cum["c3"], cum["c12"], cum["c21"])
    assert int(np.argmin(tau.sum(axis=1))) == 0
