import numpy as np
import pytest

from fedcausal.lingam.solvers import estimate_b_from_cov, estimate_b_lasso_from_cov


def _sem(seed: int, n: int = 20000):
    """x0 -> x1 -> x2 with known coefficients; returns (data, true B)."""
    rng = np.random.default_rng(seed)
    b_true = np.zeros((3, 3))
    b_true[1, 0] = 0.8
    b_true[2, 1] = -0.6
    e = rng.laplace(0.0, 1.0, size=(n, 3))
    x0 = e[:, 0]
    x1 = b_true[1, 0] * x0 + e[:, 1]
    x2 = b_true[2, 1] * x1 + e[:, 2]
    return np.column_stack([x0, x1, x2]), b_true


def test_ols_recovers_the_coefficients():
    x, b_true = _sem(0)
    cov = np.cov(x, rowvar=False)
    b_est = estimate_b_from_cov([0, 1, 2], cov)
    np.testing.assert_allclose(b_est, b_true, atol=0.05)


def test_result_is_lower_triangular_in_the_given_order():
    """No variable may take a parent that comes after it in the causal order."""
    x, _ = _sem(1)
    cov = np.cov(x, rowvar=False)
    order = [2, 0, 1]
    b_est = estimate_b_from_cov(order, cov)
    position = {var: i for i, var in enumerate(order)}
    for child in range(3):
        for parent in range(3):
            if position[parent] >= position[child]:
                assert b_est[child, parent] == 0.0


def test_lasso_shrinks_towards_zero_but_keeps_the_true_edges():
    """Shrinkage is a Lasso-vs-OLS statement, not a Lasso-vs-ground-truth one.

    The ground truth is not what Lasso is shrinking relative to -- the OLS
    estimate on the same covariance is. Comparing against `b_true` instead
    would pass even when Lasso's L1 norm is *larger* than OLS's, which is the
    opposite of shrinkage.
    """
    x, _ = _sem(2)
    cov = np.cov(x, rowvar=False)
    b_ols = estimate_b_from_cov([0, 1, 2], cov)
    b_lasso = estimate_b_lasso_from_cov([0, 1, 2], cov, n_samples=x.shape[0])
    assert b_lasso[1, 0] != 0.0
    assert b_lasso[2, 1] != 0.0
    assert np.abs(b_lasso).sum() <= np.abs(b_ols).sum()


def test_lasso_rejects_an_unknown_lambda_type():
    cov = np.eye(3)
    with pytest.raises(ValueError, match="lmbda_type"):
        estimate_b_lasso_from_cov([0, 1, 2], cov, n_samples=100, lmbda_type="nonsense")
