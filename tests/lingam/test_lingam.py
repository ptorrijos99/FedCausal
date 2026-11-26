"""The FedRCD family: cumulants from moment sums, the tau discriminator, and the solvers."""

import numpy as np
import pytest
from causalbase.graph import DAG

from fedcausal.lingam import RCDResult, cumulants_from_sums, fedrcd, fedrcd_exact, moment_sums
from fedcausal.lingam.solvers import estimate_b_from_cov, estimate_b_lasso_from_cov
from fedcausal.lingam.tau import tau_matrix_3rd, tau_matrix_4th

KEYS = ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22")


def _chain(seed: int, n: int = 6000, skewed: bool = False) -> np.ndarray:
    """x0 -> x1 -> x2 -> x3 with Laplace (or skewed) noise. The order is 0, 1, 2, 3."""
    rng = np.random.default_rng(seed)
    e = rng.exponential(1.0, size=(n, 4)) - 1.0 if skewed else rng.laplace(size=(n, 4))
    x = np.zeros((n, 4))
    x[:, 0] = e[:, 0]
    for j in range(1, 4):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return x


def _pooled(x: np.ndarray, order: int = 4, n_sites: int = 3) -> dict:
    return cumulants_from_sums([moment_sums(c, order) for c in np.array_split(x, n_sites)])


def test_sites_sums_give_the_pooled_k_statistics() -> None:
    x = np.random.default_rng(3).laplace(size=(2000, 3))
    pooled = cumulants_from_sums([moment_sums(x)])
    # Sites with their own means, far from zero, merge exactly into the pooled values.
    parts = [x[:700] + 5.0, x[700:1500] - 2.0, x[1500:] + 1e6]
    federated = cumulants_from_sums([moment_sums(p) for p in parts])
    shifted = cumulants_from_sums([moment_sums(np.vstack([x[:700], x[700:1500], x[1500:]]))])
    for key in KEYS:
        np.testing.assert_allclose(shifted[key], pooled[key], rtol=1e-9, atol=1e-9)
    for key in ("c4", "c13", "c22"):  # still accurate a million deviations away
        np.testing.assert_allclose(
            cumulants_from_sums([moment_sums(p - np.mean(p, 0) + 1e6) for p in parts])[key],
            cumulants_from_sums([moment_sums(p - np.mean(p, 0)) for p in parts])[key],
            rtol=1e-4,
            atol=1e-6,
        )
    assert federated["n"] == 2000
    n, c = len(x), x - x.mean(axis=0)
    m2, m3, m4 = (c**2).sum(0), (c**3).sum(0), (c**4).sum(0)
    k3 = n * m3 / ((n - 1) * (n - 2))
    k4 = (n * (n + 1) * m4 - 3 * (n - 1) * m2**2) / ((n - 1) * (n - 2) * (n - 3))
    np.testing.assert_allclose(pooled["c3"], k3, rtol=1e-9)
    np.testing.assert_allclose(pooled["c4"], k4, rtol=1e-9)


def test_sums_of_different_orders_or_too_few_rows_are_refused() -> None:
    x = np.random.default_rng(6).laplace(size=(100, 3))
    third = moment_sums(x, order=3)
    assert "m31" not in third
    assert "c4" not in cumulants_from_sums([third])
    assert cumulants_from_sums([]) == {}
    with pytest.raises(ValueError, match="same order"):
        cumulants_from_sums([moment_sums(x), third])
    with pytest.raises(ValueError, match="rows in total"):
        cumulants_from_sums([moment_sums(x[:3])])


def test_tau_ranks_the_source_lowest() -> None:
    """The exogenous variable has the smallest total tau. This is the whole method."""
    cum = _pooled(_chain(1), n_sites=1)
    tau = tau_matrix_4th(cum["c4"], cum["c13"], cum["c31"], cum["c22"])
    assert np.all(tau >= 0) and not np.diagonal(tau).any()
    assert int(np.argmin(tau.sum(axis=1))) == 0
    cum = _pooled(_chain(2, skewed=True), order=3, n_sites=1)
    assert int(np.argmin(tau_matrix_3rd(cum["c3"], cum["c12"], cum["c21"]).sum(axis=1))) == 0


def test_every_method_recovers_the_chain() -> None:
    stats, skewed = _pooled(_chain(0)), _pooled(_chain(8, skewed=True), order=3)
    for method in ("deflation", "ratio", "sort"):
        assert fedrcd(stats, method=method).causal_order == (0, 1, 2, 3)  # type: ignore[arg-type]
        assert fedrcd(skewed, method=method).causal_order == (0, 1, 2, 3)  # type: ignore[arg-type]
    sites = np.array_split(_chain(1), 3)
    assert fedrcd_exact(sites).causal_order == (0, 1, 2, 3)
    assert fedrcd_exact([np.vstack(sites)]).causal_order == (0, 1, 2, 3)
    result = fedrcd(stats)
    assert isinstance(result, RCDResult)
    assert {(0, 1), (1, 2), (2, 3)} <= set(result.to_dag(threshold=0.3).edges())
    assert result.to_dag(threshold=10.0).edges() == []
    with pytest.raises(AttributeError):
        result.causal_order = (1, 0, 2, 3)  # type: ignore[misc]


def test_ratio_deflation_removes_the_source_with_its_coefficient() -> None:
    """Cum(s, s, j) / Cum(s, s, s) is the coefficient of s in j; Cum(j, j, s) gives its square."""
    from fedcausal.lingam.rcd import _deflated_order

    e = np.random.default_rng(7).exponential(1.0, size=(200_000, 2)) - 1.0
    stats = cumulants_from_sums(
        [moment_sums(np.column_stack([e[:, 0], 1.5 * e[:, 0] + e[:, 1]]), 3)]
    )
    cur = {k: np.array(stats[k]) for k in ("cov", "c3", "c12", "c21")}
    assert _deflated_order(cur, 3, "ratio") == [0, 1]
    assert cur["cov"][1, 1] == pytest.approx(1.0, abs=0.05)


def test_coefficients_on_a_known_order() -> None:
    rng = np.random.default_rng(0)
    e = rng.laplace(size=(20000, 3))
    x1 = 0.8 * e[:, 0] + e[:, 1]
    x = np.column_stack([e[:, 0], x1, -0.6 * x1 + e[:, 2]])
    cov = np.cov(x, rowvar=False)
    ols = estimate_b_from_cov([0, 1, 2], cov)
    np.testing.assert_allclose(ols, [[0, 0, 0], [0.8, 0, 0], [0, -0.6, 0]], atol=0.05)
    # No variable takes a parent that comes after it in the order.
    reordered = estimate_b_from_cov([2, 0, 1], cov)
    assert reordered[2].sum() == 0 and reordered[0, 1] == 0
    # The lasso keeps the true edges and shrinks: a "lasso" returning OLS would fail.
    lasso = estimate_b_lasso_from_cov([0, 1, 2], cov, n_samples=len(x))
    assert lasso[1, 0] != 0 and lasso[2, 1] != 0
    assert np.abs(lasso).sum() < np.abs(ols).sum()


def test_solvers_refuse_bad_settings() -> None:
    stats = _pooled(_chain(9))
    for kwargs, match in [
        ({"method": "bogus"}, "method"),
        ({"order": 5}, "order"),
        ({"pruning": "lasso"}, "pruning"),
        ({"lambda_selection": "aic"}, "lambda_selection"),
    ]:
        with pytest.raises(ValueError, match=match):
            fedrcd(stats, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fourth-order"):
        fedrcd(_pooled(_chain(9), order=3), order=4)
    without_n = dict(stats)
    del without_n["n"]  # a silent default would mis-scale the lasso penalty
    with pytest.raises(KeyError, match="missing 'n'"):
        fedrcd(without_n)
    with pytest.raises(ValueError, match="lambda_selection"):
        estimate_b_lasso_from_cov([0, 1], np.eye(2), n_samples=10, lambda_selection="x")
    assert isinstance(fedrcd(stats).to_dag(), DAG)
