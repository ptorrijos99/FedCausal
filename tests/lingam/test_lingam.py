"""The FedRCD family over sites. The solvers are causalbase's, tested there."""

import numpy as np
import pytest
from causalbase.graph import DAG
from causalbase.lingam import RCD, rcd, rcd_exact

from fedcausal import FedRCD
from fedcausal.lingam import RCDResult, cumulants_from_sums, fedrcd, fedrcd_exact, moment_sums

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
    # Three sites, each summing about its own mean, merge exactly into the pooled values.
    federated = cumulants_from_sums([moment_sums(p) for p in (x[:700], x[700:1500], x[1500:])])
    for key in KEYS:
        np.testing.assert_allclose(federated[key], pooled[key], rtol=1e-9, atol=1e-9)
    assert federated["n"] == 2000
    # Still accurate when every site's data sit a million deviations from zero.
    parts = [x[:700], x[700:1500], x[1500:]]
    for key in ("c4", "c13", "c22"):
        np.testing.assert_allclose(
            cumulants_from_sums([moment_sums(p + 1e6) for p in parts])[key],
            federated[key],
            rtol=1e-4,
            atol=1e-6,
        )
    with pytest.raises(ValueError, match="same order"):
        cumulants_from_sums([moment_sums(x), moment_sums(x, order=3)])


def test_every_method_recovers_the_chain_across_sites() -> None:
    stats, skewed = _pooled(_chain(0)), _pooled(_chain(8, skewed=True), order=3)
    for method in ("deflation", "ratio", "sort"):
        assert fedrcd(stats, method=method).causal_order == (0, 1, 2, 3)  # type: ignore[arg-type]
        assert fedrcd(skewed, method=method).causal_order == (0, 1, 2, 3)  # type: ignore[arg-type]
    sites = np.array_split(_chain(1), 3)
    assert fedrcd_exact(sites).causal_order == (0, 1, 2, 3)
    result = fedrcd(stats)
    assert isinstance(result, RCDResult)
    assert {(0, 1), (1, 2), (2, 3)} <= set(result.to_dag(threshold=0.3).edges())


def test_the_federated_names_run_causalbases_solvers() -> None:
    """Over sites, the pooled sums give what causalbase learns from the pooled table."""
    x = _chain(2)
    sites = np.array_split(x, 3)
    for method in ("deflation", "exact", "ratio", "sort"):
        model = FedRCD(method=method, pruning=None).fit(list(sites))  # type: ignore[arg-type]
        alone = RCD(method=method, pruning=None).fit(x)  # type: ignore[arg-type]
        assert model.causal_order_ == alone.causal_order_
        np.testing.assert_allclose(model.coef_, alone.coef_, rtol=1e-8, atol=1e-10)
    stats = _pooled(x)
    for federated, central in [
        (fedrcd(stats, method="ratio"), rcd(stats, method="ratio")),
        (fedrcd_exact([x]), rcd_exact(x)),
    ]:
        assert federated.causal_order == central.causal_order
        np.testing.assert_allclose(federated.coef, central.coef, rtol=1e-12, atol=1e-15)


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
    with pytest.raises(ValueError, match="pruning"):
        fedrcd_exact(np.array_split(_chain(9), 2), pruning="lasso")  # type: ignore[arg-type]
    without_n = dict(stats)
    del without_n["n"]  # a silent default would mis-scale the lasso penalty
    with pytest.raises(KeyError, match="missing 'n'"):
        fedrcd(without_n)
    assert isinstance(fedrcd(stats).to_dag(), DAG)
