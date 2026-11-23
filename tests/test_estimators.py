"""The three estimators' shared contract: parameters, input checks, fitted attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcausal import GES, FedGES, FedRCD
from fedcausal.data import ContinuousDataset
from fedcausal.exceptions import FedCausalError, NotFittedError
from fedcausal.lingam import aggregate_and_compute_cumulants, compute_local_raw_stats, fedrcd


def _linear_sites(n_sites: int = 3, n_rows: int = 2000, seed: int = 0) -> list[np.ndarray]:
    """x0 -> x1 -> x2 with uniform (non-Gaussian) noise, split over `n_sites`."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(-1, 1, size=(n_sites * n_rows, 3))
    x0 = e[:, 0]
    x1 = 0.9 * x0 + e[:, 1]
    x2 = -0.8 * x1 + e[:, 2]
    x = np.column_stack([x0, x1, x2])
    return np.array_split(x, n_sites)


def _discrete(n_rows: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 3, size=(n_rows, 4))


@pytest.mark.parametrize("cls", [GES, FedGES, FedRCD])
def test_params_round_trip_and_repr_shows_only_changes(cls: type) -> None:
    model = cls()
    assert repr(model) == f"{cls.__name__}()"
    params = model.get_params()
    assert model.set_params(**params) is model
    assert model.get_params() == params
    with pytest.raises(ValueError, match="no parameter"):
        model.set_params(not_a_parameter=1)


def test_sklearn_clone_keeps_the_parameters() -> None:
    sklearn_base = pytest.importorskip("sklearn.base")
    for model in (GES(ess=5.0), FedGES(n_rounds=3), FedRCD(order=3)):
        assert sklearn_base.clone(model).get_params() == model.get_params()


@pytest.mark.parametrize("cls", [GES, FedGES, FedRCD])
def test_results_before_fit_raise_not_fitted(cls: type) -> None:
    with pytest.raises(NotFittedError):
        _ = cls().adjacency_
    assert issubclass(NotFittedError, FedCausalError)
    assert not hasattr(cls(), "adjacency_")


def test_fedrcd_recovers_a_chain_and_names_its_variables() -> None:
    sites = [pd.DataFrame(x, columns=["a", "b", "c"]) for x in _linear_sites()]
    model = FedRCD().fit(sites)
    assert model.causal_order_ == ["a", "b", "c"]
    assert model.n_features_in_ == 3
    assert list(model.feature_names_in_) == ["a", "b", "c"]
    assert sorted(model.dag_.edges()) == [(0, 1), (1, 2)]
    assert model.adjacency_.loc["a", "b"] == 1
    assert model.adjacency_.loc["b", "a"] == 0
    assert model.coef_.shape == (3, 3)


def test_fedrcd_is_the_solver_over_the_pooled_statistics() -> None:
    sites = _linear_sites()
    model = FedRCD(pruning=None).fit(sites)
    stats = aggregate_and_compute_cumulants([compute_local_raw_stats(x) for x in sites])
    expected = fedrcd(stats)
    assert tuple(int(v[1:]) for v in model.causal_order_) == expected.causal_order
    np.testing.assert_array_equal(model.coef_, expected.b)


@pytest.mark.parametrize("method", ["deflation", "exact", "ratio", "sort"])
def test_fedrcd_methods_agree_on_an_easy_chain(method: str) -> None:
    model = FedRCD(method=method).fit(_linear_sites())  # type: ignore[arg-type]
    assert model.causal_order_ == ["x0", "x1", "x2"]


@pytest.mark.parametrize(
    "params",
    [{"method": "nope"}, {"order": 5}, {"pruning": "lasso"}, {"lambda_selection": "aic"}],
)
def test_fedrcd_rejects_unknown_settings(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FedRCD(**params).fit(_linear_sites())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "single",
    [
        np.zeros((10, 3)),
        pd.DataFrame(np.zeros((10, 3))),
        ContinuousDataset(np.zeros((10, 3)), ["a", "b", "c"]),
    ],
    ids=["array", "frame", "dataset"],
)
def test_federated_estimators_want_one_table_per_site(single: object) -> None:
    with pytest.raises(TypeError, match="one table per site"):
        FedRCD().fit(single)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="one table per site"):
        FedGES(n_rounds=1).fit(single)  # type: ignore[arg-type]


def test_fedges_rejects_sites_without_rows_and_n_jobs_zero() -> None:
    x = _discrete()
    with pytest.raises(ValueError, match="no rows"):
        FedGES(n_rounds=1).fit([x[:0], x[:0]])
    with pytest.raises(ValueError, match="n_jobs"):
        FedGES(n_rounds=1, n_jobs=0).fit([x, x])
    # One empty site among others is fine: it only adds its prior.
    FedGES(n_rounds=1).fit([x, x[:0]])


def test_ges_rejects_data_without_rows() -> None:
    with pytest.raises(ValueError, match="no rows"):
        GES().fit(np.zeros((0, 3), dtype=int))
    with pytest.raises(ValueError, match="no rows"):
        GES(criterion="gaussian_bic").fit(np.zeros((0, 3)))


@pytest.mark.parametrize("name", ["max_parents", "max_t_size"])
@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_ges_rejects_limits_that_are_not_counts(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        GES(**{name: value}).fit(_discrete())  # type: ignore[arg-type]


@pytest.mark.parametrize("params", [{"ess": float("nan")}, {"ess": float("inf")}])
def test_ges_rejects_a_non_finite_ess(params: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="ess"):
        GES(**params).fit(_discrete())  # type: ignore[arg-type]


def test_ges_learns_from_a_single_row_of_continuous_data() -> None:
    model = GES(criterion="gaussian_bic").fit(np.array([[0.1, 0.2, 0.3]]))
    assert model.dag_.num_edges == 0
    assert np.isfinite(model.score_)
