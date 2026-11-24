"""The estimators' shared contract: parameters, input checks, fitted attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcausal import GES, XGES, FedGES, FedRCD
from fedcausal.data import ContinuousDataset
from fedcausal.exceptions import FedCausalError, NotFittedError
from fedcausal.lingam import cumulants_from_sums, fedrcd, moment_sums


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


def test_the_scikit_learn_parameter_protocol() -> None:
    sklearn_base = pytest.importorskip("sklearn.base")
    for cls in (GES, XGES, FedGES, FedRCD):
        model = cls()
        assert repr(model) == f"{cls.__name__}()"
        assert model.set_params(**model.get_params()) is model
        with pytest.raises(ValueError, match="no parameter"):
            model.set_params(not_a_parameter=1)
        # Before fit, a result is missing, as scikit-learn expects.
        with pytest.raises(NotFittedError):
            _ = model.adjacency_
        assert not hasattr(model, "adjacency_")
    assert issubclass(NotFittedError, FedCausalError)
    for model in (
        GES(ess=5.0),
        XGES(extended=False),
        FedGES(n_rounds=3, n_jobs=-1),
        FedRCD(order=3),
    ):
        assert sklearn_base.clone(model).get_params() == model.get_params()
    assert repr(FedGES(n_rounds=3, n_jobs=-1)) == "FedGES(n_rounds=3, n_jobs=-1)"


def test_fedrcd_is_the_solver_over_the_pooled_sums_and_names_its_result() -> None:
    sites = _linear_sites()
    model = FedRCD(pruning=None).fit([pd.DataFrame(x, columns=["a", "b", "c"]) for x in sites])
    expected = fedrcd(cumulants_from_sums([moment_sums(x) for x in sites]), pruning=None)
    assert model.causal_order_ == ["abc"[i] for i in expected.causal_order] == ["a", "b", "c"]
    np.testing.assert_array_equal(model.coef_, expected.coef)
    assert list(model.feature_names_in_) == ["a", "b", "c"]
    pruned = FedRCD().fit(sites)
    assert sorted(pruned.dag_.edges()) == [(0, 1), (1, 2)]
    assert (pruned.adjacency_.loc["x0", "x1"], pruned.adjacency_.loc["x1", "x0"]) == (1, 0)
    assert FedRCD(method="exact").fit(sites).causal_order_ == ["x0", "x1", "x2"]


def test_estimators_refuse_bad_input_and_settings() -> None:
    x = _discrete()
    for single in (
        np.zeros((10, 3)),
        pd.DataFrame(np.zeros((10, 3))),
        ContinuousDataset(np.zeros((10, 3)), list("abc")),
    ):
        for model in (FedRCD(), FedGES(n_rounds=1)):
            with pytest.raises(TypeError, match="one table per site"):
                model.fit(single)  # type: ignore[arg-type]
    refusals = [
        (lambda: FedGES(n_rounds=1).fit([x[:0], x[:0]]), "no rows"),
        (lambda: FedGES(n_rounds=1, n_jobs=0).fit([x, x]), "n_jobs"),
        (lambda: GES().fit(np.zeros((0, 3), dtype=int)), "no rows"),
        (lambda: GES(criterion="gaussian_bic").fit(np.zeros((0, 3))), "no rows"),
        (lambda: GES(ess=float("nan")).fit(x), "ess"),
        (lambda: FedRCD(method="nope").fit(_linear_sites()), "method"),  # type: ignore[arg-type]
        (lambda: FedRCD(order=5).fit(_linear_sites()), "order"),  # type: ignore[arg-type]
    ]
    for name in ("max_parents", "max_t_size"):
        for value in (-1, 1.5, True, "2"):
            refusals.append((lambda n=name, v=value: GES(**{n: v}).fit(x), name))  # type: ignore[misc]
    for fit, match in refusals:
        with pytest.raises(ValueError, match=match):
            fit()


def test_degenerate_but_valid_input_is_learned_from() -> None:
    x = _discrete()
    FedGES(n_rounds=1).fit([x, x[:0]])  # an empty site only adds its prior
    model = GES(criterion="gaussian_bic").fit(np.array([[0.1, 0.2, 0.3]]))
    assert model.dag_.num_edges == 0
    assert np.isfinite(model.score_)
