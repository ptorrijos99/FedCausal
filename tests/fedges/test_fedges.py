"""FedGES: the sites, the round loop, and the estimator."""

from __future__ import annotations

import pytest

from fedcausal import FedGES
from fedcausal.data import contiguous_split
from fedcausal.datasets import load_alarm, load_asia, load_child
from fedcausal.exceptions import ConvergedBeforeFusionError
from fedcausal.fedges import (
    AnyOf,
    BNModel,
    ModelEquality,
    ScoreImprovement,
    Site,
    run_rounds,
)
from fedcausal.fusion import ConsensusUnion
from fedcausal.graph import DAG
from fedcausal.metrics import shd


def _model(edges: list[tuple[int, int]], score: float) -> BNModel:
    return BNModel(DAG.from_edges(3, edges), score)


def test_federating_child_costs_little_against_learning_it_in_one_place() -> None:
    bundle = load_child()
    model = FedGES().fit(contiguous_split(bundle.data, 5))
    assert shd(bundle.truth, model.dag_) <= 10


def test_parallel_sites_learn_what_sequential_ones_do() -> None:
    sites = contiguous_split(load_asia().data, 3)
    sequential = FedGES(n_rounds=4).fit(sites)
    parallel = FedGES(n_rounds=4, n_jobs=2).fit(sites)
    assert parallel.dag_ == sequential.dag_
    assert parallel.history_.equals(sequential.history_)


@pytest.mark.parametrize("refinement", [None, "forward", "backward", "both"])
def test_every_refinement_runs_and_never_lowers_a_sites_score(refinement: str | None) -> None:
    site = Site(contiguous_split(load_alarm().data, 4)[0], refinement=refinement)  # type: ignore[arg-type]
    own = site.learn()
    other = Site(contiguous_split(load_alarm().data, 4)[1]).learn().dag
    merged = site.merge(other)
    if refinement is not None:
        assert merged.score >= own.score - 1e-6


def test_a_site_refuses_to_merge_before_it_has_learned() -> None:
    site = Site(load_asia().data)
    with pytest.raises(RuntimeError, match="learn"):
        site.merge(DAG(8))


def test_the_history_records_each_site_and_the_global_model_per_round() -> None:
    model = FedGES(n_rounds=3, convergence=AnyOf()).fit(contiguous_split(load_asia().data, 2))
    history = model.history_
    assert model.n_rounds_ == 3
    assert list(history["site"]) == [0, 1, -1] * 3
    assert history.loc[history["site"] == -1, "score"].isna().all()
    assert len(model.site_models_) == 2


def test_the_sites_may_pool_levels_unseen_at_some_of_them() -> None:
    frame = load_asia().data.to_pandas()
    first, second = frame.iloc[:2500], frame.iloc[2500:].copy()
    second.iloc[:, 0] = "rare"
    model = FedGES(n_rounds=2).fit([first, second])
    assert model.n_features_in_ == 8
    assert list(model.adjacency_.columns) == list(frame.columns)


def test_stopping_before_the_first_fusion_is_an_error() -> None:
    class StopAtOnce:
        def check_convergence(self, local_models: object) -> bool:
            return True

    with pytest.raises(ConvergedBeforeFusionError):
        FedGES(convergence=StopAtOnce()).fit(contiguous_split(load_asia().data, 2))


def test_run_rounds_stops_on_convergence_and_reports_the_rounds_run() -> None:
    builds = iter([[_model([(0, 1)], -10.0)], [_model([(0, 1)], -10.0)]])
    merged: list[DAG] = []

    global_dag, rounds = run_rounds(
        lambda: next(builds), merged.append, ConsensusUnion(), ScoreImprovement(), n_rounds=5
    )

    assert rounds == 2
    assert global_dag is not None
    assert global_dag.edges() == [(0, 1)]
    assert len(merged) == 1


def test_model_equality_waits_until_every_site_repeats() -> None:
    criterion = ModelEquality()
    a, b = _model([(0, 1)], 0.0), _model([(1, 2)], 0.0)
    assert not criterion.check_convergence([a, b])
    assert not criterion.check_convergence([a, _model([(0, 2)], 0.0)])
    assert criterion.check_convergence([a, b])


def test_estimators_expose_their_parameters_like_scikit_learn() -> None:
    model = FedGES(n_rounds=3, n_jobs=-1)
    assert model.get_params()["n_rounds"] == 3
    assert model.set_params(n_rounds=5).n_rounds == 5
    assert repr(model) == "FedGES(n_rounds=5, n_jobs=-1)"
    with pytest.raises(ValueError, match="no parameter"):
        model.set_params(rounds=1)
    sklearn = pytest.importorskip("sklearn.base")
    assert sklearn.clone(model).get_params() == model.get_params()


def test_knowledge_holds_in_every_site_and_in_the_global_model() -> None:
    bundle = load_asia()
    names = bundle.data.names
    model = FedGES(n_rounds=3, required=[(names[0], names[7])], forbidden=[(5, 6)]).fit(
        contiguous_split(bundle.data, 3)
    )
    for dag in [model.dag_] + [m.dag for m in model.site_models_]:
        skeleton = {frozenset(e) for e in dag.edges()}
        assert frozenset((0, 7)) in skeleton
        assert frozenset((5, 6)) not in skeleton


def test_a_global_model_far_denser_than_the_sites_is_reported() -> None:
    from fedcausal.exceptions import DenseFusionWarning

    class Densify:
        def fuse(self, dags: object) -> DAG:
            return DAG.from_edges(3, [(0, 1), (0, 2), (1, 2)])

    with pytest.warns(DenseFusionWarning, match="3 edges against 1"):
        run_rounds(lambda: [_model([(0, 1)], -1.0)], lambda g: None, Densify(), None, n_rounds=1)


def test_continuous_sites_federate_under_the_gaussian_bic() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 6000
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = 0.8 * b + rng.normal(size=n)
    sites = np.array_split(np.column_stack([a, b, c]), 3)
    model = FedGES(criterion="gaussian_bic", n_rounds=3).fit(sites)
    assert {frozenset(e) for e in model.dag_.edges()} == {frozenset((0, 1)), frozenset((1, 2))}
