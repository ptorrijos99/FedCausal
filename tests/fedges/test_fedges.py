"""FedGES: what it learns, the sites, the round loop, and the estimator."""

from __future__ import annotations

import numpy as np
import pytest
from causalbase.datasets import load_alarm, load_asia, load_network
from causalbase.graph import DAG
from causalbase.metrics import shd
from causalbase.scores import BDeuScore
from causalbase.search import ges, xges

from fedcausal import FedGES
from fedcausal.data import contiguous_split
from fedcausal.exceptions import ConvergedBeforeFusionError, DenseFusionWarning
from fedcausal.fedges import BNModel, ModelEquality, ScoreImprovement, Site, run_rounds
from fedcausal.fusion import ConsensusUnion

# The pooled BDeu score of the global model FedGES learns from the bundled
# samples, split in contiguous blocks, and its SHD. A change means FedGES learns
# another structure: check whether that is intended, and say so in the changelog.
LEARNED = {
    ("alarm", 4, 5): (-53509.38211001393, 6),
    ("child", 5, 10): (-62517.16761369385, 6),
}


@pytest.mark.parametrize(("key", "expected"), LEARNED.items(), ids=lambda v: str(v))
def test_learns_the_same_structures_as_before(key: tuple[str, int, int], expected) -> None:  # type: ignore[no-untyped-def]
    name, n_sites, n_rounds = key
    bundle = load_network(name)
    model = FedGES(n_rounds=n_rounds).fit(contiguous_split(bundle.data, n_sites))
    score, distance = expected
    assert BDeuScore(bundle.data).score_dag(model.dag_) == pytest.approx(score, rel=1e-9)
    assert shd(bundle.truth, model.dag_) == distance


def test_parallel_sites_learn_what_sequential_ones_do() -> None:
    sites = contiguous_split(load_asia().data, 3)
    sequential = FedGES(n_rounds=4).fit(sites)
    parallel = FedGES(n_rounds=4, n_jobs=2).fit(sites)
    assert parallel.dag_ == sequential.dag_
    assert parallel.history_.equals(sequential.history_)


def test_every_refinement_runs_and_none_lowers_a_sites_score() -> None:
    parts = contiguous_split(load_alarm().data, 4)
    other = Site(parts[1]).learn().dag
    for refinement in (None, "forward", "backward", "both"):
        site = Site(parts[0], refinement=refinement)  # type: ignore[arg-type]
        own = site.learn()
        merged = site.merge(other)
        if refinement is not None:
            assert merged.score >= own.score - 1e-6
    with pytest.raises(RuntimeError, match="learn"):
        Site(parts[0]).merge(other)


def test_sites_learn_with_the_search_they_are_given() -> None:
    # On this third of insurance, GES, XGES-0 and XGES learn three different DAGs.
    sites = contiguous_split(load_network("insurance").data, 3)
    learned = set()
    for search in ("ges", "xges0", "xges"):
        own = Site(sites[0], search=search).learn()  # type: ignore[arg-type]
        expected = (
            ges(own_score := Site(sites[0]).score)
            if search == "ges"
            else xges(own_score, extended=search == "xges")
        )
        assert own.dag == expected.cpdag.to_dag()
        learned.add(tuple(sorted(own.dag.edges())))
        history = FedGES(n_rounds=1, search=search).fit(sites).history_  # type: ignore[arg-type]
        assert history.loc[history["site"] == 0, "score"].iloc[0] == pytest.approx(own.score)
    assert len(learned) == 3
    with pytest.raises(ValueError, match="search must be one of"):
        FedGES(search="pc").fit(sites)  # type: ignore[arg-type]


def test_history_pooled_levels_and_knowledge() -> None:
    bundle = load_asia()
    frame = bundle.data.to_pandas()
    first, second = frame.iloc[:2500], frame.iloc[2500:].copy()
    second.iloc[:, 0] = "rare"  # a level only one site has
    names = bundle.data.names
    model = FedGES(n_rounds=3, required=[(names[0], names[7])], forbidden=[(5, 6)])
    model.fit([first, second])
    assert list(model.history_["site"][:3]) == [0, 1, -1]
    assert model.history_.loc[model.history_["site"] == -1, "score"].isna().all()
    assert list(model.adjacency_.columns) == list(frame.columns)
    for dag in [model.dag_] + [m.dag for m in model.site_models_]:
        skeleton = {frozenset(e) for e in dag.edges()}
        assert frozenset((0, 7)) in skeleton
        assert frozenset((5, 6)) not in skeleton


def _model(edges: list[tuple[int, int]], score: float) -> BNModel:
    return BNModel(DAG.from_edges(3, edges), score)


def test_the_round_loop_stops_reports_and_warns() -> None:
    builds = iter([[_model([(0, 1)], -10.0)], [_model([(0, 1)], -10.0)]])
    merged: list[DAG] = []
    global_dag, rounds = run_rounds(
        lambda: next(builds), merged.append, ConsensusUnion(), ScoreImprovement(), n_rounds=5
    )
    assert rounds == 2
    assert global_dag is not None and global_dag.edges() == [(0, 1)]
    assert len(merged) == 1

    criterion = ModelEquality()  # stops once every site repeats an earlier model
    a, b = _model([(0, 1)], 0.0), _model([(1, 2)], 0.0)
    assert not criterion.check_convergence([a, b])
    assert not criterion.check_convergence([a, _model([(0, 2)], 0.0)])
    assert criterion.check_convergence([a, b])

    class Densify:
        def fuse(self, dags: object) -> DAG:
            return DAG.from_edges(3, [(0, 1), (0, 2), (1, 2)])

    with pytest.warns(DenseFusionWarning, match="3 edges against 1"):
        run_rounds(lambda: [_model([(0, 1)], -1.0)], lambda g: None, Densify(), None, n_rounds=1)

    class StopAtOnce:
        def check_convergence(self, local_models: object) -> bool:
            return True

    with pytest.raises(ConvergedBeforeFusionError):
        FedGES(convergence=StopAtOnce()).fit(contiguous_split(load_asia().data, 2))


def test_continuous_sites_federate_under_the_gaussian_bic() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=6000)
    b = 0.9 * a + rng.normal(size=6000)
    c = 0.8 * b + rng.normal(size=6000)
    sites = np.array_split(np.column_stack([a, b, c]), 3)
    model = FedGES(criterion="gaussian_bic", n_rounds=3).fit(sites)
    assert {frozenset(e) for e in model.dag_.edges()} == {frozenset((0, 1)), frozenset((1, 2))}
