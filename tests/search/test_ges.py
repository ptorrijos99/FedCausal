"""Greedy Equivalence Search: what it recovers, and that it stops at a true local optimum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcausal import GES
from fedcausal.data import as_discrete
from fedcausal.datasets import load_alarm, load_asia, load_child
from fedcausal.graph import DAG, PDAG
from fedcausal.graph.cpdag import dag_to_cpdag
from fedcausal.metrics import shd
from fedcausal.scores import BDeuScore
from fedcausal.search import _operators as ops
from fedcausal.search import ges


@pytest.mark.parametrize(("loader", "max_shd"), [(load_asia, 3), (load_child, 3), (load_alarm, 12)])
def test_recovers_the_bundled_networks(loader, max_shd: int) -> None:  # type: ignore[no-untyped-def]
    bundle = loader()
    assert shd(bundle.truth, GES().fit(bundle.data).dag_) <= max_shd


def test_stops_where_no_single_operator_improves_the_score() -> None:
    data = load_child().data
    score = BDeuScore(data)
    cpdag = ges(score, data.n_vars)
    for y in range(data.n_vars):
        assert ops.inserts_into(score, cpdag, y, max_t_size=2) == {}
        assert ops.deletes_into(score, cpdag, y) == {}


def test_never_ends_below_where_it_starts() -> None:
    bundle = load_alarm()
    score = BDeuScore(bundle.data)
    learned = ges(score, bundle.data.n_vars, bundle.truth).to_dag()
    assert score.score_dag(learned) >= score.score_dag(bundle.truth)


def test_capping_the_conditioning_subsets_finds_the_same_optimum_here() -> None:
    data = load_child().data
    score = BDeuScore(data)
    assert ges(score, data.n_vars, max_t_size=2) == ges(score, data.n_vars, max_t_size=None)


def test_only_candidate_pairs_gain_edges() -> None:
    data = load_asia().data
    allowed = frozenset({(0, 1), (1, 0), (2, 3), (3, 2)})
    cpdag = ges(BDeuScore(data), data.n_vars, candidates=allowed)
    edges = cpdag.directed_edges() + cpdag.undirected_edges()
    assert {frozenset(e) for e in edges} <= {frozenset(p) for p in allowed}


def test_the_estimator_takes_frames_and_arrays_and_names_its_result() -> None:
    frame = load_asia().data.to_pandas()
    by_frame = GES().fit(frame)
    by_array = GES().fit(frame.to_numpy())

    assert by_frame.cpdag_ == by_array.cpdag_
    assert list(by_frame.feature_names_in_) == list(frame.columns)
    assert list(by_array.feature_names_in_) == [f"x{j}" for j in range(frame.shape[1])]
    adjacency = by_frame.adjacency_
    assert isinstance(adjacency, pd.DataFrame)
    assert adjacency.to_numpy().sum() == 2 * len(by_frame.cpdag_.undirected_edges()) + len(
        by_frame.cpdag_.directed_edges()
    )
    assert by_frame.score_ == pytest.approx(BDeuScore(load_asia().data).score_dag(by_frame.dag_))


def test_a_stronger_structure_prior_learns_fewer_edges() -> None:
    data = load_alarm().data
    plain = GES(expected_parents=None).fit(data).dag_.num_edges
    sparse = GES(expected_parents=0.01).fit(data).dag_.num_edges
    assert sparse < plain


def test_an_insert_that_would_close_a_directed_cycle_is_never_offered() -> None:
    """x and y are strongly dependent, but y -> z -> x already exists."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    z = np.where(rng.random(2000) < 0.9, y, 1 - y)
    x = np.where(rng.random(2000) < 0.9, z, 1 - z)
    dataset = as_discrete(np.column_stack([x, y, z]))
    graph = PDAG(3)
    graph.add_directed(1, 2)
    graph.add_directed(2, 0)
    assert 0 not in ops.inserts_into(BDeuScore(dataset), graph, 1, max_t_size=2)


def test_a_delete_orients_the_kept_neighbours_into_a_collider() -> None:
    graph = PDAG(3)
    for u, v in [(0, 1), (0, 2), (1, 2)]:
        graph.add_undirected(u, v)
    ops.apply_delete(graph, 0, 1, frozenset({2}))
    assert sorted(graph.directed_edges()) == [(0, 2), (1, 2)]
    assert graph.undirected_edges() == []


def test_the_search_starts_from_a_given_dag() -> None:
    bundle = load_asia()
    model = GES().fit(bundle.data, initial_graph=bundle.truth)
    assert model.cpdag_ == dag_to_cpdag(model.dag_)
    with pytest.raises(ValueError, match="nodes"):
        GES().fit(bundle.data, initial_graph=DAG(3))


def test_required_and_forbidden_adjacencies_hold_whatever_the_data_say() -> None:
    bundle = load_asia()
    names = bundle.data.names
    learned = {frozenset(e) for e in GES().fit(bundle.data).dag_.edges()}
    present = next(iter(learned))
    absent = next(
        frozenset((a, b))
        for a in range(len(names))
        for b in range(a + 1, len(names))
        if frozenset((a, b)) not in learned
    )
    model = GES(
        required=[tuple(names[i] for i in absent)],
        forbidden=[tuple(sorted(present))],
    ).fit(bundle.data)
    skeleton = {frozenset(e) for e in model.dag_.edges()}
    assert absent in skeleton
    assert present not in skeleton


def test_knowledge_rejects_contradictions_and_unknown_names() -> None:
    from fedcausal.search import Knowledge

    with pytest.raises(ValueError, match="both required and forbidden"):
        Knowledge.from_pairs([("a", "b")], [("b", "a")], names=["a", "b"])
    with pytest.raises(ValueError, match="unknown variable"):
        Knowledge.from_pairs([("a", "z")], names=["a", "b"])
    with pytest.raises(ValueError, match="no variable at index"):
        Knowledge.from_pairs(forbidden=[(0, 2)], names=["a", "b"])
    with pytest.raises(ValueError, match="neither"):
        Knowledge.from_pairs([(0, 1.7)], names=["a", "b"])


def test_max_parents_caps_every_family() -> None:
    bundle = load_alarm()
    model = GES(max_parents=1).fit(bundle.data)
    assert max(len(model.dag_.parents(v)) for v in range(bundle.data.n_vars)) <= 1
    assert model.dag_.num_edges > 0


def _linear_gaussian(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = rng.normal(size=n)
    d = 0.7 * b - 0.8 * c + rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


def test_gaussian_bic_finds_the_collider_of_continuous_data() -> None:
    model = GES(criterion="gaussian_bic").fit(_linear_gaussian(5000, 0))
    adjacency = model.adjacency_
    assert adjacency.loc["b", "d"] == 1 and adjacency.loc["d", "b"] == 0
    assert adjacency.loc["c", "d"] == 1 and adjacency.loc["d", "c"] == 0
    assert adjacency.loc["a", "b"] == 1 and adjacency.loc["b", "a"] == 1
    assert adjacency.loc["a", "c"] == 0


@pytest.mark.parametrize("criterion", ["bdeu", "bic"])
def test_discrete_criteria_recover_asia_closely(criterion: str) -> None:
    bundle = load_asia()
    assert shd(bundle.truth, GES(criterion=criterion).fit(bundle.data).dag_) <= 3


def test_score_is_the_held_out_likelihood_per_row_and_ranks_structures() -> None:
    frame = load_asia().data.to_pandas()
    train, test = frame.iloc[:3000], frame.iloc[3000:]
    learned = GES().fit(train)
    empty = GES(forbidden=[(a, b) for a in frame for b in frame if a < b]).fit(train)
    assert learned.dag_.num_edges > 0
    assert empty.dag_.num_edges == 0
    assert learned.score(test) > empty.score(test)
    assert -10 < learned.score(test) < 0
    with pytest.raises(ValueError, match="outside its levels"):
        learned.score(test.assign(asia="maybe"))


def test_a_parameter_search_selects_hyperparameters_on_held_out_rows() -> None:
    model_selection = pytest.importorskip("sklearn.model_selection")
    search = model_selection.GridSearchCV(GES(), {"max_parents": [0, 2]}, cv=2)
    search.fit(load_asia().data.to_pandas())
    assert search.best_params_ == {"max_parents": 2}


def test_an_unknown_criterion_is_an_error() -> None:
    with pytest.raises(ValueError, match="criterion must be one of"):
        GES(criterion="aic").fit(load_asia().data)  # type: ignore[arg-type]
