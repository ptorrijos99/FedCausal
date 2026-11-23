"""Greedy Equivalence Search: what it recovers, and that it stops at a true local optimum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcausal import GES
from fedcausal.data import as_discrete
from fedcausal.datasets import load_alarm, load_asia, load_child, load_network
from fedcausal.graph import DAG, PDAG
from fedcausal.graph.cpdag import dag_to_cpdag
from fedcausal.metrics import shd
from fedcausal.scores import BDeuScore
from fedcausal.search import _operators as ops
from fedcausal.search import ges

# The score GES reaches on each bundled sample, and the SHD of what it learns. A
# change in a score means GES now stops at another local optimum; a change in an
# SHD alone, another structure of the same score. Check that either is intended,
# and say so in the changelog. Scores are compared to 1e-9, not bit for bit.
LEARNED = {
    ("asia", "bdeu"): (-11292.382734195426, 2),
    ("child", "bdeu"): (-62114.38162544265, 2),
    ("insurance", "bdeu"): (-67131.24624087544, 17),
    ("alarm", "bdeu"): (-53644.30126961614, 10),
    ("asia", "bic"): (-11303.722075470887, 2),
    ("alarm", "bic"): (-54594.88083963548, 12),
}


def test_learns_the_same_structures_as_before() -> None:
    for (name, criterion), (score, distance) in LEARNED.items():
        bundle = load_network(name)
        model = GES(criterion=criterion).fit(bundle.data)
        assert model.score_ == pytest.approx(score, rel=1e-9), (name, criterion)
        assert shd(bundle.truth, model.dag_) == distance, (name, criterion)


def test_stops_where_no_single_operator_improves_the_score() -> None:
    data = load_child().data
    score = BDeuScore(data)
    cpdag = ges(score, data.n_vars)
    for y in range(data.n_vars):
        assert ops.inserts_into(score, cpdag, y, max_t_size=2) == {}
        assert ops.deletes_into(score, cpdag, y) == {}


def test_a_search_from_a_given_dag_never_ends_below_it() -> None:
    bundle = load_alarm()
    score = BDeuScore(bundle.data)
    learned = ges(score, bundle.data.n_vars, bundle.truth).to_dag()
    assert score.score_dag(learned) >= score.score_dag(bundle.truth)
    model = GES().fit(bundle.data, initial_graph=bundle.truth)
    assert model.cpdag_ == dag_to_cpdag(model.dag_)
    with pytest.raises(ValueError, match="nodes"):
        GES().fit(bundle.data, initial_graph=DAG(3))


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
    # A Delete orients the neighbours it keeps into a collider.
    graph = PDAG(3)
    for u, v in [(0, 1), (0, 2), (1, 2)]:
        graph.add_undirected(u, v)
    ops.apply_delete(graph, 0, 1, frozenset({2}))
    assert sorted(graph.directed_edges()) == [(0, 2), (1, 2)]
    assert graph.undirected_edges() == []


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
    with pytest.raises(ValueError, match="criterion must be one of"):
        GES(criterion="aic").fit(load_asia().data)  # type: ignore[arg-type]


def test_max_parents_caps_every_family() -> None:
    model = GES(max_parents=1).fit(load_alarm().data)
    assert max(len(model.dag_.parents(v)) for v in range(model.n_features_in_)) == 1


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
    model_selection = pytest.importorskip("sklearn.model_selection")
    search = model_selection.GridSearchCV(GES(), {"max_parents": [0, 2]}, cv=2)
    search.fit(load_asia().data.to_pandas())
    assert search.best_params_ == {"max_parents": 2}


def _inserts_by_definition(
    score: BDeuScore, pdag: PDAG, y: int, max_t_size: int
) -> dict[int, ops.Candidate]:
    """Every Insert into `y` checked one at a time, straight from the operator's definition."""
    best: dict[int, ops.Candidate] = {}
    undirected = pdag.undirected_neighbors(y)
    for x in range(pdag.num_nodes):
        if x == y or x in pdag.adjacent(y):
            continue
        na = undirected & pdag.adjacent(x)
        for t in ops._subsets(sorted(undirected - na), max_t_size):
            cond = na | t
            if not ops.is_clique(pdag, cond):
                continue
            reached, stack = {y}, [y]
            while stack:
                v = stack.pop()
                for w in (pdag.children(v) | pdag.undirected_neighbors(v)) - reached:
                    reached.add(w)
                    if w not in cond:
                        stack.append(w)
            if x in reached:
                continue
            base = pdag.parents(y) | cond
            without, with_x = score.local_score(y, base), score.local_score(y, base | {x})
            delta = with_x - without
            if delta <= 0 or ops._is_noise(delta, with_x, without):
                continue
            if x not in best or delta > best[x][0]:
                best[x] = (delta, t)
    return best


def test_inserts_into_finds_what_the_definition_gives() -> None:
    bundle = load_alarm()
    data = as_discrete(bundle.data)
    partial = DAG(data.n_vars)
    for i, (u, v) in enumerate(sorted(bundle.truth.edges())):
        if i % 3:
            partial.add_edge(u, v)
    cpdag = dag_to_cpdag(partial)
    assert cpdag.undirected_edges()
    score = BDeuScore(data)
    for y in range(data.n_vars):
        assert ops.inserts_into(score, cpdag, y, max_t_size=2) == _inserts_by_definition(
            score, cpdag, y, 2
        )
