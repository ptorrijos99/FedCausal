import numpy as np

from fedcausal.data import DiscreteDataset
from fedcausal.fedges.learner import GESLearner
from fedcausal.fedges.model import BNModel
from fedcausal.graph import DAG


def _chain_dataset(n_rows=2000, seed=0):
    # X0 -> X1 -> X2, all binary, strong dependence.
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.random(n_rows) < 0.9, x1, 1 - x1)
    values = np.stack([x0, x1, x2], axis=1).astype(np.int32)
    cards = np.array([2, 2, 2], dtype=np.int32)
    cats = [["0", "1"]] * 3
    return DiscreteDataset(values, cards, ["X0", "X1", "X2"], cats)


def test_build_recovers_the_chain_skeleton():
    ds = _chain_dataset()
    learner = GESLearner()
    model = learner.build_local_model(None, ds)
    skel = {frozenset(e) for e in model.dag.edges()}
    assert skel == {frozenset((0, 1)), frozenset((1, 2))}
    assert model.score < 0  # BDeu is negative here


def test_build_warm_starts_from_previous_without_error():
    ds = _chain_dataset()
    learner = GESLearner()
    first = learner.build_local_model(None, ds)
    second = learner.build_local_model(first, ds)
    assert {frozenset(e) for e in second.dag.edges()} == {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }


def test_score_dag_matches_a_direct_bdeu():
    ds = _chain_dataset()
    learner = GESLearner()
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    assert learner.score_dag(dag, ds) < 0


def test_refinement_none_is_identity():
    ds = _chain_dataset()
    learner = GESLearner(refinement="None")
    old = BNModel(DAG.from_edges(3, [(0, 1)]))
    fused = BNModel(DAG.from_edges(3, [(0, 1), (1, 2)]))
    out = learner.refine_local_model(old, fused, ds)
    assert out is fused
    assert learner.refinement_name == "None"


def test_refinement_fes_can_add_a_true_edge():
    ds = _chain_dataset()
    learner = GESLearner(refinement="FES")
    old = BNModel(DAG.from_edges(3, [(0, 1)]))
    fused = BNModel(DAG.from_edges(3, [(0, 1), (1, 2)]))
    out = learner.refine_local_model(old, fused, ds)
    skel = {frozenset(e) for e in out.dag.edges()}
    assert frozenset((1, 2)) in skel


def test_refinement_bes_can_remove_a_spurious_edge():
    ds = _chain_dataset()
    learner = GESLearner(refinement="BES")
    # old has a spurious 0->2 edge; fused dropped it.
    old = BNModel(DAG.from_edges(3, [(0, 1), (1, 2), (0, 2)]))
    fused = BNModel(DAG.from_edges(3, [(0, 1), (1, 2)]))
    out = learner.refine_local_model(old, fused, ds)
    skel = {frozenset(e) for e in out.dag.edges()}
    assert frozenset((0, 2)) not in skel
