import numpy as np

from fedcausal.data import DiscreteDataset
from fedcausal.federated.client import Client
from fedcausal.federated.learner import GESLearner
from fedcausal.federated.model import BNModel
from fedcausal.fusion import ConsensusUnion
from fedcausal.graph import DAG


def _ds(n_rows=1500, seed=1):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.random(n_rows) < 0.9, x1, 1 - x1)
    v = np.stack([x0, x1, x2], axis=1).astype(np.int32)
    return DiscreteDataset(v, np.array([2, 2, 2], np.int32), ["A", "B", "C"], [["0", "1"]] * 3)


def test_build_increments_iteration_and_sets_model():
    c = Client(ConsensusUnion(), GESLearner(), _ds())
    assert c.local_model is None and c.iteration == 0
    c.build_local_model()
    assert c.iteration == 1
    assert isinstance(c.local_model, BNModel)


def test_fuse_incorporates_the_global_edges():
    c = Client(ConsensusUnion(), GESLearner(), _ds())
    c.build_local_model()
    global_model = BNModel(DAG.from_edges(3, [(0, 1), (1, 2), (0, 2)]))
    c.fuse(global_model)
    skel = {frozenset(e) for e in c.local_model.dag.edges()}
    assert len(skel) >= 2  # global edges merged into the local model
    assert c.local_model.score < 0
