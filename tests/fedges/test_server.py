import numpy as np

from fedcausal.data import DiscreteDataset, contiguous_split
from fedcausal.fedges.client import Client
from fedcausal.fedges.convergence import (
    ModelEquality,
    MultipleConvergence,
    ScoreImprovement,
)
from fedcausal.fedges.learner import GESLearner
from fedcausal.fedges.observer import MetricsObserver, RoundStats
from fedcausal.fedges.server import Server, run_fedges
from fedcausal.fusion import Consensus, ConsensusUnion


def _ds(n_rows=3000, seed=2):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.random(n_rows) < 0.9, x1, 1 - x1)
    v = np.stack([x0, x1, x2], axis=1).astype(np.int32)
    return DiscreteDataset(v, np.array([2, 2, 2], np.int32), ["A", "B", "C"], [["0", "1"]] * 3)


def test_server_assigns_ids_and_runs():
    ds = _ds()
    parts = contiguous_split(ds, 3)
    clients = [Client(ConsensusUnion(), GESLearner(), p) for p in parts]
    conv = MultipleConvergence([ScoreImprovement(), ModelEquality()])
    server = Server(Consensus(50), conv, clients, n_iterations=10)
    obs = MetricsObserver()
    server.run(obs)
    assert [c.id for c in clients] == [0, 1, 2]
    assert server.global_model is not None
    assert any(r.stage == "server" for r in obs.rows)
    assert all(isinstance(r, RoundStats) for r in obs.rows)
    # server rows carry no data-derived score; build rows do
    assert all(r.score is None for r in obs.rows if r.stage == "server")


def test_server_converges_before_budget_on_stable_data():
    ds = _ds()
    parts = contiguous_split(ds, 4)
    clients = [Client(ConsensusUnion(), GESLearner(), p) for p in parts]
    conv = MultipleConvergence([ScoreImprovement(), ModelEquality()])
    server = Server(Consensus(50), conv, clients, n_iterations=50)
    obs = MetricsObserver()
    server.run(obs)
    last_it = max(r.iteration for r in obs.rows)
    assert last_it < 50  # converged early


def test_run_fedges_end_to_end():
    ds = _ds()
    global_model, stats = run_fedges(ds, n_clients=3, n_iterations=10)
    assert global_model.dag.num_nodes == 3
    assert len(stats) > 0
