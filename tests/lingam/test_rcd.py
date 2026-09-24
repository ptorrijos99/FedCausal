import numpy as np
import pytest

from fedcausal.graph import DAG
from fedcausal.lingam import RCDResult, fedrcd, fedrcd_exact, fedrcd_noniterative, ishc_beta
from fedcausal.lingam.cumulants import aggregate_and_compute_cumulants, compute_local_raw_stats


def _chain_clients(seed: int, n: int = 6000, n_clients: int = 3):
    """x0 -> x1 -> x2 -> x3, split across clients. True order is 0,1,2,3."""
    rng = np.random.default_rng(seed)
    e = rng.laplace(0.0, 1.0, size=(n, 4))
    x = np.zeros((n, 4))
    x[:, 0] = e[:, 0]
    for j in range(1, 4):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return list(np.array_split(x, n_clients, axis=0))


def _stats(clients):
    return aggregate_and_compute_cumulants([compute_local_raw_stats(c, order=4) for c in clients])


@pytest.mark.parametrize("solver", [fedrcd, fedrcd_noniterative, ishc_beta])
def test_solvers_recover_the_chain_order(solver):
    result = solver(_stats(_chain_clients(0)))
    assert result.causal_order == (0, 1, 2, 3)


def test_exact_solver_recovers_the_chain_order():
    result = fedrcd_exact(_chain_clients(1))
    assert result.causal_order == (0, 1, 2, 3)


def test_result_is_a_frozen_dataclass_with_the_right_shapes():
    result = fedrcd(_stats(_chain_clients(2)))
    assert isinstance(result, RCDResult)
    assert result.b.shape == (4, 4)
    assert len(result.causal_order) == 4
    with pytest.raises(AttributeError):
        result.causal_order = (1, 0, 2, 3)


def test_to_dag_orients_parents_towards_children():
    result = fedrcd(_stats(_chain_clients(3)))
    dag = result.to_dag(threshold=0.3)
    assert isinstance(dag, DAG)
    edges = set(dag.edges())
    assert (0, 1) in edges
    assert (1, 2) in edges
    assert (2, 3) in edges


def test_to_dag_threshold_drops_weak_coefficients():
    result = fedrcd(_stats(_chain_clients(4)))
    assert len(result.to_dag(threshold=0.0).edges()) >= len(result.to_dag(threshold=10.0).edges())
    assert result.to_dag(threshold=10.0).edges() == []


def _skewed_chain_clients(seed: int, n: int = 6000, n_clients: int = 3):
    """The same chain with skewed noise, which the third-order discriminator needs."""
    rng = np.random.default_rng(seed)
    e = rng.exponential(1.0, size=(n, 4)) - 1.0
    x = np.zeros((n, 4))
    x[:, 0] = e[:, 0]
    for j in range(1, 4):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return list(np.array_split(x, n_clients, axis=0))


def test_third_order_only_stats_still_work():
    clients = _skewed_chain_clients(5)
    stats = aggregate_and_compute_cumulants([compute_local_raw_stats(c, order=3) for c in clients])
    assert fedrcd(stats).causal_order == (0, 1, 2, 3)


def test_missing_n_raises_key_error_in_the_lasso_path():
    """A silent 1000-sample default would quietly mis-scale the Lasso penalty."""
    stats = dict(_stats(_chain_clients(6)))
    del stats["n"]
    with pytest.raises(KeyError, match="stats is missing 'n'"):
        fedrcd(stats, use_lasso=True)
