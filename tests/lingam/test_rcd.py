import numpy as np
import pytest

from fedcausal.graph import DAG
from fedcausal.lingam import RCDResult, fedrcd, fedrcd_exact
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


@pytest.mark.parametrize("method", ["deflation", "ratio", "sort"])
def test_every_method_recovers_the_chain_order(method):
    assert fedrcd(_stats(_chain_clients(0)), method=method).causal_order == (0, 1, 2, 3)


def test_exact_solver_recovers_the_chain_order():
    assert fedrcd_exact(_chain_clients(1)).causal_order == (0, 1, 2, 3)
    assert fedrcd_exact([np.vstack(_chain_clients(1))]).causal_order == (0, 1, 2, 3)


def test_ratio_deflation_removes_the_source_with_its_coefficient():
    """Cum(s, s, j) / Cum(s, s, s) is the coefficient of s in j; Cum(j, j, s) gives its square."""
    from fedcausal.lingam.rcd import _deflated_order

    rng = np.random.default_rng(7)
    e = rng.exponential(1.0, size=(200_000, 2)) - 1.0
    x = np.column_stack([e[:, 0], 1.5 * e[:, 0] + e[:, 1]])
    stats = aggregate_and_compute_cumulants([compute_local_raw_stats(x, 3)])
    cur = {k: np.array(stats[k]) for k in ("cov", "c3", "c12", "c21")}
    assert _deflated_order(cur, 3, "ratio") == [0, 1]
    assert cur["cov"][1, 1] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("method", ["deflation", "ratio", "sort"])
def test_third_order_methods_need_only_third_order_sums(method):
    clients = _skewed_chain_clients(8)
    stats = aggregate_and_compute_cumulants([compute_local_raw_stats(c, order=3) for c in clients])
    assert fedrcd(stats, method=method).causal_order == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="fourth-order"):
        fedrcd(stats, order=4)


def test_bad_method_or_order_is_an_error():
    stats = _stats(_chain_clients(9))
    with pytest.raises(ValueError, match="method"):
        fedrcd(stats, method="bogus")
    with pytest.raises(ValueError, match="order"):
        fedrcd(stats, order=5)


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
