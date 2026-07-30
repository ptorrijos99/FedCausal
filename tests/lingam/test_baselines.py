import numpy as np
import pytest

from fedcausal.lingam import RCDResult, fedhc, fedishc, fedrcd, hc, hc_lingam
from fedcausal.lingam.cumulants import aggregate_and_compute_cumulants, compute_local_raw_stats


def _chain(seed: int, n: int = 6000, n_vars: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.laplace(0.0, 1.0, size=(n, n_vars))
    x = np.zeros((n, n_vars))
    x[:, 0] = e[:, 0]
    for j in range(1, n_vars):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return x


def _stats(x, n_clients=3):
    return aggregate_and_compute_cumulants(
        [compute_local_raw_stats(c, order=4) for c in np.array_split(x, n_clients, axis=0)]
    )


def test_fedishc_misorders_where_fedrcd_succeeds():
    """FedISHC's alpha-ratio deflation is the very thing FedRCD improves on.

    On this chain (0.8-coefficient Laplace-noise chain 0->1->2->3, seed 0, 3
    clients), FedISHC's iterative alpha-ratio deflation misorders variables 1
    and 2, while FedRCD's beta/OLS deflation recovers the chain exactly. This
    was verified directly against the unmodified original ``ishc()`` in
    ``$SRC/src/fedishc/core.py``, which returns the same (0, 2, 1, 3) on the
    identical data -- so this is not a porting bug, it is alpha-ratio
    deflation's known limitation under finite-sample noise, and it is
    precisely the comparison the PGM paper's baseline section is built on.
    Do not "fix" this by changing the seed or loosening the assertion.
    """
    stats = _stats(_chain(0))
    assert fedishc(stats).causal_order == (0, 2, 1, 3)
    assert fedrcd(stats).causal_order == (0, 1, 2, 3)


@pytest.mark.parametrize("solver", [fedhc])
def test_federated_baselines_recover_the_chain(solver):
    result = solver(_stats(_chain(0)))
    assert isinstance(result, RCDResult)
    assert result.causal_order == (0, 1, 2, 3)


@pytest.mark.parametrize("solver", [hc_lingam, hc])
def test_centralised_baselines_recover_the_chain(solver):
    result = solver(_chain(1))
    assert isinstance(result, RCDResult)
    assert result.causal_order == (0, 1, 2, 3)


def test_every_baseline_returns_a_square_b_of_the_right_size():
    x = _chain(2, n_vars=5)
    for result in (fedishc(_stats(x)), fedhc(_stats(x)), hc_lingam(x), hc(x)):
        assert result.b.shape == (5, 5)
