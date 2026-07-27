"""Network-recovery tests: GES.fit() against real Bayesian networks.

These pin *quality* (does the search recover something close to the true
network from forward-sampled data?), not exact Java equivalence -- that is
Task 8's job. Each threshold below was set by running the test once,
observing the actual value, and setting the bound just above it (the exact
observed value is recorded in a comment next to each threshold).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.io import forward_sample, load_xbif
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

NETWORKS = Path(__file__).resolve().parents[1] / "fixtures" / "networks"

# net -> (n_rows, max_smhd, max_shd_vs_true_cpdag)
#
# `max_shd_vs_true_cpdag` compares two independently-chosen consistent DAG
# extensions of (in general) equivalent CPDAGs -- `bn.dag`'s true CPDAG's own
# `to_dag()` and `GES.fit(ds)`'s own `to_dag()` -- so it can be noticeably
# larger than `max_smhd` even for a good recovery: a reversible edge oriented
# one way by one extension and the other way by the other counts as a full
# SHD reversal, even though both orientations are equally valid members of
# the same equivalence class. This is deterministic (both `to_dag()` calls
# use the same fixed sink-selection rule), not test flakiness, but it means
# `max_shd_vs_true_cpdag` measures "how different do the two chosen DAGs
# happen to look", not purely "how wrong is the learned skeleton/orientation".
FAST_NETS = {
    "asia": (5000, 2, 5),  # observed: smhd=1, shd=4
    "cancer": (5000, 1, 1),  # observed: smhd=0, shd=0
    "child": (5000, 1, 1),  # observed: smhd=0, shd=0
}
SLOW_NETS = {
    "alarm": (5000, 14, 12),  # observed: smhd=13, shd=11
    "insurance": (5000, 24, 24),  # observed: smhd=23, shd=23
}

# Nets where GES.fit recovers the true CPDAG *exactly* (observed shd=0,
# smhd=0 above). For these, PDAG.__eq__ gives a strictly stronger, noise-free
# assertion than max_shd_vs_true_cpdag: comparing the two CPDAGs directly
# rather than two independently-chosen consistent DAG extensions of them
# (see the module-level comment on max_shd_vs_true_cpdag's extension-choice
# noise). asia/alarm/insurance keep only the tolerant SHD/SMHD checks since
# exact recovery does not hold for them.
EXACT_RECOVERY_NETS = frozenset({"cancer", "child"})


def _run_recovery(net: str, n_rows: int, max_smhd: int, max_shd: int) -> None:
    xbif = NETWORKS / f"{net}.xbif"
    bn = load_xbif(str(xbif))
    ds = forward_sample(bn, n_rows, seed=0)
    # forward_sample keeps bn.dag's node order (see its docstring's "same
    # node order" guarantee) -- verified once against asia.xbif's actual
    # bn.names / ds.names before writing this test, so no name->index remap
    # is needed here; a future BN loader/sampler change that reordered
    # columns would need to rebuild this alignment.
    assert ds.names == tuple(bn.names)
    fitted = GES(BDeuScore(ds)).fit(ds)
    true_cpdag = dag_to_cpdag(bn.dag)
    if net in EXACT_RECOVERY_NETS:
        # Strictly stronger than the SHD/SMHD checks below: compares the two
        # CPDAGs directly via PDAG.__eq__, so a future regression that still
        # slips under max_smhd/max_shd_vs_true_cpdag's tolerance still fails
        # loudly here.
        assert true_cpdag == fitted
    result = fitted.to_dag()
    true_cpdag_dag = true_cpdag.to_dag()
    assert smhd(result, bn.dag) <= max_smhd
    assert shd(true_cpdag_dag, result) <= max_shd


@pytest.mark.parametrize("net,params", FAST_NETS.items(), ids=list(FAST_NETS))
def test_recovery_fast_networks(net: str, params: tuple[int, int, int]) -> None:
    n_rows, max_smhd, max_shd = params
    _run_recovery(net, n_rows, max_smhd, max_shd)


@pytest.mark.slow
@pytest.mark.parametrize("net,params", SLOW_NETS.items(), ids=list(SLOW_NETS))
def test_recovery_slow_networks(net: str, params: tuple[int, int, int]) -> None:
    n_rows, max_smhd, max_shd = params
    _run_recovery(net, n_rows, max_smhd, max_shd)
