"""GES recovers the network the data was sampled from.

A quality check, not an equivalence check: each bound is loose enough that a
different but equally good search passes, and tight enough that one which
stops finding the structure does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.io import forward_sample, load_xbif
from fedcausal.metrics import smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

NETWORKS = Path(__file__).resolve().parents[1] / "fixtures" / "networks"

# net -> worst acceptable SMHD against the true network, from 5000 sampled rows.
# SMHD rather than SHD because it compares moral graphs, which every member of
# an equivalence class shares, so the arbitrary DAG a CPDAG is extended to
# cannot move it.
FAST_NETS = {"asia": 2, "cancer": 2, "child": 1}
SLOW_NETS = {"alarm": 14, "insurance": 24}

# On `child` GES recovers the true equivalence class exactly, which is a
# stronger check than any SMHD bound.
EXACT_RECOVERY_NETS = frozenset({"child"})


def _run_recovery(net: str, max_smhd: int) -> None:
    bn = load_xbif(NETWORKS / f"{net}.xbif")
    ds = forward_sample(bn, 5000, seed=0)
    fitted = GES(BDeuScore(ds)).fit(ds)

    if net in EXACT_RECOVERY_NETS:
        assert fitted == dag_to_cpdag(bn.dag)
    assert smhd(fitted.to_dag(), bn.dag) <= max_smhd


@pytest.mark.parametrize(("net", "max_smhd"), FAST_NETS.items(), ids=list(FAST_NETS))
def test_recovery_fast_networks(net: str, max_smhd: int) -> None:
    _run_recovery(net, max_smhd)


@pytest.mark.slow
@pytest.mark.parametrize(("net", "max_smhd"), SLOW_NETS.items(), ids=list(SLOW_NETS))
def test_recovery_slow_networks(net: str, max_smhd: int) -> None:
    _run_recovery(net, max_smhd)
