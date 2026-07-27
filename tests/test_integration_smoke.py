import math
from pathlib import Path

from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.io import forward_sample, load_xbif
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ASIA_XBIF = FIXTURES / "networks" / "asia.xbif"


def test_phase1_stack_end_to_end() -> None:
    # load_xbif -> forward_sample -> BDeuScore -> score_dag
    bn = load_xbif(ASIA_XBIF)
    dataset = forward_sample(bn, 2000, seed=0)
    score = BDeuScore(dataset)
    total = score.score_dag(bn.dag)
    assert math.isfinite(total)
    assert total < 0

    # dag_to_cpdag round-trips through to_dag() into the same CPDAG.
    cpdag = dag_to_cpdag(bn.dag)
    roundtrip = dag_to_cpdag(cpdag.to_dag())
    assert roundtrip == cpdag

    # shd/smhd of a DAG against itself are both zero.
    assert shd(bn.dag, bn.dag) == 0
    assert smhd(bn.dag, bn.dag) == 0
