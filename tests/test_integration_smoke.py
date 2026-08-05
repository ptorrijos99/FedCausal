import math

from fedcausal.datasets import load_asia
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.io import forward_sample
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore


def test_phase1_stack_end_to_end() -> None:
    # load_asia -> forward_sample -> BDeuScore -> score_dag
    bn = load_asia().network
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
