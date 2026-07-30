from fedcausal.fedges.convergence import (
    IterationEquality,
    ModelEquality,
    MultipleConvergence,
    NoneConvergence,
    ScoreImprovement,
)
from fedcausal.fedges.model import BNModel
from fedcausal.graph import DAG


def _m(edges, score=0.0, n=3):
    return BNModel(DAG.from_edges(n, edges), score=score)


def test_none_never_converges():
    c = NoneConvergence()
    assert c.check_convergence([_m([(0, 1)])]) is False
    assert c.check_convergence([_m([(0, 1)])]) is False


def test_score_improvement_converges_when_mean_does_not_improve():
    c = ScoreImprovement()
    assert c.check_convergence([_m([], score=-100.0)]) is False  # first: -inf -> improves
    assert c.check_convergence([_m([], score=-90.0)]) is False  # improved
    assert c.check_convergence([_m([], score=-90.0)]) is True  # no improvement -> converged
    assert c.check_convergence([_m([], score=-95.0)]) is True  # got worse -> converged


def test_model_equality_converges_on_repeat():
    c = ModelEquality()
    a = [_m([(0, 1)]), _m([(1, 2)])]
    assert c.check_convergence(a) is False  # first sighting
    assert c.check_convergence(a) is True  # both seen before
    b = [_m([(0, 1)]), _m([(0, 2)])]  # client 1 changed
    assert c.check_convergence(b) is False


def test_iteration_equality_uses_score_when_nonzero():
    c = IterationEquality(3)
    assert c.check_convergence([_m([(0, 1)], score=-10.0)]) is False
    assert c.check_convergence([_m([(1, 2)], score=-10.0004)]) is True  # score within 0.001


def test_iteration_equality_uses_models_when_score_zero():
    c = IterationEquality(3)
    assert c.check_convergence([_m([(0, 1)], score=0.0)]) is False
    assert c.check_convergence([_m([(0, 1)], score=0.0)]) is True  # same model
    assert c.check_convergence([_m([(1, 2)], score=0.0)]) is False


def test_multiple_is_or_and_calls_all():
    c = MultipleConvergence([ScoreImprovement(), ModelEquality()])
    a = [_m([(0, 1)], score=-10.0)]
    assert c.check_convergence(a) is False  # neither converged (first round)
    # ScoreImprovement: no improvement (same mean) -> converged
    assert c.check_convergence(a) is True
