import math

import numpy as np

from fedcausal.data import DiscreteDataset
from fedcausal.graph import PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search import gesops as ops


def _toy_score(seed: int = 0, n: int = 60, p: int = 4) -> BDeuScore:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2, size=(n, p)).astype(np.int32)
    ds = DiscreteDataset.from_arrays(
        values, np.full(p, 2, np.int32), list("ABCDEFGH"[:p]), [["0", "1"]] * p
    )
    return BDeuScore(ds)


def test_valid_delete_clique_rule() -> None:
    p = PDAG(4)
    p.add_directed(0, 1)  # x=0 -> y=1
    p.add_undirected(2, 1)  # 2 - 1, 2 adjacent to 0 -> in NaYX
    p.add_directed(0, 2)
    na = ops.na_yx(p, 0, 1)  # {2}
    assert ops.valid_delete(p, 0, 1, frozenset(), na)  # na\{} = {2} is a clique (singleton)


def _non_clique_na_graph() -> PDAG:
    # NaYX(0, 1) = {2, 3}, and 2 and 3 are not adjacent.
    p = PDAG(4)
    p.add_directed(0, 1)  # x=0 -> y=1, the edge under consideration
    p.add_undirected(2, 1)
    p.add_undirected(3, 1)
    p.add_directed(0, 2)  # 2 adjacent to x -> in NaYX
    p.add_directed(0, 3)  # 3 adjacent to x -> in NaYX
    return p


def test_valid_delete_rejects_non_clique() -> None:
    # na\h = {2, 3}; 2 and 3 are not adjacent, so this must be rejected.
    p = _non_clique_na_graph()
    na = ops.na_yx(p, 0, 1)
    assert na == frozenset({2, 3})
    assert not ops.valid_delete(p, 0, 1, frozenset(), na)


def test_valid_delete_h_removes_the_non_clique_member() -> None:
    # Only removing 3 through h makes the rest a clique.
    p = _non_clique_na_graph()
    na = ops.na_yx(p, 0, 1)
    assert na == frozenset({2, 3})
    h = frozenset({3})
    assert ops.valid_delete(p, 0, 1, h, na)


def test_delete_eval_matches_local_score_difference() -> None:
    s = _toy_score()
    p = PDAG(4)
    p.add_directed(0, 1)  # y=1 has parent 0
    na = frozenset()
    got = ops.delete_eval(s, p, 0, 1, frozenset(), na)  # remove 0 -> 1
    exp = s.local_score(1, ()) - s.local_score(1, (0,))
    assert math.isclose(got, exp, rel_tol=1e-12)


def test_delete_eval_combines_na_h_and_parents() -> None:
    # NaYX, H and Pa(y) all non-empty.
    s = _toy_score(p=6)
    p = PDAG(6)
    p.add_directed(0, 1)  # x=0 -> y=1 (the edge being deleted); Pa(y) = {0, 4}
    p.add_directed(4, 1)  # another, unrelated parent of y
    na = frozenset({2, 3})
    h = frozenset({3})
    got = ops.delete_eval(s, p, x=0, y=1, h=h, na=na)
    # before = (na - h) | (Pa(y) - {x}) = {2} | {4} = {2, 4}
    # after = before | {x} = {0, 2, 4}
    exp = s.local_score(1, (2, 4)) - s.local_score(1, (0, 2, 4))
    assert math.isclose(got, exp, rel_tol=1e-12)


def test_apply_delete_removes_and_orients_h() -> None:
    p = PDAG(3)
    p.add_directed(0, 1)  # x=0 -> y=1
    p.add_undirected(1, 2)  # y=1 - 2 ; 2 in H
    p.add_undirected(0, 2)  # x=0 - 2
    ops.apply_delete(p, 0, 1, frozenset({2}))
    assert not (p.has_directed(0, 1) or p.has_undirected(0, 1))  # x-y edge gone
    assert p.has_directed(1, 2)  # y -> h
    assert p.has_directed(0, 2)  # x -> h


def test_apply_delete_removes_undirected_xy_edge() -> None:
    p = PDAG(3)
    p.add_undirected(0, 1)  # x=0 - y=1, undirected
    p.add_undirected(1, 2)  # y - 2 ; 2 in H
    p.add_undirected(0, 2)  # x - 2, undirected
    ops.apply_delete(p, 0, 1, frozenset({2}))
    assert not (p.has_directed(0, 1) or p.has_directed(1, 0) or p.has_undirected(0, 1))
    assert p.has_directed(1, 2)  # y -> h
    assert p.has_directed(0, 2)  # x -> h


def test_apply_delete_handles_multiple_h_members_independently() -> None:
    p = PDAG(4)
    p.add_directed(0, 1)  # x=0 -> y=1, the edge being deleted
    p.add_undirected(1, 2)  # y - w1=2, undirected
    p.add_undirected(0, 2)  # x - w1, undirected
    p.add_undirected(1, 3)  # y - w2=3, undirected
    p.add_directed(3, 0)  # w2=3 -> x=0, already directed the other way
    ops.apply_delete(p, 0, 1, frozenset({2, 3}))
    assert not (p.has_directed(0, 1) or p.has_undirected(0, 1))  # x-y edge gone
    assert p.has_directed(1, 2)  # y -> w1
    assert p.has_directed(0, 2)  # x -> w1
    assert p.has_directed(1, 3)  # y -> w2
    assert p.has_directed(3, 0)  # w2 -> x left as it was
    assert not p.has_directed(0, 3)


def test_apply_delete_keeps_a_directed_x_w_edge() -> None:
    p = PDAG(3)
    p.add_directed(0, 1)  # x=0 -> y=1, the edge being deleted
    p.add_directed(2, 0)  # w=2 -> x=0, already directed the other way
    p.add_undirected(1, 2)  # y - w undirected, so w is a legitimate H member
    ops.apply_delete(p, 0, 1, frozenset({2}))
    assert p.has_directed(2, 0)  # w -> x is left untouched
    assert not p.has_directed(0, 2)
    assert p.has_directed(1, 2)  # y -> w
