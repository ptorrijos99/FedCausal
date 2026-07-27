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
    # y=1, x=0: 2 and 3 are both undirected neighbours of y and both adjacent
    # to x (via directed edges), so na_yx(p, 0, 1) genuinely computes {2, 3};
    # 2 and 3 are not adjacent to each other, so that na is not a clique.
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
    # Bug this catches: an implementation that ignores `h` (e.g. checks
    # `is_clique(pdag, na)` instead of `is_clique(pdag, na - h)`) would still
    # reject this case, since na={2,3} alone is not a clique; only removing 3
    # via h makes na\h={2} a (trivial) clique.
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
    # Non-empty NaYX, H, and Pa(y) together, with an explicitly constructed
    # expected parent set. This fails if the implementation drops
    # pdag.parents(y), uses pdag.parents(x) instead, or forgets to subtract
    # `h` from `na`.
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
    # Covers the other edge kind for the x-y edge being deleted: an
    # undirected x - y edge (Java's `removeEdges(x, y)` handles either kind).
    p = PDAG(3)
    p.add_undirected(0, 1)  # x=0 - y=1, undirected
    p.add_undirected(1, 2)  # y - 2 ; 2 in H
    p.add_undirected(0, 2)  # x - 2, undirected (2 is adjacent to x, per the H/NaYX contract)
    ops.apply_delete(p, 0, 1, frozenset({2}))
    assert not (p.has_directed(0, 1) or p.has_directed(1, 0) or p.has_undirected(0, 1))
    assert p.has_directed(1, 2)  # y -> h
    assert p.has_directed(0, 2)  # x -> h


def test_apply_delete_force_flips_wrongly_directed_h_edge_on_y_side() -> None:
    # Regression pinning Java `GESThread.delete`'s unconditional force-flip
    # on the y side (`removeEdge(y, aSubset); addDirectedEdge(y, aSubset)`
    # for every node in the subset, with no guard at all). If apply_delete
    # instead only oriented `w` when the y-w edge was already undirected
    # (mirroring the naive, non-force-flipping reading), this pre-existing
    # w -> y edge (the wrong direction) would be left untouched instead of
    # being replaced by y -> w.
    p = PDAG(3)
    p.add_directed(0, 1)  # x=0 -> y=1, the edge being deleted
    p.add_directed(2, 1)  # w=2 -> y=1 : wrong direction for an H node
    p.add_undirected(0, 2)  # x - w undirected, so the x-side edit is unaffected
    ops.apply_delete(p, 0, 1, frozenset({2}))
    assert p.has_directed(1, 2)  # y -> w, force-flipped
    assert not p.has_directed(2, 1)  # old w -> y direction is gone


def test_apply_delete_handles_multiple_h_members_independently() -> None:
    # |H| > 1: pins that each w in h is processed independently -- the
    # y-side force-flip and the x-side guard are each evaluated per member,
    # not conflated or short-circuited across members -- and that iterating
    # in sorted(h) order does not change the outcome for either member.
    p = PDAG(4)
    p.add_directed(0, 1)  # x=0 -> y=1, the edge being deleted
    p.add_undirected(1, 2)  # y - w1=2, undirected
    p.add_undirected(0, 2)  # x - w1, undirected -> x-side force-flips to x -> w1
    p.add_undirected(1, 3)  # y - w2=3, undirected
    p.add_directed(3, 0)  # w2=3 -> x=0, already directed the other way
    ops.apply_delete(p, 0, 1, frozenset({2, 3}))
    assert not (p.has_directed(0, 1) or p.has_undirected(0, 1))  # x-y edge gone
    assert p.has_directed(1, 2)  # y -> w1
    assert p.has_directed(0, 2)  # x -> w1, force-flipped (was undirected)
    assert p.has_directed(1, 3)  # y -> w2
    assert p.has_directed(3, 0)  # w2 -> x left untouched (guard fired)
    assert not p.has_directed(0, 3)


def test_apply_delete_skips_x_side_when_already_directed_the_other_way() -> None:
    # Regression pinning Java `delete`'s guard on the x side
    # (`if (!isParentOf(aSubset, x) && !isParentOf(x, aSubset))`): when x and
    # w already have a directed edge between them (in *either* direction),
    # Java leaves it untouched rather than force-flipping it towards x -> w.
    # An implementation that copies apply_insert's unconditional force-flip
    # pattern onto the x side would wrongly turn this pre-existing w -> x
    # edge into x -> w.
    p = PDAG(3)
    p.add_directed(0, 1)  # x=0 -> y=1, the edge being deleted
    p.add_directed(2, 0)  # w=2 -> x=0, already directed the other way
    p.add_undirected(1, 2)  # y - w undirected, so w is a legitimate H member
    ops.apply_delete(p, 0, 1, frozenset({2}))
    assert p.has_directed(2, 0)  # w -> x is left untouched
    assert not p.has_directed(0, 2)
    assert p.has_directed(1, 2)  # y -> w, force-flipped (unconditional on the y side)
