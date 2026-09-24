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


def test_na_yx_and_t0() -> None:
    p = PDAG(4)
    p.add_undirected(0, 1)  # 0 - 1
    p.add_undirected(2, 1)  # 2 - 1
    p.add_directed(0, 2)  # 0 -> 2  (so 2 is adjacent to 0)
    # y = 1, x = 0: undirected neighbours of 1 are {0, 2}; adjacent-to-0 among them = {2}
    assert ops.na_yx(p, 0, 1) == frozenset({2})
    assert ops.t0_neighbors(p, 0, 1) == frozenset()  # 0 itself excluded; 2 is adjacent to 0


def test_is_clique() -> None:
    p = PDAG(3)
    p.add_undirected(0, 1)
    p.add_undirected(1, 2)
    assert ops.is_clique(p, frozenset({0, 1}))
    assert not ops.is_clique(p, frozenset({0, 1, 2}))  # 0-2 missing
    p.add_directed(0, 2)
    assert ops.is_clique(p, frozenset({0, 1, 2}))  # directed counts as adjacent


def test_is_clique_empty_and_singleton_are_trivially_cliques() -> None:
    p = PDAG(3)  # no edges at all
    assert ops.is_clique(p, frozenset())
    assert ops.is_clique(p, frozenset({1}))


def test_semidirected_blocked() -> None:
    p = PDAG(3)
    p.add_undirected(1, 2)  # y=1 - 2
    p.add_directed(2, 0)  # 2 -> x=0 ; path 1 - 2 -> 0 is semi-directed
    assert not ops.is_semidirected_blocked(p, 1, 0, frozenset())  # unblocked
    assert ops.is_semidirected_blocked(p, 1, 0, frozenset({2}))  # 2 blocks it


def test_semidirected_blocked_route_against_arrowhead_is_not_a_path() -> None:
    # The only route from 1 to 0 runs against the arrowhead of 0 -> 2.
    p = PDAG(3)
    p.add_undirected(1, 2)  # y=1 - 2
    p.add_directed(0, 2)  # 0 -> 2 ; 2 cannot be left towards 0
    assert ops.is_semidirected_blocked(p, 1, 0, frozenset())


def test_semidirected_blocked_directed_edge_leaving_source_is_a_path() -> None:
    # The path 1 -> 2 - 0 starts with a directed edge.
    p = PDAG(3)
    p.add_directed(1, 2)  # y=1 -> 2
    p.add_undirected(2, 0)  # 2 - x=0
    assert not ops.is_semidirected_blocked(p, 1, 0, frozenset())


def test_semidirected_blocked_x_in_cond_is_blocked() -> None:
    # The only path reaches x, which is in cond, so it is blocked.
    p = PDAG(3)
    p.add_undirected(1, 2)
    p.add_directed(2, 0)
    assert ops.is_semidirected_blocked(p, 1, 0, frozenset({0}))  # x in cond -> blocked


def test_valid_insert_accepts_valid_case() -> None:
    p = PDAG(3)  # no edges: NaYX | T is empty (trivial clique), no path from y to x at all
    assert ops.valid_insert(p, x=0, y=1, t=frozenset(), na=frozenset())


def test_valid_insert_rejects_non_clique() -> None:
    p = PDAG(4)
    # na={2}, t={3}; 2 and 3 are not adjacent, so na | t = {2, 3} is not a clique.
    p.add_undirected(2, 1)
    p.add_undirected(3, 1)
    assert not ops.valid_insert(p, x=0, y=1, t=frozenset({3}), na=frozenset({2}))


def test_valid_insert_rejects_unblocked_semidirected_path() -> None:
    p = PDAG(3)
    p.add_undirected(1, 2)  # y=1 - 2
    p.add_directed(2, 0)  # 2 -> x=0 ; path 1 - 2 -> 0 is semi-directed and unblocked by {}
    assert not ops.valid_insert(p, x=0, y=1, t=frozenset(), na=frozenset())


def test_insert_eval_matches_local_score_difference() -> None:
    s = _toy_score()
    p = PDAG(4)
    na, t = frozenset(), frozenset()
    got = ops.insert_eval(s, p, 0, 1, t, na)  # add 0 -> 1 on empty graph
    exp = s.local_score(1, (0,)) - s.local_score(1, ())
    assert math.isclose(got, exp, rel_tol=1e-12)


def test_insert_eval_combines_parents_na_and_t() -> None:
    # Non-empty Pa(y), na, and t together, so this fails if the implementation
    # drops pdag.parents(y), uses parents(x) instead, or forgets `| t`.
    s = _toy_score(p=5)
    p = PDAG(5)
    p.add_directed(3, 4)  # existing Pa(y=4) = {3}
    na = frozenset({1})
    t = frozenset({2})
    got = ops.insert_eval(s, p, x=0, y=4, t=t, na=na)
    # Explicitly constructed expected parent sets: na | t | Pa(y) = {1, 2, 3}
    exp = s.local_score(4, (0, 1, 2, 3)) - s.local_score(4, (1, 2, 3))
    assert math.isclose(got, exp, rel_tol=1e-12)


def test_apply_insert_orients_t() -> None:
    p = PDAG(3)
    p.add_undirected(2, 1)  # t candidate: 2 - 1
    ops.apply_insert(p, 0, 1, frozenset({2}))
    assert p.has_directed(0, 1)  # x -> y
    assert p.has_directed(2, 1)  # t -> y
