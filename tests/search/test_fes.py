import numpy as np

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG, PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search.ges import _best_insert, run_fes


def _dep_dataset(n: int = 4000, seed: int = 0) -> DiscreteDataset:
    # A -> B deterministic-ish: B copies A with 10% noise; C, D independent.
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = np.where(rng.random(n) < 0.9, a, 1 - a)
    c = rng.integers(0, 2, n)
    d = rng.integers(0, 2, n)
    values = np.column_stack([a, b, c, d]).astype(np.int32)
    return DiscreteDataset.from_arrays(
        values, np.full(4, 2, np.int32), list("ABCD"), [["0", "1"]] * 4
    )


def test_fes_recovers_ab_adjacency() -> None:
    ds = _dep_dataset()
    score = BDeuScore(ds)
    cpdag, flag = run_fes(score, PDAG(4))
    assert flag is True
    # A and B are strongly dependent -> FES must connect them (either orientation / undirected).
    assert 1 in cpdag.adjacent(0)
    # C and D are independent of everything -> no spurious C-D edge.
    assert 3 not in cpdag.adjacent(2)


def test_fes_monotone_score_increase() -> None:
    ds = _dep_dataset()
    score = BDeuScore(ds)
    empty = PDAG(4)
    cpdag, _ = run_fes(score, empty)
    s_empty = score.score_dag(DAG(4))
    s_final = score.score_dag(cpdag.to_dag())
    # Strict `>`, not `>=`: `>=` would also pass for a no-op implementation
    # that returns the empty graph unchanged. Strictness is safe here (not a
    # risk of spurious failure on a legitimately flat score surface) because
    # the sibling test `test_fes_recovers_ab_adjacency` already asserts
    # `flag is True` on this exact fixture, so FES demonstrably inserted at
    # least one edge here.
    assert s_final > s_empty


def test_fes_does_not_mutate_input_pdag() -> None:
    # run_fes must search on an internal copy; a caller passing in a PDAG
    # (e.g. reused across search restarts) must see it untouched.
    ds = _dep_dataset()
    score = BDeuScore(ds)
    start = PDAG(4)
    run_fes(score, start)
    assert start.directed_edges() == []
    assert start.undirected_edges() == []


def test_run_fes_no_improvement_returns_unmodified_graph() -> None:
    # Four mutually independent columns built from a balanced full factorial
    # (every one of the 16 combinations of 4 binary variables repeated the
    # same number of times), so the empirical joint distribution is *exactly*
    # the product of uniform marginals, not merely approximately independent.
    # No Insert should improve the BDeu score of the empty graph over such
    # data, so run_fes must report changed=False and hand back a CPDAG with
    # no edges at all, structurally equal to the input.
    reps = 200
    combos = np.array([[i >> b & 1 for b in range(4)] for i in range(16)], dtype=np.int32)
    values = np.repeat(combos, reps, axis=0)
    ds = DiscreteDataset.from_arrays(
        values, np.full(4, 2, np.int32), list("ABCD"), [["0", "1"]] * 4
    )
    score = BDeuScore(ds)
    start = PDAG(4)
    cpdag, changed = run_fes(score, start)
    assert changed is False
    assert cpdag.directed_edges() == []
    assert cpdag.undirected_edges() == []


def test_run_fes_restricted_candidate_edges_omits_edge_unrestricted_would_add() -> None:
    # A and B are strongly dependent (see _dep_dataset), so unrestricted FES
    # connects them (test_fes_recovers_ab_adjacency / the determinism test
    # above both pin the resulting undirected (0, 1) edge). Restricting
    # candidate_edges to every ordered pair *except* (0, 1) and (1, 0) must
    # therefore prevent that edge from ever being scanned, leaving the graph
    # empty even though the data would otherwise justify the edge.
    ds = _dep_dataset()
    score = BDeuScore(ds)
    n = 4
    restricted = frozenset(
        (x, y) for x in range(n) for y in range(n) if x != y and (x, y) not in {(0, 1), (1, 0)}
    )
    cpdag, changed = run_fes(score, PDAG(4), restricted)
    assert changed is False
    assert cpdag.directed_edges() == []
    assert cpdag.undirected_edges() == []


class _DirectedInsertScore:
    """A fake score where Insert(x=0, y=2, T=empty) improves and everything else is flat."""

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        if node == 2 and parents == (0,):
            return 10.0
        return 0.0


def test_run_fes_candidate_edges_pair_order_is_directional() -> None:
    # candidate_edges holds *ordered* pairs: (x, y) means "consider inserting
    # x -> y", not "this undirected edge is a candidate". A caller who builds
    # an undirected candidate set (e.g. from a federated fusion step) and
    # forgets to include both (x, y) and (y, x) silently gets a degraded
    # search: frozenset({(0, 2)}) allows the beneficial insert below, while
    # frozenset({(2, 0)}) -- the same two nodes, opposite order -- allows
    # nothing at all, even though the score only depends on the edge
    # existing, not on which literal ordered pair the caller happened to
    # list.
    score = _DirectedInsertScore()
    cpdag_allowed, changed_allowed = run_fes(score, PDAG(3), frozenset({(0, 2)}))
    assert changed_allowed is True
    assert 2 in cpdag_allowed.adjacent(0)

    cpdag_blocked, changed_blocked = run_fes(score, PDAG(3), frozenset({(2, 0)}))
    assert changed_blocked is False
    assert cpdag_blocked.directed_edges() == []
    assert cpdag_blocked.undirected_edges() == []


class _NonEmptyTScore:
    """A fake score where Insert(x=0, y=1, T={2}) strictly beats Insert(x=0, y=1, T=empty).

    Meant to be paired with a starting `PDAG(3)` that already has an
    undirected edge `1 - 2` (so `2` is a candidate `T`-neighbour of `y=1` not
    yet adjacent to `x=0`, i.e. `2` is in `t0_neighbors(pdag, 0, 1)`).
    Conditioning on `2` in addition to `0` is worth strictly more than
    conditioning on `0` alone in this table, so the `T={2}` Insert beats the
    `T=empty` Insert on the same `(x, y)` pair.
    """

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        table = {
            (1, ()): 0.0,
            (1, (0,)): 1.0,
            (1, (2,)): 0.0,
            (1, (0, 2)): 5.0,
        }
        return table.get((node, parents), 0.0)


def test_best_insert_selects_non_empty_t_when_it_wins() -> None:
    # Regression coverage for the non-empty-T path through valid_insert /
    # insert_eval / apply_insert: no other test in this file exercises it,
    # since every other fixture either starts from an empty PDAG (no
    # candidate T exists at all) or has its winning Insert use T=empty
    # anyway. Asserting directly on _best_insert's returned t pins that a
    # non-empty T can in fact be selected when it scores higher.
    pdag = PDAG(3)
    pdag.add_undirected(1, 2)
    score = _NonEmptyTScore()
    best = _best_insert(score, pdag, None)
    assert best == (5.0, 0, 1, frozenset({2}))


def test_run_fes_non_empty_t_orients_t_neighbor_into_y() -> None:
    # Same fixture as above, run through the full run_fes loop instead of
    # calling _best_insert directly: apply_insert force-flips every node in T
    # towards y, so T={2} being selected must show up as a directed edge
    # 2 -> 1 in the resulting CPDAG. It stays directed (not re-undirected by
    # re-canonicalization) because 0 and 2 are non-adjacent, making
    # 0 -> 1 <- 2 an unshielded collider, which Meek orientation preserves.
    pdag = PDAG(3)
    pdag.add_undirected(1, 2)
    score = _NonEmptyTScore()
    cpdag, changed = run_fes(score, pdag)
    assert changed is True
    assert cpdag.directed_edges() == [(0, 1), (2, 1)]
