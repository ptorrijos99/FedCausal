import numpy as np

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG, PDAG
from fedcausal.scores import BDeuScore
from fedcausal.search._candidates import subsets
from fedcausal.search.ges import _best_insert, run_fes


def testsubsets_order_two_elements() -> None:
    assert list(subsets([1, 2])) == [
        frozenset(),
        frozenset({1}),
        frozenset({2}),
        frozenset({1, 2}),
    ]


def testsubsets_order_three_elements() -> None:
    # Pins the full ascending-size-then-lexicographic contract: two elements
    # alone can't distinguish "ascending size" from some other tie-break
    # among same-size subsets, since there is only one subset of size 2 (the
    # full set). Three elements have three distinct size-2 subsets, so this
    # catches an implementation that orders sizes correctly but shuffles
    # same-size subsets (e.g. by sorting subset contents instead of preserving
    # itertools.combinations' source order).
    assert list(subsets([1, 2, 3])) == [
        frozenset(),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
        frozenset({1, 2, 3}),
    ]


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


def test_fes_is_deterministic_across_repeated_runs() -> None:
    # Comparing two runs against *each other* in the same process does not
    # actually pin scan/tie-break order on this fixture: both runs execute
    # identical construction sequences over small ints, and hash(int) == int,
    # so PYTHONHASHSEED perturbs nothing here. An implementation that
    # iterates frozenset candidates directly (instead of via sorted(...))
    # would still traverse them in the same order both runs, within and
    # across processes, and would still pass a cpdag1 == cpdag2 check alone.
    # The assertions below pin the *exact* expected CPDAG instead, so any
    # change to candidate scan order, tie-breaking, or subset order changes
    # the result and fails here -- this is a change-detector for those three
    # things, not a real "is it deterministic" test. Expected value obtained
    # by running the current implementation and eyeballed for plausibility:
    # A and B are strongly dependent (see _dep_dataset), so FES connects them
    # (undirected, since nothing forces an orientation); C and D are
    # independent of everything, so no other edge appears.
    ds = _dep_dataset()
    score = BDeuScore(ds)
    cpdag1, flag1 = run_fes(score, PDAG(4))
    cpdag2, flag2 = run_fes(score, PDAG(4))
    assert flag1 is True
    assert flag2 is True
    assert cpdag1 == cpdag2
    assert cpdag1.directed_edges() == []
    assert cpdag1.undirected_edges() == [(0, 1)]


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


class _TiedFakeScore:
    """A hand-built score with three exactly-tied best Inserts on an empty PDAG(3).

    ``Insert(x=0, y=1, T=empty)``, ``Insert(x=0, y=2, T=empty)``, and
    ``Insert(x=1, y=0, T=empty)`` all have delta 10.0; every other candidate
    has delta 0.0 (not an improvement). The FES scan order is `x` outer, `y`
    inner, both ascending, so among these three tied candidates `(x=0, y=1)`
    is considered strictly first.

    This single fixture pins two independent things at once, both required
    for the assertion below to hold:

    * Tie-break rule: a correct first-seen-on-tie implementation (replace the
      incumbent only on strictly greater delta) keeps `(x=0, y=1)` once it
      becomes the incumbent, since the other two candidates only ever tie
      it, never beat it. An implementation that instead replaces on
      `delta >= best_delta` would keep overwriting the incumbent as each
      later tie arrives and, even under the correct scan order, would end up
      on the *last*-scanned tied candidate, `(x=1, y=0)`, not `(x=0, y=1)`.
    * Scan order: `(x=1, y=0)` is what makes this fixture actually
      distinguish `x` outer / `y` inner from `y` outer / `x` inner -- a
      two-candidate version of this fixture (just `(x=0, y=1)` and
      `(x=0, y=2)`) does not, since both scan orders agree on it. Under a
      (wrong) `y` outer / `x` inner scan, even paired with the correct strict
      tie-break, `(x=1, y=0)` is the *first* tied candidate encountered (at
      `y=0`, before `y=1` or `y=2` are reached), so it becomes the incumbent
      and is never displaced by the later ties, wrongly returning
      `(x=1, y=0)` instead of `(x=0, y=1)`.
    """

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        if node in (1, 2) and parents == (0,):
            return 10.0
        if node == 0 and parents == (1,):
            return 10.0
        return 0.0


def test_best_insert_keeps_first_seen_candidate_on_exact_tie() -> None:
    fake_score = _TiedFakeScore()
    pdag = PDAG(3)
    best = _best_insert(fake_score, pdag, None)
    assert best == (10.0, 0, 1, frozenset())


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
