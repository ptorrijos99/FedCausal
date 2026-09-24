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
    # Every combination of four binary values equally often: exactly independent.
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
    # The data justify A - B, but it is not a candidate.
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
    # Candidates are ordered pairs: only (0, 2) allows the insert this score rewards.
    score = _DirectedInsertScore()
    cpdag_allowed, changed_allowed = run_fes(score, PDAG(3), frozenset({(0, 2)}))
    assert changed_allowed is True
    assert 2 in cpdag_allowed.adjacent(0)

    cpdag_blocked, changed_blocked = run_fes(score, PDAG(3), frozenset({(2, 0)}))
    assert changed_blocked is False
    assert cpdag_blocked.directed_edges() == []
    assert cpdag_blocked.undirected_edges() == []


class _NonEmptyTScore:
    """A score on which Insert(0, 1) is best with T = {2}, given the edge 1 - 2."""

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
    pdag = PDAG(3)
    pdag.add_undirected(1, 2)
    score = _NonEmptyTScore()
    best = _best_insert(score, pdag, None)
    assert best == (5.0, 0, 1, frozenset({2}))


def test_run_fes_non_empty_t_orients_t_neighbor_into_y() -> None:
    # T = {2} orients 2 -> 1, and the collider 0 -> 1 <- 2 keeps it directed.
    pdag = PDAG(3)
    pdag.add_undirected(1, 2)
    score = _NonEmptyTScore()
    cpdag, changed = run_fes(score, pdag)
    assert changed is True
    assert cpdag.directed_edges() == [(0, 1), (2, 1)]
