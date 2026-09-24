import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG, PDAG
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.scores import BDeuScore
from fedcausal.search import GES
from fedcausal.search.ges import run_bes, run_fes


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


def _dummy_dataset(n_vars: int) -> DiscreteDataset:
    # Only n_vars is read when the score is a fake.
    values = np.zeros((1, n_vars), dtype=np.int32)
    cardinalities = np.full(n_vars, 2, np.int32)
    names = [chr(ord("A") + i) for i in range(n_vars)]
    categories = [["0", "1"]] * n_vars
    return DiscreteDataset.from_arrays(values, cardinalities, names, categories)


def test_fit_returns_pdag_with_expected_adjacency() -> None:
    ds = _dep_dataset()
    cpdag = GES(BDeuScore(ds)).fit(ds)
    assert isinstance(cpdag, PDAG)
    assert 1 in cpdag.adjacent(0)
    assert 3 not in cpdag.adjacent(2)


def test_fit_score_at_least_empty_graph() -> None:
    ds = _dep_dataset()
    score = BDeuScore(ds)
    cpdag = GES(score).fit(ds)
    assert score.score_dag(cpdag.to_dag()) >= score.score_dag(DAG(ds.n_vars))


def test_fit_equals_fes_then_bes() -> None:
    ds = _dep_dataset()
    g = GES(BDeuScore(ds))
    combined = g.fit(ds)
    fes = g.fes_only(ds)
    both = g.bes_only(ds, initial_graph=fes.to_dag())
    assert combined == both


class _OvershootScore:
    """A score on which the forward phase adds an edge the backward phase then removes.

    The forward phase ends on the triangle 0-1, 0-2, 1-2; deleting 1-2 is then
    worth 10, more than any insert was.
    """

    _TABLE = {
        (1, (0,)): 10.0,
        (2, (1,)): 8.0,
        (2, (0, 1)): 9.0,
    }

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        return self._TABLE.get((node, parents), 0.0)


def test_fit_composes_fes_then_bes_not_fes_only_or_reversed() -> None:
    # Forward only, or backward before forward, would both keep the triangle.
    ds = _dummy_dataset(3)
    score = _OvershootScore()
    g = GES(score)

    combined = g.fit(ds)
    fes_only_result = g.fes_only(ds)
    bes_then_fes, _ = run_fes(score, run_bes(score, PDAG(3))[0])

    assert fes_only_result.undirected_edges() == [(0, 1), (0, 2), (1, 2)]
    assert bes_then_fes.undirected_edges() == [(0, 1), (0, 2), (1, 2)]
    assert combined.undirected_edges() == [(0, 1), (0, 2)]
    assert combined.directed_edges() == []
    assert combined != fes_only_result
    assert combined != bes_then_fes


def test_candidate_edges_restrict_fes() -> None:
    ds = _dep_dataset()
    cpdag = GES(BDeuScore(ds)).fes_only(ds, candidate_edges=frozenset({(0, 1)}))
    assert set(cpdag.adjacent(0)) <= {1}


class _DirectedInsertScore:
    """A fake score where only Insert(x=0, y=2, T=empty) improves."""

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        if node == 2 and parents == (0,):
            return 10.0
        return 0.0


def test_fes_only_candidate_edges_pair_order_is_directional() -> None:
    # Candidates are ordered pairs: only (0, 2) allows the insert this score rewards.
    ds = _dummy_dataset(3)
    score = _DirectedInsertScore()
    g = GES(score)

    allowed = g.fes_only(ds, candidate_edges=frozenset({(0, 2)}))
    assert 2 in allowed.adjacent(0)

    blocked = g.fes_only(ds, candidate_edges=frozenset({(2, 0)}))
    assert blocked.directed_edges() == []
    assert blocked.undirected_edges() == []


class _DirectedDeleteScore:
    """A fake score where only Delete(x=0, y=1, H=empty) improves."""

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        table = {
            (1, ()): 5.0,
            (1, (0,)): 0.0,
        }
        return table.get((node, parents), 0.0)


def test_bes_only_candidate_edges_pair_order_is_directional() -> None:
    ds = _dummy_dataset(2)
    score = _DirectedDeleteScore()
    g = GES(score)
    start_pdag = PDAG(2)
    start_pdag.add_undirected(0, 1)
    start = start_pdag.to_dag()

    allowed = g.bes_only(ds, initial_graph=start, candidate_edges=frozenset({(0, 1)}))
    assert allowed.directed_edges() == []
    assert allowed.undirected_edges() == []

    blocked = g.bes_only(ds, initial_graph=start, candidate_edges=frozenset({(1, 0)}))
    assert blocked.undirected_edges() == [(0, 1)]


def test_fit_threads_candidate_edges_to_bes_phase() -> None:
    # From the complete graph, BES may not remove C - D if it is not a candidate.
    ds = _dep_dataset()
    full = DAG.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    n = 4
    restricted = frozenset(
        (x, y) for x in range(n) for y in range(n) if x != y and (x, y) not in {(2, 3), (3, 2)}
    )
    cpdag = GES(BDeuScore(ds)).fit(ds, initial_graph=full, candidate_edges=restricted)
    assert 3 in cpdag.adjacent(2)


def test_fit_threads_candidate_edges_to_fes_phase() -> None:
    # BES only deletes, so A - B can appear only if FES was allowed to add it.
    ds = _dep_dataset()
    n = 4
    restricted = frozenset(
        (x, y) for x in range(n) for y in range(n) if x != y and (x, y) not in {(0, 1), (1, 0)}
    )
    cpdag = GES(BDeuScore(ds)).fit(ds, candidate_edges=restricted)
    assert 1 not in cpdag.adjacent(0)


@pytest.mark.parametrize("method", ["fit", "fes_only", "bes_only"])
def test_initial_graph_must_match_the_data(method: str) -> None:
    ds = _dummy_dataset(3)
    with pytest.raises(ValueError):
        getattr(GES(BDeuScore(ds)), method)(ds, initial_graph=DAG(4))


def test_ges_init_rejects_invalid_h_pool() -> None:
    ds = _dummy_dataset(3)
    with pytest.raises(ValueError):
        GES(BDeuScore(ds), h_pool="bogus")  # type: ignore[arg-type]


class _CommonNeighbourScore:
    """A fake score built for `_common_neighbour_dag`, matching test_bes.py's fixture."""

    def local_score(self, node: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))  # accept frozenset/tuple per Score contract
        table = {
            (1, ()): 5.0,
            (1, (0,)): 0.0,
            (1, (2,)): 3.0,
            (1, (0, 2)): 1.0,
            (1, (3,)): 5.0,
        }
        return table.get((node, parents), 0.0)


def _common_neighbour_dag() -> DAG:
    # The same graph as test_bes.py's _common_neighbour_fixture, as a DAG.
    return DAG.from_edges(4, [(0, 2), (0, 1), (2, 1), (1, 3)])


def test_ges_h_pool_threads_to_bes_and_changes_result() -> None:
    ds = _dummy_dataset(4)
    score = _CommonNeighbourScore()
    start = _common_neighbour_dag()

    nayx_result = GES(score, h_pool="nayx").bes_only(ds, initial_graph=start)
    bayesfl_result = GES(score, h_pool="bayesfl").bes_only(ds, initial_graph=start)

    assert nayx_result.directed_edges() == [(0, 2), (1, 2)]
    assert nayx_result.undirected_edges() == [(1, 3)]
    assert bayesfl_result.directed_edges() == []
    assert bayesfl_result.undirected_edges() == [(0, 2), (1, 3)]
    assert nayx_result != bayesfl_result


def test_fit_uses_initial_graph_as_fes_start_not_empty() -> None:
    # No insert pays under this score, so the edge must come from initial_graph.
    ds = _dummy_dataset(2)

    class _FlatScore:
        def local_score(self, node: int, parents: tuple[int, ...]) -> float:
            return 0.0

    score = _FlatScore()
    from_empty = GES(score).fit(ds)
    assert from_empty.directed_edges() == []
    assert from_empty.undirected_edges() == []

    start = DAG.from_edges(2, [(0, 1)])
    from_nonempty = GES(score).fit(ds, initial_graph=start)
    assert dag_to_cpdag(start) == from_nonempty
