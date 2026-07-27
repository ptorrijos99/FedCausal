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
    # A stand-in dataset for fake-Score fixtures below, where only `n_vars`
    # (via `data.n_vars`) is ever consulted by GES -- the fake scores never
    # touch `data.values`, so a single all-zero row is enough.
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
    # `fit`'s composed FES-then-BES result must never score worse than doing
    # nothing at all -- a minimal sanity floor that would catch, e.g., `fit`
    # accidentally returning the wrong intermediate PDAG (say the pre-FES
    # empty start) instead of the final one.
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
    """A fake score whose FES phase overshoots and whose BES phase corrects it.

    On an empty `PDAG(3)`, FES greedily inserts, in order: `Insert(0, 1,
    T=empty)` (delta 10), then `Insert(1, 2, T=empty)` (delta 8), then
    `Insert(0, 2, T=empty)` (delta 9 - 8 = 1, since by then `Pa(2) = {}` --
    the 1-2 edge re-canonicalizes to undirected, so `local_score(2, (1,))`
    is still the baseline -- and adding 0 raises the family score of node 2
    from 8 to 9). FES terminates at the fully-undirected triangle {0-1, 0-2,
    1-2}, since every pair is now adjacent.

    BES then reconsiders the graph fresh: `Delete(x=2, y=1, H=empty)` asks
    for `local_score(1, (0,)) - local_score(1, (0, 2))` = `10 - 0 = 10` (the
    `(1, (0, 2))` entry is deliberately left at the 0.0 default), a far
    larger delta than any Insert step saw, so BES removes the 1-2 edge,
    leaving {0-1, 0-2}.

    Values obtained by running `run_fes`/`run_bes` directly against this
    table once implemented, and
    recorded here as the fixture's contract, not derived from first
    principles in this docstring.
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
    # This fixture is built so the three candidate implementations of `fit`
    # give three *different* answers, so a single assertion set rules all of
    # them out at once:
    #   - "FES only" (BES never runs): full triangle, same as `fes_only`.
    #   - "BES then FES" (wrong order): BES on the empty start graph is a
    #     no-op (nothing adjacent to delete yet), so this degrades to the
    #     same full triangle as "FES only".
    #   - "FES then BES" (correct): BES's fresh look at the finished triangle
    #     finds a large, genuine Delete that neither of the above ever
    #     considers, pruning the 1-2 edge.
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
    # candidate_edges holds *ordered* pairs on the public API too: (x, y)
    # means "consider inserting x -> y", not "this undirected edge is a
    # candidate". frozenset({(0, 2)}) allows the beneficial insert below;
    # frozenset({(2, 0)}) -- the same two nodes, opposite order -- allows
    # nothing.
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
    # `fit` must restrict *both* phases with the same `candidate_edges`, not
    # just FES: start from the fully-connected graph (so FES itself has
    # nothing left to insert -- every pair is already adjacent) and confirm
    # that excluding the spurious C-D pair from `candidate_edges` keeps BES
    # from ever pruning it, exactly as the equivalent `run_bes`-level test
    # (test_run_bes_restricted_candidate_edges_keeps_spurious_edge) pins for
    # the private phase function.
    ds = _dep_dataset()
    full = DAG.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    n = 4
    restricted = frozenset(
        (x, y) for x in range(n) for y in range(n) if x != y and (x, y) not in {(2, 3), (3, 2)}
    )
    cpdag = GES(BDeuScore(ds)).fit(ds, initial_graph=full, candidate_edges=restricted)
    assert 3 in cpdag.adjacent(2)


def test_fit_threads_candidate_edges_to_fes_phase() -> None:
    # Mirrors test_fit_threads_candidate_edges_to_bes_phase for the forward
    # leg: candidate_edges restriction on FES is otherwise exercised only via
    # fes_only, never through fit itself, so a fit that dropped
    # candidate_edges on its run_fes call (while still forwarding it to
    # run_bes) would pass every other test in this file. Since BES only ever
    # deletes edges, restricting the A-B pair out of candidate_edges means
    # the A-B edge can appear in the final result only if FES was allowed to
    # insert it -- i.e. only if fit failed to forward the restriction to
    # run_fes.
    ds = _dep_dataset()
    n = 4
    restricted = frozenset(
        (x, y) for x in range(n) for y in range(n) if x != y and (x, y) not in {(0, 1), (1, 0)}
    )
    cpdag = GES(BDeuScore(ds)).fit(ds, candidate_edges=restricted)
    assert 1 not in cpdag.adjacent(0)


def test_bes_only_requires_initial_graph() -> None:
    # bes_only's initial_graph has no default -- a pure BES run from an
    # empty graph is meaningless (nothing is ever adjacent, so BES can never
    # do anything). This pins that contract against a future edit that adds
    # a convenience default of None.
    ds = _dummy_dataset(3)
    g = GES(BDeuScore(ds))
    with pytest.raises(TypeError):
        g.bes_only(ds)  # type: ignore[call-arg]


def test_fit_rejects_initial_graph_node_count_mismatch() -> None:
    ds = _dummy_dataset(3)
    mismatched = DAG(4)
    with pytest.raises(ValueError):
        GES(BDeuScore(ds)).fit(ds, initial_graph=mismatched)


def test_fes_only_rejects_initial_graph_node_count_mismatch() -> None:
    # fes_only routes through the same _check_node_count helper as fit
    # (test_fit_rejects_initial_graph_node_count_mismatch above), but that
    # sharing was previously untested on this method directly.
    ds = _dummy_dataset(3)
    mismatched = DAG(4)
    with pytest.raises(ValueError):
        GES(BDeuScore(ds)).fes_only(ds, initial_graph=mismatched)


def test_bes_only_rejects_initial_graph_node_count_mismatch() -> None:
    # Same as test_fes_only_rejects_initial_graph_node_count_mismatch above,
    # for bes_only.
    ds = _dummy_dataset(3)
    mismatched = DAG(4)
    with pytest.raises(ValueError):
        GES(BDeuScore(ds)).bes_only(ds, initial_graph=mismatched)


def test_ges_init_rejects_invalid_h_pool() -> None:
    # Validated eagerly at construction time, not lazily on first use --
    # otherwise a GES instance only ever used for fes_only would silently
    # carry a bogus h_pool that never gets checked.
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
    # dag_to_cpdag of this DAG reproduces the fully-undirected 4-node
    # skeleton {0-1, 0-2, 1-2, 1-3} test_bes.py's `_common_neighbour_fixture`
    # builds directly as a PDAG (verified empirically: no v-structure exists,
    # since node 1's two parents 0 and 2 are themselves adjacent via 0->2).
    return DAG.from_edges(4, [(0, 2), (0, 1), (2, 1), (1, 3)])


def test_ges_h_pool_threads_to_bes_and_changes_result() -> None:
    # Demonstrates h_pool is not merely accepted and ignored: the same
    # starting graph and score, run through bes_only under each pool,
    # produce different final CPDAGs -- matching the divergence
    # test_run_bes_pools_diverge_on_common_neighbourhood pins at the
    # run_bes level, now exercised through the public GES API and its
    # constructor argument instead of a direct run_bes(..., h_pool=...) call.
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
    # A non-default initial_graph must actually change fit's behaviour: with
    # a starting DAG that already has the 0-1 edge, and a score under which
    # inserting it from scratch would never be attractive (delta 0), fit
    # must still keep the edge it started from -- proving initial_graph is
    # really threaded into the FES start rather than fit always beginning
    # from PDAG(n_vars) regardless of the argument.
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
