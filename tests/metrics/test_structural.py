import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from fedcausal.graph import DAG
from fedcausal.graph.undirected import moralize
from fedcausal.metrics.structural import (
    EdgeScores,
    edge_scores,
    shd,
    shd_skeleton,
    smhd,
    smhd_graphs,
)

# --- Hand-counted SHD examples -------------------------------------------------
#
# Utils.SHD(Dag bn1, Dag bn2) (Utils.java lines 523-536, countDifferences at
# 658-669) sums, per unordered node pair {u, v}:
#   - 0 if both graphs agree (same directed edge, or both non-adjacent);
#   - 1 if the pair is an edge in exactly one graph (addition/deletion);
#   - 1 (NOT 2) if both graphs have an edge between u and v but in opposite
#     directions ("reversed arcs are counted only once", line 526).
# Crucially, Utils.SHD takes two Dag objects directly and never CPDAG-ifies
# them (confirmed by its only call site,
# ExperimentUtils.calculateSHD:148, which passes two plain Dags) -- so this
# is a DAG-vs-DAG exact-edge comparison, not a CPDAG comparison.


def test_shd_identical_graphs_is_zero() -> None:
    g = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert shd(g, g) == 0


def test_shd_one_extra_edge_is_one() -> None:
    g1 = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    # g2 adds 0->3, wholly absent (in either direction) from g1: +1.
    g2 = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert shd(g1, g2) == 1
    assert shd(g2, g1) == 1  # symmetric for this case


def test_shd_reversed_compelled_edge_counts_one() -> None:
    # 0->2, 1->2 is a v-structure/collider: both edges are "compelled" in the
    # CPDAG sense (reversing either would destroy the collider). Reverse the
    # 0-2 edge only.
    g1 = DAG.from_edges(3, [(0, 2), (1, 2)])
    g2 = DAG.from_edges(3, [(2, 0), (1, 2)])
    # Pair {0,2}: 0->2 vs 2->0 -> reversed, counts 1 (not 2).
    # Pair {1,2}: 1->2 in both -> counts 0.
    assert shd(g1, g2) == 1


def test_shd_reversed_reversible_edge_counts_one() -> None:
    # 0->1->2 is a chain with no v-structure: both edges are "reversible" in
    # the CPDAG sense (the whole chain's Markov equivalence class allows
    # flipping either edge). Reverse the 0-1 edge only.
    g1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    g2 = DAG.from_edges(3, [(1, 0), (1, 2)])
    assert shd(g1, g2) == 1


def test_shd_reversed_compelled_and_reversible_edges_are_indistinguishable() -> None:
    # Because Utils.SHD operates directly on DAGs (not CPDAGs), it has no
    # notion of "compelled" vs. "reversible" edges -- both of the above cases
    # produce the identical count (1), which is the point of this test.
    compelled_g1 = DAG.from_edges(3, [(0, 2), (1, 2)])
    compelled_g2 = DAG.from_edges(3, [(2, 0), (1, 2)])
    reversible_g1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    reversible_g2 = DAG.from_edges(3, [(1, 0), (1, 2)])
    assert shd(compelled_g1, compelled_g2) == shd(reversible_g1, reversible_g2) == 1


# --- Hand-counted shd_skeleton examples ----------------------------------------
#
# shd_skeleton ignores direction entirely: it counts node pairs whose
# adjacency (in either direction) differs between the two graphs. This is
# the "undirected adjacency differences" count the brief asks for -- on two
# purely-undirected graphs, Utils.SHDundir (line 552-568) degenerates to
# exactly this same symmetric-difference count (no orientation cost applies
# to an already-undirected edge).


def test_shd_skeleton_identical_graphs_is_zero() -> None:
    g = DAG.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert shd_skeleton(g, g) == 0


def test_shd_skeleton_ignores_direction_of_shared_edge() -> None:
    # A reversed edge does NOT change the skeleton -- shd is 1 (direction
    # differs) but shd_skeleton is 0 (adjacency does not).
    g1 = DAG.from_edges(3, [(0, 1), (1, 2)])
    g2 = DAG.from_edges(3, [(1, 0), (1, 2)])
    assert shd(g1, g2) == 1
    assert shd_skeleton(g1, g2) == 0


def test_shd_skeleton_counts_differing_pairs() -> None:
    # dag: skeleton {0-2, 1-2}. dag2: skeleton {0-1, 1-2}.
    # Symmetric difference: {0-2} (only in dag) and {0-1} (only in dag2) -> 2.
    dag = DAG.from_edges(3, [(0, 2), (1, 2)])
    dag2 = DAG.from_edges(3, [(0, 1), (1, 2)])
    assert shd_skeleton(dag, dag2) == 2


# --- Hand-counted SMHD examples -------------------------------------------------
#
# Utils.SMHD(Graph bn1, Graph bn2) (lines 449-454) moralizes both graphs
# (Utils.moralize, lines 1275-1299: skeleton + an edge between every pair of
# parents sharing a child) then calls SMHDwithoutMoralize (lines 475-485),
# which sums, over each graph's (undirected, so unordered-pair-unique) edge
# set, the edges absent from the other graph -- i.e. the symmetric
# difference of the two moral skeletons.


def test_smhd_identical_graphs_is_zero() -> None:
    g = DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert smhd(g, g) == 0


def test_smhd_moralization_changes_the_count_vs_shd_skeleton() -> None:
    # dag: 0->2, 1->2 (collider). moralize adds {0,1} (co-parents of 2):
    #   moral(dag) = {0-1, 0-2, 1-2} (triangle).
    # dag2: 0->1->2 (chain, no collider). moralize adds nothing:
    #   moral(dag2) = {0-1, 1-2} (path).
    # Symmetric difference of the moral graphs: {0-2} only -> smhd == 1.
    # (Contrast with shd_skeleton(dag, dag2) == 2, computed above on the raw,
    # non-moralized skeletons.)
    dag = DAG.from_edges(3, [(0, 2), (1, 2)])
    dag2 = DAG.from_edges(3, [(0, 1), (1, 2)])
    assert shd_skeleton(dag, dag2) == 2
    assert smhd(dag, dag2) == 1


def test_smhd_graphs_matches_smhd_on_precomputed_moral_matrices() -> None:
    dag = DAG.from_edges(3, [(0, 2), (1, 2)])
    dag2 = DAG.from_edges(3, [(0, 1), (1, 2)])
    m1 = moralize(dag)
    m2 = moralize(dag2)
    assert smhd_graphs(m1, m2) == 1
    assert smhd_graphs(m1, m2) == smhd(dag, dag2)


def test_smhd_graphs_hand_built_matrices() -> None:
    # Triangle {0-1,0-2,1-2} vs path {0-1,1-2}: differ only on {0,2} -> 1.
    triangle = np.array(
        [
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ]
    )
    path = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ]
    )
    assert smhd_graphs(triangle, path) == 1
    assert smhd_graphs(triangle, triangle) == 0


# --- Metamorphic properties -----------------------------------------------------


def _random_dag(draw: st.DrawFn, n: int) -> DAG:
    perm = draw(st.permutations(list(range(n))))
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if draw(st.booleans()):
                edges.append((perm[i], perm[j]))
    return DAG.from_edges(n, edges)


@st.composite
def random_dag_pairs(draw: st.DrawFn) -> tuple[DAG, DAG]:
    n = draw(st.integers(min_value=0, max_value=7))
    return _random_dag(draw, n), _random_dag(draw, n)


@settings(max_examples=100, deadline=None)
@given(random_dag_pairs())
def test_shd_of_graph_with_itself_is_zero(pair: tuple[DAG, DAG]) -> None:
    g1, _ = pair
    assert shd(g1, g1) == 0
    assert shd_skeleton(g1, g1) == 0
    assert smhd(g1, g1) == 0


@settings(max_examples=100, deadline=None)
@given(random_dag_pairs())
def test_shd_is_symmetric(pair: tuple[DAG, DAG]) -> None:
    g1, g2 = pair
    assert shd(g1, g2) == shd(g2, g1)
    assert shd_skeleton(g1, g2) == shd_skeleton(g2, g1)


@settings(max_examples=100, deadline=None)
@given(random_dag_pairs())
def test_smhd_is_symmetric(pair: tuple[DAG, DAG]) -> None:
    g1, g2 = pair
    assert smhd(g1, g2) == smhd(g2, g1)


@settings(max_examples=100, deadline=None)
@given(random_dag_pairs())
def test_edge_scores_are_within_unit_bounds(pair: tuple[DAG, DAG]) -> None:
    g1, g2 = pair
    scores = edge_scores(g1, g2)
    for value in (
        scores.precision,
        scores.recall,
        scores.f1,
        scores.precision_skel,
        scores.recall_skel,
        scores.f1_skel,
    ):
        assert 0.0 <= value <= 1.0


@settings(max_examples=100, deadline=None)
@given(random_dag_pairs())
def test_edge_scores_perfect_match_when_identical(pair: tuple[DAG, DAG]) -> None:
    g1, _ = pair
    scores = edge_scores(g1, g1)
    assert scores.precision == scores.recall == scores.f1 == 1.0
    assert scores.precision_skel == scores.recall_skel == scores.f1_skel == 1.0


# --- edge_scores: zero-division policy ------------------------------------------


def test_edge_scores_both_empty_is_perfect_by_convention() -> None:
    empty = DAG(3)
    scores = edge_scores(empty, empty)
    assert scores == EdgeScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_edge_scores_true_empty_est_nonempty_is_zero() -> None:
    true = DAG(3)
    est = DAG.from_edges(3, [(0, 1)])
    scores = edge_scores(true, est)
    assert scores == EdgeScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_edge_scores_true_nonempty_est_empty_is_zero() -> None:
    true = DAG.from_edges(3, [(0, 1)])
    est = DAG(3)
    scores = edge_scores(true, est)
    assert scores == EdgeScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_edge_scores_partial_overlap() -> None:
    # true: 0->1, 1->2.  est: 0->1, 2->1 (reversed).
    # Directed: TP={0->1}, so precision = 1/2, recall = 1/2, f1 = 1/2.
    # Skeleton: both pairs {0,1} and {1,2} adjacent in both -> perfect skeleton.
    true = DAG.from_edges(3, [(0, 1), (1, 2)])
    est = DAG.from_edges(3, [(0, 1), (2, 1)])
    scores = edge_scores(true, est)
    assert scores.precision == 0.5
    assert scores.recall == 0.5
    assert scores.f1 == 0.5
    assert scores.precision_skel == 1.0
    assert scores.recall_skel == 1.0
    assert scores.f1_skel == 1.0
