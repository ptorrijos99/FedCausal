from hypothesis import given, settings
from hypothesis import strategies as st

from fedcausal.graph import DAG, PDAG
from fedcausal.graph.meek import apply_meek_rules, dag_to_cpdag


def test_meek_r1_orients_chain() -> None:
    # 0→1, 1—2, 0 and 2 non-adjacent  =>  1→2
    p = PDAG(3)
    p.add_directed(0, 1)
    p.add_undirected(1, 2)
    apply_meek_rules(p)
    assert p.has_directed(1, 2)


def test_meek_r2() -> None:
    # 0→1→2 and 0—2  =>  0→2
    p = PDAG(3)
    p.add_directed(0, 1)
    p.add_directed(1, 2)
    p.add_undirected(0, 2)
    apply_meek_rules(p)
    assert p.has_directed(0, 2)


def test_meek_r3() -> None:
    # a—b, a—c, a—d, c→b, d→b, c≁d  =>  a→b
    # nodes: a=0, b=1, c=2, d=3
    p = PDAG(4)
    p.add_undirected(0, 1)
    p.add_undirected(0, 2)
    p.add_undirected(0, 3)
    p.add_directed(2, 1)
    p.add_directed(3, 1)
    apply_meek_rules(p)
    assert p.has_directed(0, 1)


def test_meek_r4_fires() -> None:
    # Nodes: a=0, b=1, c=2, d=3.
    #
    # Target: undirected a-b, with c a parent of b, d a common adjacent of a
    # and c (d != b), d not adjacent to b, d->c directed, a-d undirected.
    # R4 should then orient a->b.
    #
    #   a(0) - b(1)      undirected edge to be oriented by R4
    #   c(2) -> b(1)     c is a parent of b
    #   d(3) -> c(2)     d -> c directed
    #   a(0) - d(3)      a - d undirected
    #   a(0) - c(2)      EXTRA edge, added only to block R1 pre-emption (see
    #                    below); not part of R4's precondition list itself.
    #
    # Without the extra a-c edge, R1 fires first and misorients the edge:
    # parents(b) = {c}, and if c were not adjacent to a, R1(b, a) would see
    # "c -> b, b - a, c not adjacent to a" and orient b -> a before R4 is
    # ever reached (R1 has strictly higher precedence in
    # ``apply_meek_rules``'s rule ordering). Adding the undirected a-c edge
    # makes c adjacent to a, which is exactly the guard clause
    # ``if a in pdag.adjacent(c): continue`` in ``_meek_r1`` -- so R1 no
    # longer fires for this parent, and R4 gets to run. This was verified by
    # hand and confirmed empirically: disabling ``_meek_r4`` leaves a-b
    # undirected after ``apply_meek_rules``, while the real R4 orients it
    # a -> b, so the assertion below genuinely exercises R4 (and not a
    # lower-precedence rule) through the public entry point.
    p = PDAG(4)
    p.add_undirected(0, 1)  # a - b
    p.add_directed(2, 1)  # c -> b
    p.add_directed(3, 2)  # d -> c
    p.add_undirected(0, 3)  # a - d
    p.add_undirected(0, 2)  # a - c (blocks R1 pre-emption)
    apply_meek_rules(p)
    assert p.has_directed(0, 1)


def test_dag_to_cpdag_v_structure() -> None:
    # collider 0→2←1 is compelled; chain 2→3 gets... (3-node star checked exactly)
    dag = DAG.from_edges(3, [(0, 2), (1, 2)])
    c = dag_to_cpdag(dag)
    assert c.has_directed(0, 2) and c.has_directed(1, 2)
    assert not c.undirected_edges()


def test_dag_to_cpdag_chain_fully_undirected() -> None:
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    c = dag_to_cpdag(dag)
    assert set(c.undirected_edges()) == {(0, 1), (1, 2)}


def test_to_dag_extension_roundtrip() -> None:
    dag = DAG.from_edges(4, [(0, 1), (1, 3), (2, 3)])
    ext = dag_to_cpdag(dag).to_dag()
    # extension must be in the same Markov equivalence class:
    assert dag_to_cpdag(ext) == dag_to_cpdag(dag)


def _random_dag(draw: st.DrawFn, n: int) -> DAG:
    """Draw a random DAG on ``n`` nodes via a random permutation + lower-tri mask."""
    perm = draw(st.permutations(list(range(n))))
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if draw(st.booleans()):
                # perm[i] precedes perm[j] in the causal order
                edges.append((perm[i], perm[j]))
    return DAG.from_edges(n, edges)


@st.composite
def random_dags(draw: st.DrawFn) -> DAG:
    n = draw(st.integers(min_value=1, max_value=8))
    return _random_dag(draw, n)


@settings(max_examples=200, deadline=None)
@given(random_dags())
def test_cpdag_extension_roundtrip_property(dag: DAG) -> None:
    cpdag = dag_to_cpdag(dag)
    ext = cpdag.to_dag()  # must never raise
    assert dag_to_cpdag(ext) == cpdag


@settings(max_examples=100, deadline=None)
@given(random_dags())
def test_cpdag_same_skeleton_as_dag(dag: DAG) -> None:
    cpdag = dag_to_cpdag(dag)
    dag_skel = {tuple(sorted(e)) for e in dag.edges()}
    cpdag_skel = set(cpdag.directed_edges()) | set(cpdag.undirected_edges())
    cpdag_skel = {tuple(sorted(e)) for e in cpdag_skel}
    assert cpdag_skel == dag_skel
