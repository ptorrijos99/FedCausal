"""Meek orientation rules and DAG-to-CPDAG conversion.

This module implements Chris Meek's (1995) complete orientation rule set
(R1-R4) for closing a partially directed acyclic graph under the constraints
that keep it Markov-equivalence preserving, plus :func:`dag_to_cpdag`, which
builds the completed PDAG (pattern) of a DAG.

The rule implementations follow Tetrad's
``edu.cmu.tetrad.search.utils.MeekRules``. Its ``meekPreventCycles`` flag
defaults to ``false`` there, so :func:`apply_meek_rules` performs plain
orientation without the arbitrary-collider cycle-prevention branch, and without
Tetrad's ``revertToUnshieldedColliders`` pre-step, since the compelled edges
:func:`dag_to_cpdag` orients beforehand must be preserved.

Note on R4: in Tetrad, rule R4 fires only when background knowledge is present
(``useRule4 = !knowledge.isEmpty()``). ``fedcausal`` has no background-knowledge
mechanism yet, but R4 is ported and kept active here for fidelity; it is a sound
orientation rule, so enabling it never introduces an incorrect orientation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedcausal.graph.dag import DAG
    from fedcausal.graph.pdag import PDAG

__all__ = ["apply_meek_rules", "dag_to_cpdag"]


def _common_adjacents(pdag: PDAG, x: int, y: int) -> set[int]:
    """Return nodes adjacent to both `x` and `y`."""
    return set(pdag.adjacent(x) & pdag.adjacent(y))


def _direct(pdag: PDAG, a: int, c: int) -> bool:
    """Orient the undirected edge `a - c` as `a -> c`.

    Returns ``True`` iff `a - c` was undirected and has now been oriented.
    Mirrors Tetrad's ``direct`` with ``meekPreventCycles == false`` and no
    background knowledge (arrowheads always allowed).
    """
    if not pdag.has_undirected(a, c):
        return False
    pdag.orient(a, c)
    return True


def _meek_r1(pdag: PDAG, b: int, c: int) -> bool:
    """Meek R1: if `a -> b`, `b - c`, and `a` not adjacent to `c`, then `b -> c`."""
    for a in sorted(pdag.parents(b)):
        if a in pdag.adjacent(c):
            continue
        if _direct(pdag, b, c):
            return True
    return False


def _r2_helper(pdag: PDAG, a: int, c: int) -> bool:
    return _direct(pdag, a, c)


def _meek_r2(pdag: PDAG, a: int, c: int) -> bool:
    """Meek R2: if `a -> b -> c` and `a - c`, then `a -> c`.

    Both chain directions through a
    common adjacent `b` are checked, so this may orient `a - c` either way.
    """
    oriented = False
    for b in sorted(_common_adjacents(pdag, a, c)):
        if pdag.has_directed(a, b) and pdag.has_directed(b, c) and _r2_helper(pdag, a, c):
            oriented = True
        if pdag.has_directed(c, b) and pdag.has_directed(b, a) and _r2_helper(pdag, c, a):
            oriented = True
    return oriented


def _r3_helper(pdag: PDAG, a: int, d: int, b: int, c: int) -> bool:
    b4 = pdag.has_undirected(d, a)
    b5 = pdag.has_undirected(d, b)
    b6 = pdag.has_undirected(d, c)
    b7 = pdag.has_directed(b, a)
    b8 = pdag.has_directed(c, a)
    if b4 and b5 and b6 and b7 and b8:
        return _direct(pdag, d, a)
    return False


def _meek_r3(pdag: PDAG, d: int, a: int) -> bool:
    """Meek R3: if `d - a`, `d - b`, `d - c`, `b -> a`, `c -> a`, `b` not adj `c`, then `d -> a`."""
    common = sorted(_common_adjacents(pdag, a, d))
    if len(common) < 2:
        return False
    oriented = False
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            b = common[i]
            c = common[j]
            if c not in pdag.adjacent(b) and _r3_helper(pdag, a, d, b, c):
                oriented = True
    return oriented


def _meek_r4(pdag: PDAG, a: int, b: int) -> bool:
    """Meek R4: orient `a -> b` when a directed chain forces it.

    Given the undirected edge `a - b`, for each parent `c` of `b` and each
    common adjacent `d` of `a` and `c` (``d != b``) with `d` not adjacent to
    `b`, `d -> c` directed, and `a - d` *not* directed, orient `a -> b`.

    In Tetrad this rule is gated behind ``useRule4 = !knowledge.isEmpty()``
    and
    thus never fires without background knowledge; ``fedcausal`` keeps it active
    for fidelity. R4 is sound, so enabling it cannot mis-orient an edge.
    """
    oriented = False
    for c in sorted(pdag.parents(b)):
        adj = _common_adjacents(pdag, a, c)
        adj.discard(b)
        for d in sorted(adj):
            if d in pdag.adjacent(b):
                continue
            # edge(d, c) must point towards c, i.e. d -> c
            if not pdag.has_directed(d, c):
                continue
            # edge(a, d) must not be directed (must be undirected)
            if not pdag.has_undirected(a, d):
                continue
            if _direct(pdag, a, b):
                oriented = True
    return oriented


def apply_meek_rules(pdag: PDAG) -> None:
    """Close `pdag` under Meek's rules R1-R4 in place, to a fixpoint.

    Iterates over the undirected edges in sorted order, applying rules R1-R4
    (each in both endpoint orderings, matching Tetrad's ``orientImplied``
    loop),
    and repeats until a full pass orients nothing. Scans are deterministic:
    undirected edges, parent sets, and common-adjacent sets are all traversed in
    ascending index order, so the result does not depend on set iteration order.

    Parameters
    ----------
    pdag : PDAG
        The graph to orient. Mutated in place; undirected edges compelled by the
        rules become directed.

    Notes
    -----
    ``meekPreventCycles == false`` and ``revertToUnshieldedColliders`` disabled
    (callers such as :func:`dag_to_cpdag` orient v-structures beforehand and
    those must be preserved). See the module docstring on R4.
    """
    oriented = True
    while oriented:
        oriented = False
        for x, y in pdag.undirected_edges():
            # The edge may have been oriented earlier in this pass.
            if not pdag.has_undirected(x, y):
                continue
            if (
                _meek_r1(pdag, x, y)
                or _meek_r1(pdag, y, x)
                or _meek_r2(pdag, x, y)
                or _meek_r2(pdag, y, x)
                or _meek_r3(pdag, x, y)
                or _meek_r3(pdag, y, x)
                or _meek_r4(pdag, x, y)
                or _meek_r4(pdag, y, x)
            ):
                oriented = True


def dag_to_cpdag(dag: DAG) -> PDAG:
    """Return the CPDAG (pattern) of `dag`.

    Builds a PDAG with every DAG edge undirected, orients both edges of every
    v-structure (a pair of non-adjacent parents of a common child) as directed
    (compelled), then closes under Meek's rules via :func:`apply_meek_rules`.

    Parameters
    ----------
    dag : DAG
        The source directed acyclic graph.

    Returns
    -------
    PDAG
        The completed PDAG representing the Markov equivalence class of `dag`.

    Notes
    -----
    This mirrors Tetrad's pattern-from-DAG construction, pinned against
    reference output by the regression tests.
    """
    from fedcausal.graph.pdag import PDAG

    pdag = PDAG(dag.num_nodes)
    # Skeleton: every DAG edge as undirected.
    for u, v in dag.edges():
        pdag.add_undirected(u, v)

    # Orient v-structures: non-adjacent parent pairs of a common child.
    for v in range(dag.num_nodes):
        pars = sorted(dag.parents(v))
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                a = pars[i]
                b = pars[j]
                if b in dag.adjacent(a):
                    continue
                # a -> v <- b is an unshielded collider; compel both edges.
                if pdag.has_undirected(a, v):
                    pdag.orient(a, v)
                if pdag.has_undirected(b, v):
                    pdag.orient(b, v)

    apply_meek_rules(pdag)
    return pdag
