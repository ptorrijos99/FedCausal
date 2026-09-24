"""Meek's orientation rules (1995) and DAG-to-CPDAG conversion.

All four rules are applied. R4 only ever fires with background knowledge,
which fedcausal does not take, but it is sound, so keeping it costs nothing.
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
    """Meek R2: if `a -> b -> c` and `a - c`, then `a -> c` (checked both ways)."""
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
    `b`, `d -> c` directed, and `a - d` undirected, orient `a -> b`.
    """
    oriented = False
    for c in sorted(pdag.parents(b)):
        adj = _common_adjacents(pdag, a, c)
        adj.discard(b)
        for d in sorted(adj):
            if d in pdag.adjacent(b):
                continue
            if not pdag.has_directed(d, c):
                continue
            if not pdag.has_undirected(a, d):
                continue
            if _direct(pdag, a, b):
                oriented = True
    return oriented


def apply_meek_rules(pdag: PDAG) -> None:
    """Close `pdag` under Meek's rules R1-R4 in place, to a fixpoint.

    Every scan runs in ascending index order, so the result never depends on
    set iteration order.

    Parameters
    ----------
    pdag : PDAG
        The graph to orient. Mutated in place; undirected edges compelled by the
        rules become directed.
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

    Starts from the skeleton, orients every v-structure, and closes the result
    under Meek's rules.

    Parameters
    ----------
    dag : DAG
        The source directed acyclic graph.

    Returns
    -------
    PDAG
        The completed PDAG representing the Markov equivalence class of `dag`.
    """
    from fedcausal.graph.pdag import PDAG

    pdag = PDAG(dag.num_nodes)
    for u, v in dag.edges():
        pdag.add_undirected(u, v)

    for v in range(dag.num_nodes):
        pars = sorted(dag.parents(v))
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                a = pars[i]
                b = pars[j]
                if b in dag.adjacent(a):
                    continue
                if pdag.has_undirected(a, v):
                    pdag.orient(a, v)
                if pdag.has_undirected(b, v):
                    pdag.orient(b, v)

    apply_meek_rules(pdag)
    return pdag
