"""Minimal-I-map transform of a DAG into a target order (BetaToAlpha port).

Ported from ``consensusBN/BetaToAlpha.java``. Given a DAG ``G`` and a target
order ``alpha``, return a new DAG that is (a) consistent with ``alpha`` (every
arc points forward in the order) and (b) a minimal I-map of ``G`` -- obtained
by first building an intermediate ``beta`` order by sink-peeling ``G``, then
bubbling nodes into ``alpha`` order by adjacent swaps, reversing each crossed
arc and moralizing so that no conditional independence of ``G`` is lost.

Determinism: the sink chosen at each peeling step is the lowest-index ready
sink, where the original picked an arbitrary element of an unordered set. Documented
as
a deliberate substitution; where it can change the intermediate ``beta`` the
final union (:func:`~fedcausal.fusion.consensus.consensus_union`) is the
binding reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fedcausal.graph import DAG

__all__ = ["beta_to_alpha"]


def beta_to_alpha(dag: DAG, order: Sequence[int]) -> DAG:
    """Return `dag` re-expressed as a minimal I-map consistent with `order`.

    Parameters
    ----------
    dag : DAG
        Input DAG over ``0..p-1``. Not mutated.
    order : Sequence[int]
        Target topological order (e.g. from
        :func:`~fedcausal.fusion.alpha.alpha_order`).

    Returns
    -------
    DAG
        A new DAG whose every arc points forward in `order` and which is a
        minimal I-map of `dag` (no conditional independence removed).
    """
    rank = {node: i for i, node in enumerate(order)}

    # 1. Build beta by sink-peeling a working copy, inserting each sink near
    #    its alpha position.
    work = dag.copy()
    peeled: set[int] = set()
    live = set(range(dag.num_nodes))
    beta: list[int] = []

    def ready_sinks() -> list[int]:
        return sorted(n for n in live if not (work.children(n) - peeled))

    while live:
        sink = ready_sinks()[0]
        live.discard(sink)
        peeled.add(sink)

        i = 0
        if beta:
            while True:
                nodej = beta[i]
                if rank[nodej] > rank[sink]:
                    break
                if sink in dag.parents(nodej):  # the original G, deliberately
                    break
                if i == len(beta) - 1:
                    break
                i += 1
        beta.insert(i, sink)

    # 2. Bubble beta toward alpha with arc reversal + moralization.
    out = dag.copy()
    aux: list[int] = [beta.pop(0)]
    while beta:
        aux.append(beta.pop(0))
        i = len(aux) - 1
        while i > 0:
            node_y = aux[i]
            node_z = aux[i - 1]
            if rank[node_z] > rank[node_y]:
                if node_y in out.adjacent(node_z):
                    pa_z = sorted(out.parents(node_z))
                    pa_y = [p for p in out.parents(node_y) if p != node_z]
                    if out.has_edge(node_z, node_y):
                        out.remove_edge(node_z, node_y)
                    elif out.has_edge(node_y, node_z):
                        out.remove_edge(node_y, node_z)
                    if not out.has_edge(node_y, node_z):
                        out.add_edge(node_y, node_z)
                    for p in pa_z:
                        if node_y not in out.adjacent(p):
                            out.add_edge(p, node_y)
                    for p in pa_y:
                        if node_z not in out.adjacent(p):
                            out.add_edge(p, node_z)
                aux.pop(i)
                aux.insert(i - 1, node_y)
                i -= 1
            else:
                break
    return out
