"""The unit exchanged in the federated loop: a DAG plus its cached score.

Two models are equal iff they have the same node count and the same set of
directed edges, and the hash is taken over that edge set. The score is metadata
only and never participates in equality, so a model compares the same however
it was scored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedcausal.graph import DAG

__all__ = ["BNModel"]


class BNModel:
    """A learned structure and its (BDeu) score.

    Parameters
    ----------
    dag : DAG
        The directed acyclic graph.
    score : float, optional
        Cached score of `dag` (BDeu on the owning client's local data).
        Default ``0.0``, matching ``BN``'s uninitialised score.
    """

    __slots__ = ("dag", "score")

    def __init__(self, dag: DAG, score: float = 0.0) -> None:
        self.dag = dag
        self.score = float(score)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BNModel):
            return NotImplemented
        return self.dag.num_nodes == other.dag.num_nodes and set(self.dag.edges()) == set(
            other.dag.edges()
        )

    def __hash__(self) -> int:
        return hash(frozenset(self.dag.edges()))

    def __repr__(self) -> str:
        return (
            f"BNModel(num_nodes={self.dag.num_nodes}, "
            f"num_edges={self.dag.num_edges}, score={self.score})"
        )
