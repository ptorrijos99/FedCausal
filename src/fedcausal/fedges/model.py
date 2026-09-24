"""The unit the FedGES loop exchanges: a DAG and its score."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedcausal.graph import DAG

__all__ = ["BNModel"]


class BNModel:
    """A learned structure and its score.

    Two models are equal when their DAGs have the same nodes and edges; the
    score does not take part.

    Parameters
    ----------
    dag : DAG
        The structure.
    score : float, optional
        Its score on the data of the site that learned it. Default ``0.0``.
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
