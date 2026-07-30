"""Scoring interface for candidate causal structures.

This module defines :class:`Score`, the structural-typing protocol that all
local scoring functions (BDeu, and any future BIC/BDe variants) satisfy. A
score assigns a real number to a ``(node, parents)`` family; decomposable
scores such as BDeu let a full-DAG score be recovered as the sum of local
family scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Collection


@runtime_checkable
class Score(Protocol):
    """Structural-typing protocol for a decomposable local score.

    A conforming object exposes :meth:`local_score`, returning the local
    contribution of a single node given its parent set. Because BDeu (and
    the other Bayesian/information-theoretic scores used here) are
    *decomposable*, the score of a whole DAG is the sum of the local
    scores of its families; see :meth:`fedcausal.scores.BDeuScore.score_dag`.
    """

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of `node` given `parents`."""
        ...
