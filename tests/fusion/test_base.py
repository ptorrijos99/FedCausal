"""Tests for the `FusionOperator` protocol."""

from __future__ import annotations

from collections.abc import Sequence

from fedcausal.fusion import FusionOperator
from fedcausal.graph import DAG


def test_fusion_operator_is_structural_subtype() -> None:
    """Any object with a matching ``fuse`` satisfies the runtime protocol."""

    class Identity:
        def fuse(self, dags: Sequence[DAG]) -> DAG:
            return dags[0]

    op: FusionOperator = Identity()
    assert isinstance(op, FusionOperator)
    d = DAG.from_edges(2, [(0, 1)])
    assert op.fuse([d]) is d
