"""The client side of a FedGES round, driven directly -- no simulation engine needed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict  # noqa: E402

from fedcausal.data import DiscreteDataset, contiguous_split  # noqa: E402
from fedcausal.exceptions import ClientFailureError  # noqa: E402
from fedcausal.flower.fedges import handle_train_message  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402
from fedcausal.flower.serde import dag_to_record, record_to_dag  # noqa: E402
from fedcausal.fusion import ConsensusUnion  # noqa: E402
from fedcausal.graph import DAG  # noqa: E402

from ._stubs import node_context  # noqa: E402


def _ds(n_rows: int = 1500, seed: int = 3) -> DiscreteDataset:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.random(n_rows) < 0.9, x1, 1 - x1)
    values = np.stack([x0, x1, x2], axis=1).astype(np.int32)
    return DiscreteDataset(values, np.array([2, 2, 2], np.int32), ["A", "B", "C"], [["0", "1"]] * 3)


def _message(phase: str, **records: object) -> Message:
    content = RecordDict({"control": ConfigRecord({"phase": phase}), **records})  # type: ignore[arg-type]
    return Message(content=content, dst_node_id=1, message_type=MessageType.TRAIN, group_id="0")


def _handle(msg: Message, context: Context, partitions: list[DiscreteDataset]) -> Message:
    return handle_train_message(msg, context, InMemoryPartitions(partitions), ConsensusUnion())


def test_build_round_returns_a_scored_local_model() -> None:
    parts = contiguous_split(_ds(), 2)
    context = node_context(0)

    reply = _handle(_message("build"), context, parts)

    control = reply.content["control"]
    assert control["node_index"] == 0
    assert control["n_vars"] == 3
    assert float(control["score"]) < 0  # a log-likelihood
    assert record_to_dag(reply.content["local"], 3).num_edges > 0


def test_fuse_round_incorporates_the_global_dag() -> None:
    parts = contiguous_split(_ds(), 2)
    context = node_context(0)
    _handle(_message("build"), context, parts)

    global_dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    reply = _handle(_message("fuse", **{"global": dag_to_record(global_dag)}), context, parts)

    fused = record_to_dag(reply.content["local"], 3)
    assert fused.num_edges >= global_dag.num_edges


def test_fuse_before_build_is_an_error() -> None:
    parts = contiguous_split(_ds(), 2)
    msg = _message("fuse", **{"global": dag_to_record(DAG(3))})
    with pytest.raises(ClientFailureError, match="before any build round"):
        _handle(msg, node_context(0), parts)


def test_unknown_phase_is_an_error() -> None:
    parts = contiguous_split(_ds(), 2)
    with pytest.raises(ClientFailureError, match="unknown phase"):
        _handle(_message("gossip"), node_context(0), parts)
