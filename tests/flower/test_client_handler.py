"""The client side of a round, driven directly -- no simulation engine needed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict  # noqa: E402

from fedcausal.data import DiscreteDataset, contiguous_split  # noqa: E402
from fedcausal.exceptions import ClientFailureError, ConfigurationError  # noqa: E402
from fedcausal.flower.app import handle_train_message  # noqa: E402
from fedcausal.flower.data import InMemoryPartitions, NodeConfigLoader  # noqa: E402
from fedcausal.flower.serde import dag_to_record, record_to_dag  # noqa: E402
from fedcausal.fusion import ConsensusUnion  # noqa: E402
from fedcausal.graph import DAG  # noqa: E402


def _ds(n_rows: int = 1500, seed: int = 3) -> DiscreteDataset:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.random(n_rows) < 0.9, x1, 1 - x1)
    values = np.stack([x0, x1, x2], axis=1).astype(np.int32)
    return DiscreteDataset(values, np.array([2, 2, 2], np.int32), ["A", "B", "C"], [["0", "1"]] * 3)


def _context(partition_id: int = 0, **node_config: object) -> Context:
    config = {"partition-id": partition_id, **node_config}
    return Context(run_id=1, node_id=1, node_config=config, state=RecordDict(), run_config={})


def _message(phase: str, **records: object) -> Message:
    content = RecordDict({"control": ConfigRecord({"phase": phase}), **records})  # type: ignore[arg-type]
    return Message(content=content, dst_node_id=1, message_type=MessageType.TRAIN, group_id="0")


def _handle(msg: Message, context: Context, partitions: list[DiscreteDataset]) -> Message:
    return handle_train_message(msg, context, InMemoryPartitions(partitions), ConsensusUnion())


def test_build_round_returns_a_scored_local_model() -> None:
    parts = contiguous_split(_ds(), 2)
    context = _context(0)

    reply = _handle(_message("build"), context, parts)

    control = reply.content["control"]
    assert control["node_index"] == 0
    assert control["n_vars"] == 3
    assert float(control["score"]) < 0  # a log-likelihood
    assert record_to_dag(reply.content["local"], 3).num_edges > 0


def test_the_local_model_persists_between_rounds() -> None:
    """The supernode's own state carries its model, not the server."""
    parts = contiguous_split(_ds(), 2)
    context = _context(0)

    _handle(_message("build"), context, parts)

    assert "local_dag" in context.state.array_records


def test_fuse_round_incorporates_the_global_dag() -> None:
    parts = contiguous_split(_ds(), 2)
    context = _context(0)
    _handle(_message("build"), context, parts)

    global_dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    reply = _handle(_message("fuse", **{"global": dag_to_record(global_dag)}), context, parts)

    fused = record_to_dag(reply.content["local"], 3)
    assert fused.num_edges >= global_dag.num_edges


def test_fuse_before_build_is_an_error() -> None:
    parts = contiguous_split(_ds(), 2)
    msg = _message("fuse", **{"global": dag_to_record(DAG(3))})
    with pytest.raises(ClientFailureError, match="before any build round"):
        _handle(msg, _context(0), parts)


def test_unknown_phase_is_an_error() -> None:
    parts = contiguous_split(_ds(), 2)
    with pytest.raises(ClientFailureError, match="unknown phase"):
        _handle(_message("gossip"), _context(0), parts)


def test_out_of_range_partition_id_is_an_error() -> None:
    parts = contiguous_split(_ds(), 2)
    with pytest.raises(ConfigurationError, match="out of range"):
        _handle(_message("build"), _context(7), parts)


def test_node_config_loader_reads_the_supernodes_own_shard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    csv = tmp_path / "shard.csv"
    csv.write_text("A,B\nyes,no\nno,yes\nyes,yes\nno,no\n", encoding="utf-8")

    loaded = NodeConfigLoader()(_context(0, **{"data-path": str(csv)}))

    assert loaded.names == ("A", "B")
    assert loaded.n_rows == 4


def test_node_config_loader_says_what_is_missing() -> None:
    with pytest.raises(ConfigurationError, match="--node-config"):
        NodeConfigLoader()(_context(0))


def test_node_config_loader_rejects_a_missing_file() -> None:
    context = _context(0, **{"data-path": "/nope/absent.csv"})
    with pytest.raises(ConfigurationError, match="does not exist"):
        NodeConfigLoader()(context)
