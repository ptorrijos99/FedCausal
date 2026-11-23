"""FedGES over Flower: the site handler, the server loop, and agreement with one process."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("flwr")

from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict  # noqa: E402

from fedcausal.data import DiscreteDataset, contiguous_split  # noqa: E402
from fedcausal.exceptions import (  # noqa: E402
    ClientFailureError,
    ConfigurationError,
    SchemaMismatchError,
)
from fedcausal.fedges import Site, default_convergence, run_rounds  # noqa: E402
from fedcausal.flower.fedges import handle_train_message, run_server_loop  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions, NodeConfigLoader  # noqa: E402
from fedcausal.flower.serde import dag_to_record, record_to_dag  # noqa: E402
from fedcausal.fusion import Consensus  # noqa: E402
from fedcausal.graph import DAG  # noqa: E402

from ._stubs import StubGrid, discrete_chain, node_context, server_context, single  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from fedcausal.fedges import BNModel


def _message(phase: str, **records: object) -> Message:
    content = RecordDict({"control": ConfigRecord({"phase": phase}), **records})  # type: ignore[arg-type]
    return Message(content=content, dst_node_id=1, message_type=MessageType.TRAIN, group_id="0")


def _grid(loaders: list[object], fail_nodes: set[int] | None = None) -> StubGrid:
    return StubGrid(handle_train_message, loaders, fail_nodes)  # type: ignore[arg-type]


def _partition_grid(data: DiscreteDataset, n: int, **kwargs: object) -> StubGrid:
    parts = contiguous_split(data, n)
    return _grid([InMemoryPartitions(parts) for _ in range(n)], **kwargs)  # type: ignore[arg-type]


def _run(grid: object, n_clients: int, **kwargs: object) -> DAG | None:
    return run_server_loop(
        grid,  # type: ignore[arg-type]
        n_clients=n_clients,
        **kwargs,  # type: ignore[arg-type]
    )


# --- the site handler ------------------------------------------------------


def test_a_build_replies_with_a_scored_model() -> None:
    context = node_context(0)
    reply = handle_train_message(_message("build"), context, single(discrete_chain()))  # type: ignore[arg-type]

    dag = record_to_dag(reply.content["local"], 3)  # type: ignore[arg-type]
    assert dag.num_edges == 2
    assert float(reply.content["control"]["score"]) < 0  # type: ignore[arg-type]
    assert "local_dag" in context.state.array_records


def test_a_fuse_merges_the_global_model_in() -> None:
    context = node_context(0)
    loader = single(discrete_chain())
    handle_train_message(_message("build"), context, loader)  # type: ignore[arg-type]
    global_dag = DAG.from_edges(3, [(0, 2)])

    reply = handle_train_message(
        _message("fuse", **{"global": dag_to_record(global_dag)}),
        context,
        loader,  # type: ignore[arg-type]
    )

    skeleton = {frozenset(e) for e in record_to_dag(reply.content["local"], 3).edges()}  # type: ignore[arg-type]
    assert frozenset((0, 2)) in skeleton
    for phase in ("fuse", "dance"):  # a fuse before any build, or no phase at all
        with pytest.raises(ClientFailureError):
            handle_train_message(_message(phase), node_context(0), single(discrete_chain()))  # type: ignore[arg-type]


# --- the server loop ---------------------------------------------------------


def test_the_server_refuses_what_would_give_a_wrong_answer() -> None:
    with pytest.raises(ClientFailureError, match="needs 3"):
        _run(_partition_grid(discrete_chain(), 3, fail_nodes={1}), 3, n_rounds=2)
    tolerated = _run(_partition_grid(discrete_chain(), 3, fail_nodes={1}), 3, min_clients=2)
    assert tolerated is not None
    with pytest.raises(ConfigurationError, match="min_clients"):
        _run(_partition_grid(discrete_chain(), 2), 2, min_clients=0)
    # Edges travel as bare indices, so mismatched variables would be a wrong answer.
    grid = _grid([single(discrete_chain(n_vars=3)), single(discrete_chain(n_vars=4))])
    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        _run(grid, 2, n_rounds=2)


# --- agreement with the in-process loop ---------------------------------------


def _inprocess_globals(data: DiscreteDataset, n_sites: int, n_rounds: int) -> list[DAG]:
    sites = [Site(part) for part in contiguous_split(data, n_sites)]
    globals_: list[DAG] = []

    def collect(stage: str, round_: int, site: int, model: BNModel) -> None:
        if stage == "global":
            globals_.append(model.dag)

    run_rounds(
        lambda: [s.learn() for s in sites],
        lambda g: [s.merge(g) for s in sites],
        Consensus(50),
        default_convergence(),
        n_rounds,
        collect,
    )
    return globals_


def test_flower_gives_the_in_process_models_round_by_round() -> None:
    data = discrete_chain(n_vars=5, n_rows=3000, seed=2)
    expected = _inprocess_globals(data, 3, 8)
    seen: list[DAG] = []

    context = server_context()
    final = _run(
        _partition_grid(data, 3),
        3,
        n_rounds=8,
        on_model=lambda stage, r, s, m: seen.append(m.dag) if stage == "global" else None,
        context=context,
    )

    assert seen == expected
    assert final == expected[-1]
    assert "global_dag" in context.state.array_records


@pytest.mark.slow
def test_the_flower_simulation_gives_the_in_process_models() -> None:
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower import run_flower_fedges

    data = discrete_chain(n_vars=5, n_rows=3000, seed=2)
    expected = _inprocess_globals(data, 3, 8)
    assert run_flower_fedges(contiguous_split(data, 3), n_rounds=8) == expected[-1]


# --- a deployment where each site reads its own file -------------------------


def _write_sites(tmp_path: Path, drop_column_at_last: bool = False) -> list[Path]:
    data = discrete_chain(n_vars=3, n_rows=1800, seed=11)
    paths = []
    for i, part in enumerate(contiguous_split(data, 3)):
        frame = part.to_pandas()
        if drop_column_at_last and i == 2:
            frame = frame.drop(columns=frame.columns[1])
        path = tmp_path / f"site{i}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


class _DeploymentGrid:
    """Supernodes configured as the deployment guide says: a data path, nothing else."""

    def __init__(self, paths: list[Path]) -> None:
        self._contexts = [
            Context(
                run_id=1,
                node_id=100 + i,
                node_config={"data-path": str(p)},
                state=RecordDict(),
                run_config={},
            )
            for i, p in enumerate(paths)
        ]
        self._loader = NodeConfigLoader(DiscreteDataset)

    def get_node_ids(self) -> list[int]:
        return list(range(len(self._contexts)))

    def send_and_receive(self, messages: list[Message]) -> list[Message]:
        return [
            handle_train_message(msg, self._contexts[i], self._loader)
            for i, msg in enumerate(messages)
        ]


def test_a_deployment_learns_from_every_site(tmp_path: Path) -> None:
    locals_per_round: dict[int, int] = {}

    def count(stage: str, round_: int, site: int, model: BNModel) -> None:
        if stage == "local":
            locals_per_round[round_] = locals_per_round.get(round_, 0) + 1

    final = _run(_DeploymentGrid(_write_sites(tmp_path)), 3, on_model=count)

    assert final is not None
    assert set(locals_per_round.values()) == {3}
    grid = _DeploymentGrid(_write_sites(tmp_path, drop_column_at_last=True))
    with pytest.raises((SchemaMismatchError, ValueError)):
        _run(grid, 3)
