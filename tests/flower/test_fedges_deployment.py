"""The documented deployment path works: each site reads only its own file.

Exercises what `docs/deployment.md` tells people to do, with real CSVs on
disk and each supernode given nothing but a path to its own. If this passes, the
walkthrough in the docs is not lying.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from flwr.common import Context, Message, RecordDict  # noqa: E402

from fedcausal.data import DiscreteDataset, contiguous_split  # noqa: E402
from fedcausal.exceptions import SchemaMismatchError  # noqa: E402
from fedcausal.fedges.defaults import default_convergence, default_server_fusion  # noqa: E402
from fedcausal.fedges.observer import ModelCollector  # noqa: E402
from fedcausal.flower.fedges import handle_train_message, run_server_loop  # noqa: E402
from fedcausal.flower.loaders import NodeConfigLoader  # noqa: E402
from fedcausal.fusion import ConsensusUnion  # noqa: E402

from ._stubs import server_context  # noqa: E402

N_SITES = 3


def _write_sites(tmp_path, n_rows: int = 1800, seed: int = 11, drop_column: int = -1):  # type: ignore[no-untyped-def]
    """Write one CSV per site and return their paths.

    `drop_column` removes a column from the last site's file, which is how a
    schema mismatch gets simulated.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.random(n_rows) < 0.9, x1, 1 - x1)
    values = np.stack([x0, x1, x2], axis=1).astype(np.int32)
    data = DiscreteDataset(values, np.array([2, 2, 2], np.int32), ["A", "B", "C"], [["0", "1"]] * 3)

    paths = []
    for i, part in enumerate(contiguous_split(data, N_SITES)):
        names = list(part.names)
        columns = range(part.n_vars)
        if i == N_SITES - 1 and drop_column >= 0:
            columns = [c for c in columns if c != drop_column]
            names = [names[c] for c in columns]
        lines = [",".join(names)]
        for row in range(part.n_rows):
            lines.append(",".join(part.categories[c][part.values[row, c]] for c in columns))
        path = tmp_path / f"site{i}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


class _DeploymentGrid:
    """Route messages to supernodes configured as the docs say: a data path, nothing else."""

    def __init__(self, paths: list) -> None:  # type: ignore[type-arg]
        self._contexts = [
            Context(
                run_id=1,
                node_id=i,
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
            handle_train_message(msg, self._contexts[i], self._loader, ConsensusUnion())
            for i, msg in enumerate(messages)
        ]


def _run(grid: _DeploymentGrid, observer: ModelCollector | None = None):  # type: ignore[no-untyped-def]
    return run_server_loop(
        grid,  # type: ignore[arg-type]
        server_context(),
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=N_SITES,
        num_nodes=3,
        n_iterations=6,
        observer=observer,
    )


def test_every_site_contributes_to_every_round(tmp_path) -> None:  # type: ignore[no-untyped-def]
    collector = ModelCollector()

    final = _run(_DeploymentGrid(_write_sites(tmp_path)), collector)

    assert final is not None and final.dag.num_edges > 0
    assert len(collector.builds) == N_SITES * (len(collector.globals) + 1)


def test_a_site_with_different_columns_stops_the_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The failure mode the schema check exists to prevent."""
    paths = _write_sites(tmp_path, drop_column=2)

    with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
        _run(_DeploymentGrid(paths))
