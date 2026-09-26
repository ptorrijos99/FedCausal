"""FedGES over Flower gives the same models as the in-process loop, round by round.

The main check runs over `StubGrid`, so everywhere; with Ray installed it runs
once more through a real simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from fedcausal.data import DiscreteDataset, contiguous_split  # noqa: E402
from fedcausal.fedges import Client, GESLearner, Server  # noqa: E402
from fedcausal.fedges.defaults import (  # noqa: E402
    default_client_fusion,
    default_convergence,
    default_server_fusion,
)
from fedcausal.fedges.observer import ModelCollector  # noqa: E402
from fedcausal.flower.fedges import run_server_loop  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402

from ._stubs import server_context  # noqa: E402
from .test_fedges_server_loop import _grid  # noqa: E402

N_CLIENTS = 3
N_ITERATIONS = 8


def _ds(n_rows: int = 3000, seed: int = 2) -> DiscreteDataset:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.random(n_rows) < 0.9, x1, 1 - x1)
    values = np.stack([x0, x1, x2], axis=1).astype(np.int32)
    return DiscreteDataset(values, np.array([2, 2, 2], np.int32), ["A", "B", "C"], [["0", "1"]] * 3)


def _inprocess_globals(ds: DiscreteDataset) -> list[object]:
    parts = contiguous_split(ds, N_CLIENTS)
    clients = [Client(default_client_fusion(), GESLearner(), p) for p in parts]
    server = Server(
        default_server_fusion(), default_convergence(), clients, n_iterations=N_ITERATIONS
    )
    collector = ModelCollector()
    server.run(collector)
    return list(collector.globals)


def test_flower_matches_inprocess_round_by_round() -> None:
    ds = _ds()
    expected = _inprocess_globals(ds)

    parts = contiguous_split(ds, N_CLIENTS)
    grid = _grid([InMemoryPartitions(parts) for _ in range(N_CLIENTS)])
    collector = ModelCollector()
    final = run_server_loop(
        grid,  # type: ignore[arg-type]
        server_context(),
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=N_CLIENTS,
        num_nodes=ds.n_vars,
        n_iterations=N_ITERATIONS,
        observer=collector,
    )

    assert collector.globals == expected  # BNModel equality, round by round
    assert final == expected[-1]


@pytest.mark.slow
def test_flower_simulation_matches_inprocess() -> None:
    """The same check through a real simulation, when Ray is installed."""
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower import run_flower_fedges

    ds = _ds()
    expected = _inprocess_globals(ds)

    collector = ModelCollector()
    final = run_flower_fedges(
        ds, n_clients=N_CLIENTS, n_iterations=N_ITERATIONS, observer=collector
    )

    assert collector.globals == expected
    assert final == expected[-1]
