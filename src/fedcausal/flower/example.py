"""A self-contained FedGES federation for ``flwr run .``.

Exposes the module-level ``server_app`` and ``client_app`` that ``[tool.flwr]``
in ``pyproject.toml`` points at, wired to a small synthetic dataset so the
example runs with no data files.

Everything is read from ``run_config`` at run time, so configuration drives the
federation rather than constants baked in here::

    flwr run .
    flwr run . --run-config "num-server-rounds=10 num-clients=3"

Keep ``num-clients`` and the federation's ``options.num-supernodes`` in step.

For a federation over real data on real machines, see
``examples/flower_deployment/`` and ``docs/flower-deployment.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import Grid, ServerApp

from fedcausal.data import DiscreteDataset, contiguous_split
from fedcausal.federated.defaults import default_convergence, default_server_fusion
from fedcausal.flower.app import handle_train_message, run_server_loop
from fedcausal.flower.data import InMemoryPartitions
from fedcausal.fusion import ConsensusUnion

if TYPE_CHECKING:
    from flwr.common import Message

__all__ = ["client_app", "server_app"]

_DEFAULT_CLIENTS = 5
_DEFAULT_ROUNDS = 5


def _synthetic_dataset(n_rows: int = 4000, seed: int = 0) -> DiscreteDataset:
    """Build a 4-variable binary chain X0 -> X1 -> X2 -> X3 with strong dependence."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, n_rows)]
    for _ in range(3):
        previous = cols[-1]
        cols.append(np.where(rng.random(n_rows) < 0.9, previous, 1 - previous))
    values = np.stack(cols, axis=1).astype(np.int32)
    cards = np.array([2, 2, 2, 2], dtype=np.int32)
    return DiscreteDataset(values, cards, ["X0", "X1", "X2", "X3"], [["0", "1"]] * 4)


def _n_clients(context: Context) -> int:
    """Return the client count from ``run_config``, or the default."""
    return int(cast(int, context.run_config.get("num-clients", _DEFAULT_CLIENTS)))


client_app = ClientApp()
server_app = ServerApp()


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    """Learn on this supernode's slice of the synthetic dataset."""
    partitions = contiguous_split(_synthetic_dataset(), _n_clients(context))
    return handle_train_message(msg, context, InMemoryPartitions(partitions), ConsensusUnion())


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    """Drive the loop with the round count and client count from ``run_config``."""
    dataset = _synthetic_dataset()
    run_server_loop(
        grid,
        context,
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=_n_clients(context),
        num_nodes=dataset.n_vars,
        n_iterations=int(cast(int, context.run_config.get("num-server-rounds", _DEFAULT_ROUNDS))),
    )
