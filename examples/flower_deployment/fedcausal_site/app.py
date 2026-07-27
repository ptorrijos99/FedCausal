"""A deployable FedGES federation: each site loads only its own data.

Unlike the bundled simulation example, nothing here has access to more than one
site's data. Each supernode is started with a ``data-path`` in its node config
and reads that file and nothing else, which is what makes this runnable across
real machines.

Settings come from ``run_config`` so the same app serves any federation:

    flwr run . --run-config "num-server-rounds=20 min-clients=3"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import Grid, ServerApp

from fedcausal.federated.defaults import default_convergence, default_server_fusion
from fedcausal.flower.app import handle_train_message, run_server_loop
from fedcausal.flower.data import NodeConfigLoader
from fedcausal.fusion import ConsensusUnion

if TYPE_CHECKING:
    from flwr.common import Message

app_client = ClientApp()
app_server = ServerApp()


@app_client.train()
def train(msg: Message, context: Context) -> Message:
    """Learn on this site's own data, named by its ``data-path`` node config."""
    return handle_train_message(msg, context, NodeConfigLoader(), ConsensusUnion())


@app_server.main()
def main(grid: Grid, context: Context) -> None:
    """Coordinate the federation. The server never sees any site's data."""
    config = context.run_config
    n_clients = int(cast(int, config["num-clients"]))
    min_clients = config.get("min-clients")

    run_server_loop(
        grid,
        context,
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=n_clients,
        num_nodes=int(cast(int, config["num-variables"])),
        n_iterations=int(cast(int, config["num-server-rounds"])),
        min_clients=None if min_clients is None else int(cast(int, min_clients)),
    )
