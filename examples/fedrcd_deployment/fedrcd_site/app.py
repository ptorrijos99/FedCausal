"""A deployable FedRCD federation: each site loads only its own continuous data.

Unlike the bundled simulation example, nothing here has access to more than one
site's data. Each supernode is started with a ``data-path`` in its node config
and reads that file and nothing else, which is what makes this runnable across
real machines.

Settings come from ``run_config`` so the same app serves any federation and any
solver in the family:

    flwr run . --run-config "num-clients=3 solver='fedrcd_exact'"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import Grid, ServerApp

from fedcausal.lingam.flower import (
    NodeConfigContinuousLoader,
    handle_deflate_message,
    handle_stats_message,
    run_exact_server_loop,
    run_rcd_server_loop,
)

if TYPE_CHECKING:
    from flwr.common import Message

app_client = ClientApp()
app_server = ServerApp()


@app_client.train()
def train(msg: Message, context: Context) -> Message:
    """Compute this site's local statistics, named by its ``data-path`` node config.

    ``fedrcd_exact`` is stateful across rounds (it deflates its working matrix
    in place), the other solvers are stateless one-shot replies; the ``solver``
    run-config key picks which handler answers, so the same client app serves
    every member of the family.
    """
    loader = NodeConfigContinuousLoader()
    solver = str(context.run_config.get("solver", "fedrcd"))
    if solver == "fedrcd_exact":
        return handle_deflate_message(msg, context, loader)
    return handle_stats_message(msg, context, loader)


@app_server.main()
def main(grid: Grid, context: Context) -> None:
    """Coordinate the federation. The server never sees any site's raw rows."""
    config = context.run_config
    solver = str(config.get("solver", "fedrcd"))
    n_clients = int(cast(int, config["num-clients"]))
    min_clients_cfg = config.get("min-clients")
    min_clients = None if min_clients_cfg is None else int(cast(int, min_clients_cfg))

    if solver == "fedrcd_exact":
        result = run_exact_server_loop(grid, n_clients, min_clients=min_clients)
    else:
        result = run_rcd_server_loop(grid, n_clients, solver, min_clients=min_clients)

    out_path = Path(str(config.get("result-path", "rcd_result.json")))
    out_path.write_text(
        json.dumps({"causal_order": list(result.causal_order), "b": result.b.tolist()})
    )
