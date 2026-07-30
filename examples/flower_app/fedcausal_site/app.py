"""One Flower app that runs either method family, in simulation or deployed.

The `algorithm` run-config key picks the family: `fedges` for discrete data,
`fedrcd` for continuous. Everything else -- how many sites, how many rounds,
which solver -- also comes from `run_config`, so the same app serves any
federation without editing this file.

Where a site's data comes from is the only difference between a simulation and
a deployment. A supernode started with `--node-config "data-path='...'"` reads
that file and nothing else, which is the deployment path. One started without it
falls back to a bundled synthetic dataset, which is what `flwr run .` does.

    flwr run .                                        # simulation, synthetic data
    flwr run . --run-config "algorithm='fedrcd'"      # the continuous family
    flwr run . remote --run-config "num-clients=3"    # real supernodes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import Grid, ServerApp

from fedcausal.data import ContinuousDataset, DiscreteDataset, contiguous_split
from fedcausal.fedges.defaults import default_convergence, default_server_fusion
from fedcausal.flower import fedges, lingam
from fedcausal.flower.loaders import InMemoryPartitions, NodeConfigLoader
from fedcausal.fusion import ConsensusUnion

from .data import synthetic_continuous, synthetic_discrete

if TYPE_CHECKING:
    from flwr.common import Message

app_client = ClientApp()
app_server = ServerApp()


def _config(context: Context, key: str, default: Any = None) -> Any:
    """Read a run-config value, falling back to `default`."""
    return context.run_config.get(key, default)


def _loader(context: Context, algorithm: str) -> Any:
    """Resolve where this supernode's data comes from.

    A `data-path` in the node config means a real deployment: read that file and
    nothing else. Without one, serve a slice of the bundled synthetic dataset.
    """
    if algorithm == "fedrcd":
        dataset_cls, synthetic = ContinuousDataset, synthetic_continuous
    else:
        dataset_cls, synthetic = DiscreteDataset, synthetic_discrete

    if "data-path" in context.node_config:
        return NodeConfigLoader(dataset_cls)
    n_clients = int(_config(context, "num-clients", 5))
    return InMemoryPartitions(contiguous_split(synthetic(), n_clients))


@app_client.train()
def train(msg: Message, context: Context) -> Message:
    """Do this site's share of a round, on its own data.

    FedRCD-Exact is stateful across rounds (each client deflates its working
    matrix in place); every other solver answers a single stateless message.
    """
    algorithm = str(_config(context, "algorithm", "fedges"))
    solver = str(_config(context, "solver", "fedrcd"))
    loader = _loader(context, algorithm)

    if algorithm == "fedges":
        return fedges.handle_train_message(msg, context, loader, ConsensusUnion())
    if solver == "fedrcd_exact":
        return lingam.handle_deflate_message(msg, context, loader)
    return lingam.handle_stats_message(msg, context, loader)


@app_server.main()
def main(grid: Grid, context: Context) -> None:
    """Coordinate the federation. The server never sees any site's rows."""
    algorithm = str(_config(context, "algorithm", "fedges"))
    n_clients = int(_config(context, "num-clients", 5))
    min_clients = _config(context, "min-clients")
    min_clients = None if min_clients is None else int(min_clients)

    if algorithm == "fedges":
        _run_fedges(grid, context, n_clients, min_clients)
    else:
        _run_fedrcd(grid, context, n_clients, min_clients)


def _run_fedges(grid: Grid, context: Context, n_clients: int, min_clients: int | None) -> None:
    """Run the FedGES loop and leave the global DAG on the server's state."""
    n_vars = int(_config(context, "num-variables", 0)) or synthetic_discrete().n_vars
    fedges.run_server_loop(
        grid,
        context,
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=n_clients,
        num_nodes=n_vars,
        n_iterations=int(_config(context, "num-server-rounds", 5)),
        min_clients=min_clients,
    )


def _run_fedrcd(grid: Grid, context: Context, n_clients: int, min_clients: int | None) -> None:
    """Run the selected FedRCD solver and write its result to `result-path`.

    A `ServerApp`'s `main` has no return value, so the coefficient matrix has to
    be persisted here to leave the coordinating machine.
    """
    solver = str(_config(context, "solver", "fedrcd"))
    if solver == "fedrcd_exact":
        result = lingam.run_exact_server_loop(grid, n_clients, min_clients=min_clients)
    else:
        result = lingam.run_server_loop(grid, n_clients, solver, min_clients=min_clients)

    out = Path(str(_config(context, "result-path", "rcd_result.json")))
    out.write_text(
        json.dumps({"causal_order": list(result.causal_order), "b": result.b.tolist()}),
        encoding="utf-8",
    )
    print(f"causal order {result.causal_order}, written to {out}")
