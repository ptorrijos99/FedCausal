"""One Flower app that runs either method family, in simulation or deployed.

The `algorithm` run-config key picks the family: `fedges` for discrete data,
`fedrcd` for continuous. Everything else -- how many sites, how many rounds,
which solver -- also comes from `run_config`, so the same app serves any
federation without editing this file. For `fedrcd`, `method` and `order` are
those of :class:`fedcausal.FedRCD`.

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
from fedcausal.flower import fedges, lingam
from fedcausal.flower.loaders import InMemoryPartitions, NodeConfigLoader

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
    """Do this site's share of a round, on its own data."""
    algorithm = str(_config(context, "algorithm", "fedges"))
    loader = _loader(context, algorithm)

    if algorithm == "fedges":
        return fedges.handle_train_message(msg, context, loader)
    return lingam.handle_train_message(msg, context, loader)


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
    """Run the FedGES loop and write the global DAG's edges to `result-path`."""
    final = fedges.run_server_loop(
        grid,
        n_clients=n_clients,
        n_rounds=int(_config(context, "num-server-rounds", 10)),
        min_clients=min_clients,
        context=context,
    )
    if final is None:
        raise RuntimeError("the run converged before producing a global model")
    _write_result(context, {"edges": final.edges()})
    print(f"global model with {final.num_edges} edges")


def _run_fedrcd(grid: Grid, context: Context, n_clients: int, min_clients: int | None) -> None:
    """Run the selected FedRCD method and write its result to `result-path`."""
    result = lingam.run_server_loop(
        grid,
        n_clients=n_clients,
        method=str(_config(context, "method", "deflation")),
        order=int(_config(context, "order", 4)),
        min_clients=min_clients,
    )
    _write_result(
        context, {"causal_order": list(result.causal_order), "coef": result.coef.tolist()}
    )
    print(f"causal order {result.causal_order}")


def _write_result(context: Context, result: dict[str, Any]) -> None:
    """Write `result` as JSON to `result-path`; a ``ServerApp`` cannot return it."""
    out = Path(str(_config(context, "result-path", "result.json")))
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"result written to {out}")
