# Deployment

Running fedcausal across real machines, where each site holds its own data and nothing but graphs or statistics crosses the network. A complete, runnable app is in [`flower_app/`](https://github.com/ptorrijos99/fedcausal/tree/main/flower_app); this page explains what it does and why.

## Simulation or deployment

Two different things, and it is worth being clear about which you want.

Simulation runs every site inside one process. It is what you want for experiments, and `run_flower_fedges`, `run_flower_rcd`, or `flwr run flower_app` gets you there in a line. It needs Flower's Ray-based engine, so install `fedcausal[flower-simulation]` (see [Install](install.md)).

Deployment runs each site as its own process, usually on its own machine. It needs `fedcausal[flower]` only, no Ray. This page is about that.

## What each site provides

One CSV with a header row: categorical columns for FedGES, numeric ones for the FedRCD family. Every site must hold the same variables in the same order, and for FedGES with the same levels. Only the rows differ.

Both transports index everything by column position, so a mismatch would mean something different at each end. The server checks it on the first round: each client sends a fingerprint of its variable names, and if the fingerprints disagree the run stops with a `SchemaMismatchError` naming the variable counts it saw. Better a stopped run than a plausible wrong answer.

## The app

The client side of a round is one function call, whichever family you are running:

```python
from flwr.client import ClientApp

from fedcausal.data import DiscreteDataset
from fedcausal.flower.fedges import handle_train_message
from fedcausal.flower.loaders import NodeConfigLoader
from fedcausal.fusion import ConsensusUnion

app_client = ClientApp()


@app_client.train()
def train(msg, context):
    loader = NodeConfigLoader(DiscreteDataset)
    return handle_train_message(msg, context, loader, ConsensusUnion())
```

For the continuous family, swap `handle_train_message` for `fedcausal.flower.lingam.handle_stats_message` and `DiscreteDataset` for `ContinuousDataset`. Nothing else changes.

The server side is one loop:

```python
from flwr.server import ServerApp

from fedcausal.fedges.defaults import default_convergence, default_server_fusion
from fedcausal.flower.fedges import run_server_loop

app_server = ServerApp()


@app_server.main()
def main(grid, context):
    config = context.run_config
    run_server_loop(
        grid,
        context,
        global_fusion=default_server_fusion(),
        convergence=default_convergence(),
        n_clients=int(config["num-clients"]),
        num_nodes=int(config["num-variables"]),
        n_iterations=int(config["num-server-rounds"]),
    )
```

`NodeConfigLoader` is what makes this a deployment rather than a simulation. It reads a path out of the supernode's own `node_config`, so each process opens one file and has no way of reaching anyone else's. Anything callable that takes a Flower `Context` and returns a dataset works in its place, so loading from a database or an institutional extract means writing one function:

```python
def load_from_warehouse(context):
    site = context.node_config["site-id"]
    return DiscreteDataset.from_pandas(query_cohort(site))
```

## Starting it

Install the app on every participating machine, then start the SuperLink on the coordinator:

```bash
flower-superlink --insecure
```

One SuperNode per site, each pointed at its own file:

```bash
flower-supernode --insecure \
    --superlink coordinator.example.org:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config "data-path='/data/site1.csv'"
```

Then start the run:

```bash
flwr run . remote --run-config "num-variables=12 num-server-rounds=30"
```

`--insecure` is for a trusted network. Anywhere else, give the SuperLink TLS certificates and drop the flag.

## What happens each round

FedGES needs two message round trips per iteration. In the build phase every client rebuilds its local model with GES, warm-started from the model in its own `context.state`, and replies with its DAG and the score that DAG achieves on its own data. The server never sees the data behind that number. In the fuse phase the server combines the local DAGs into a global one and broadcasts it; each client fuses that into its own model, optionally refines, and updates its state. Convergence is checked between the two, on the local models the server just received.

The FedRCD family is cheaper. Every solver but `fedrcd_exact` needs exactly one round trip: each client sends nine moment-sum arrays, and the server aggregates them into cumulants and solves. `fedrcd_exact` repeats that `p - 1` times, deflating each client's own rows between rounds. See [FedRCD](fedrcd.md).

## When a site drops out

A round needs every site by default, and one that fails stops the run with a `ClientFailureError` naming the nodes that errored. The alternative is worse: a run that quietly learns from four sites when you configured five produces a number nobody can interpret, and nothing in the output says so. Set `min_clients` if you would rather tolerate dropouts.

## Getting the result out

The server writes the final structure to `context.state`, which is process-local. Persist it yourself at the end of `main`:

```python
final = run_server_loop(grid, context, ...)
if final is not None:
    Path("global_dag.json").write_text(json.dumps(final.dag.edges()))
```

Both server loops return their result as well as storing it, so there is no need to dig it back out of the state record.

## Things worth knowing

The server holds no data and computes no scores. That is a design constraint, not an accident, and it is what lets the loops run unchanged over any transport. Metrics against a ground-truth network are computed by an observer outside the loop, so a real federation with no ground truth simply attaches no observer. And neither `fedcausal.fedges` nor `fedcausal.lingam` imports `flwr`, so the algorithms work without the extra installed.

## Environment snags

Standing up a real three-site federation on Windows turned up three problems, all of them Flower-version or console issues rather than anything to do with fedcausal:

- `flower-supernode`'s default `--isolation subprocess` failed with a SuperExec `UNAUTHENTICATED` gRPC error against `flwr==1.32.1`. `--isolation process` worked.
- `flwr run` crashed with a `UnicodeEncodeError` on a `cp1252` console, because Flower's own CLI banner contains an emoji that console cannot encode. `PYTHONIOENCODING=utf-8` fixed it.
- The SuperLink's auto-spawned `flower-superexec` failed with `unrecognized arguments: --allow-runtime-dependency-installation`, a mismatch inside that `flwr` build between the SuperLink and the SuperExec it spawns.

Expect the same running any Flower app on a similar setup.
