# Flower deployment

Running fedcausal across real machines, where each site holds its own data and
nothing but graphs crosses the network.

A complete, runnable app is in
[`examples/flower_deployment/`](../examples/flower_deployment/). This page
explains what it does and why.

## Simulation or deployment

Two different things, and it is worth being clear which you want.

Simulation runs every site inside one process. It is what you want for
experiments, and `run_flower_fedges` or `flwr run .` against the bundled
example gets you there in a line. It needs Flower's Ray-based simulation
engine, so install `fedcausal[flower-simulation]`. Ray has no wheel for every
Python version and platform, which is why it is a separate extra.

Deployment runs each site as its own process, usually on its own machine. It
needs `fedcausal[flower]` only, no Ray. This page is about that.

## What each site provides

One CSV of categorical columns. Every site must hold the same variables, in the
same order, with the same levels; only the rows differ. Structures travel as
bare column indices, so a mismatch would silently mean something different at
each end.

The server checks this on the first round. Each client sends a fingerprint of
its names and cardinalities, and if they disagree the run stops with a
`SchemaMismatchError` naming the variable counts it saw. Better a stopped run
than a plausible wrong answer.

## The app

```python
from flwr.client import ClientApp
from flwr.server import ServerApp

from fedcausal.federated.defaults import default_convergence, default_server_fusion
from fedcausal.flower.app import handle_train_message, run_server_loop
from fedcausal.flower.data import NodeConfigLoader
from fedcausal.fusion import ConsensusUnion

app_client = ClientApp()
app_server = ServerApp()


@app_client.train()
def train(msg, context):
    return handle_train_message(msg, context, NodeConfigLoader(), ConsensusUnion())


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

`NodeConfigLoader` is the part that makes this a deployment rather than a
simulation. It reads a path out of the supernode's own `node_config`, so each
process opens one file and has no way of reaching anyone else's.

Anything callable that takes a Flower `Context` and returns a
`DiscreteDataset` works in its place, so loading from a database or an
institutional extract means writing one function:

```python
def load_from_warehouse(context):
    site = context.node_config["site-id"]
    return DiscreteDataset.from_pandas(query_cohort(site))
```

## Starting it

Install the app on every participating machine, then start the SuperLink on the
coordinator:

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

`--insecure` is for a trusted network. Anywhere else, give the SuperLink TLS
certificates and drop the flag.

## What happens each round

Two message round trips.

In the build phase every client rebuilds its local model with GES, warm-started
from the model in its own `context.state`, and replies with its DAG and the
score that DAG achieves on its own data. The server never sees the data behind
that number.

In the fuse phase the server combines the local DAGs into a global one and
broadcasts it. Each client fuses that into its own model, optionally refines,
and updates its state.

The server checks convergence between the two, on the local models it just
received. When it stops, the final global model is on
`context.state["global_dag"]`.

## When a site drops out

By default a round needs every site, and one that fails stops the run with a
`ClientFailureError` naming the nodes that errored. This is deliberate. A run
that quietly learns from four sites when you configured five produces a number
you cannot interpret, and the failure is invisible in the output.

Set `min_clients` on `run_server_loop` to accept fewer.

## Getting the result out

In a deployment the server writes the final structure to `context.state`, which
is process-local. Persist it yourself at the end of `main`:

```python
final = run_server_loop(grid, context, ...)
if final is not None:
    Path("global_dag.json").write_text(json.dumps(final.dag.edges()))
```

`run_server_loop` returns the model as well as storing it, so there is no need
to dig it back out of the state record.

## Things worth knowing

The server holds no data and computes no scores. That is a design constraint,
not an accident, and it is what lets the loop run unchanged over any transport.

Metrics against a ground-truth network are computed by an observer outside the
loop, so a real federation with no ground truth simply attaches no observer.

`import fedcausal.federated` never imports `flwr`, so the in-process engine
works without the extra installed.
