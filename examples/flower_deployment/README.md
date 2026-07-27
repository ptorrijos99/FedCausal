# A deployable FedGES federation

Three sites learn a shared Bayesian network structure without any of them
sending data anywhere. Each SuperNode reads one CSV from its own disk; the
server only ever sees graphs and scores.

This is the deployment path — real processes, and on real machines, different
hosts. For a single-process run, use `fedcausal.flower.run_flower_fedges` or
`flwr run` against the bundled example instead.

## What each site needs

One CSV of categorical columns, with a header row. **Every site must have the
same columns, in the same order, with the same set of levels.** Structures
travel between sites as bare column indices, so a mismatch would silently mean
something different at each end — the server checks the schemas on the first
round and stops the run if they disagree.

The rows differ, of course. That is the point.

## Running it

Install the app on every machine that will take part:

```bash
pip install -e .          # from this directory
```

Start the SuperLink on the coordinating host:

```bash
flower-superlink --insecure
```

Start one SuperNode per site, each pointed at its own data:

```bash
# site 1
flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config "data-path='/data/site1.csv'"

# site 2
flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config "data-path='/data/site2.csv'"

# site 3
flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config "data-path='/data/site3.csv'"
```

Then start the run:

```bash
flwr run . remote
```

Set `num-variables` in `pyproject.toml` to your column count first. To override
without editing the file:

```bash
flwr run . remote --run-config "num-variables=12 num-server-rounds=30"
```

`--insecure` is for a trusted network. For anything else, give the SuperLink TLS
certificates and drop the flag; see the Flower documentation on securing a
deployment.

## What to change for your own federation

`fedcausal_site/app.py` is about forty lines, and the three things worth
changing are visible in it:

- **Where data comes from.** `NodeConfigLoader()` reads a CSV path from the node
  config. Anything callable that takes a Flower `Context` and returns a
  `DiscreteDataset` works, so loading from a database or a hospital extract
  means writing one function.
- **How structures are combined.** `ConsensusUnion()` on the client and
  `default_server_fusion()` on the server. `fedcausal.fusion` has the budgeted
  variants — `MaxParents`, `MaxEdges`, `Consensus`, `MaxFrequency` — for sparser
  consensus structures.
- **When to stop.** `default_convergence()` stops when the mean local score
  stops improving or every site repeats a model it has produced before.

## If a site drops out

By default a round needs every site, and the run stops with a
`ClientFailureError` naming the ones that failed. That is deliberate: a run that
quietly learns from two sites when you configured three gives you a number you
cannot interpret.

Set `min-clients` to accept fewer.
