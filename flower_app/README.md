# A fedcausal federation over Flower

One app, both method families, and the same code whether it runs in one process or across three machines. `algorithm = "fedges"` learns a Bayesian network structure from discrete data; `algorithm = "fedrcd"` recovers a causal order and coefficient matrix from continuous data. The server only ever sees graphs and scores in the first case, and moment sums in the second. Never a row.

## Trying it in one process

```bash
pip install -e ".[flower-simulation]"   # from the repository root
flwr run flower_app
flwr run flower_app --run-config "algorithm='fedrcd'"
```

That runs three sites inside one process on a bundled synthetic chain, so you need no data files. It does need Ray, which is what the `flower-simulation` extra pulls in. A real deployment does not.

## Running it across machines

Install the app on every machine that takes part:

```bash
pip install -e .          # from this directory
```

Start the SuperLink on the coordinating host:

```bash
flower-superlink --insecure
```

Then one SuperNode per site, each pointed at its own file:

```bash
flower-supernode --insecure \
    --superlink coordinator.example.org:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config "data-path='/data/site1.csv'"
```

Repeat with a different `--clientappio-api-address` and `data-path` per site, then start the run:

```bash
flwr run . remote --run-config "num-clients=3 num-variables=12"
```

`--insecure` is for a trusted network. Anywhere else, give the SuperLink TLS certificates and drop the flag.

## What each site needs

One CSV with a header row: categorical columns for `fedges`, numeric ones for `fedrcd`. **Every site must hold the same columns, in the same order** (and, for `fedges`, with the same levels). Graphs and statistics travel as bare column indices, so a mismatch would mean something different at each end. The server checks a fingerprint on the first round and stops the run if they disagree.

The rows differ. That is the whole point.

## What to change for your own federation

`fedcausal_site/app.py` is short, and everything worth changing is visible in it.

`_loader` decides where the data comes from. A supernode started with a `data-path` reads that file and nothing else; one started without falls back to the synthetic dataset in `data.py`. Anything callable that takes a Flower `Context` and returns a dataset works there, so loading from a database or an institutional extract means writing one function.

`ConsensusUnion()` on the client and `default_server_fusion()` on the server decide how FedGES combines structures. Swap in one of the budgeted operators from `fedcausal.fusion` for a sparser consensus. `default_convergence()` decides when to stop: when the mean local score stops improving, or when every site repeats a model it has produced before.

`solver` picks which member of the FedRCD family runs. `result-path` is where the server writes the final causal order and coefficient matrix, since a `ServerApp`'s `main` has no return value of its own.

## If a site drops out

A round needs every site by default, and the run stops with a `ClientFailureError` naming the ones that failed. A run that quietly learns from two sites when you configured three would give you a number nobody can interpret. Set `min-clients` if you would rather tolerate dropouts.
