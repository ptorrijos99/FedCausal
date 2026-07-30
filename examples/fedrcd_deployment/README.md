# A deployable FedRCD federation

Three sites recover a shared causal order and coefficient matrix over continuous data without any of them sending a row anywhere. Each SuperNode reads one CSV from its own disk; the server only ever sees the nine moment-sum arrays each site computes locally, never the data behind them.

This is the deployment path: real processes, and on real machines, different hosts. For a single-process run, use `fedcausal.lingam.flower.run_flower_rcd` / `run_flower_rcd_exact` or `flwr run` against the bundled example instead.

## What each site needs

One CSV of numeric columns, with a header row. **Every site must have the same columns, in the same order.** Statistics are aggregated as bare arrays with no variable names attached, so a mismatch would silently mean something different at each end. The server checks a fingerprint of each site's variable names on the first round and stops the run if they disagree.

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

Set `num-clients` in `pyproject.toml` to your site count first, and `solver` to whichever member of the family you want. To override without editing the file:

```bash
flwr run . remote --run-config "num-clients=5 solver='fedrcd_exact'"
```

`--insecure` is for a trusted network. For anything else, give the SuperLink TLS certificates and drop the flag; see the Flower documentation on securing a deployment.

## What to change for your own federation

`fedrcd_site/app.py` is about sixty lines, and everything worth changing is visible in it.

`NodeConfigContinuousLoader()` decides where the data comes from. It reads a CSV path from the node config, but anything callable that takes a Flower `Context` and returns a `ContinuousDataset` works, so loading from a database or an institutional extract means writing one function.

`solver` in `[tool.flwr.app.config]` picks the algorithm: `fedrcd` (the default, iterative one-shot deflation), `fedrcd_noniterative` or `ishc_beta` (its cheaper one-shot siblings), `fedishc` / `fedhc` (the FedISHC baselines), or `fedrcd_exact` (the multi-round protocol that recomputes fresh statistics from each site's residuals every round instead of deflating the aggregated statistics algebraically). See [`docs/fedrcd.md`](../../docs/fedrcd.md) for what the trade-off between them actually is.

`result-path` is where the server writes the final `causal_order` and `b` matrix as JSON once the run finishes -- a `ServerApp`'s `main` has no return value of its own, so this is how you get the result off the coordinating machine.

## If a site drops out

By default a round needs every site, and the run stops with a `ClientFailureError` naming the ones that failed. That is deliberate: a run that quietly learns from two sites when you configured three gives you a number you cannot interpret.

Set `min-clients` to accept fewer.
