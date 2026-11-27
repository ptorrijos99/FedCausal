# Validation

What the test suite checks, so you can decide how much to trust it.

```bash
pip install -e ".[dev,federated]"
pytest
```

No downloads and no environment variables: everything the tests need ships with the package.

| Layer | How it is checked |
|---|---|
| Fusion operators | Acyclicity, coverage of every input, and budget limits, over random DAGs in different orders |
| FedGES | Learns the recorded global model on two bundled samples; sequential and parallel sites learn the same models; every refinement keeps the site's score; convergence rules |
| FedRCD | The solvers and the cumulants against the original implementation's recorded output; the cumulants of sites' sums against the pooled k-statistics |
| Flower transports | The same models as the in-process loop, round by round, over a stub grid that hands each message to the real site handler; failed sites and mismatched variables stop the run; the app in `flower_app/` runs both families and writes its result |
| Estimators | Hyperparameters round-trip through `get_params`, `set_params` and `sklearn.base.clone`; `FedRCD` gives what its solver gives over the pooled statistics; a single table, an empty one or an out-of-range setting is refused with an error that names it |
| Layout | The methods never import Flower and run without it; the transport names the extra it needs |
| Documentation | Every `from fedcausal... import ...` shown in the docs runs, and every example in a docstring prints what it shows |

## Test selection

```bash
pytest -m "not slow"        # skip the long-running ones
```

The slow tests drive a real Flower simulation and need Ray (`fedcausal[federated-simulation]`).
