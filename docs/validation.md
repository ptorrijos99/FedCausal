# Validation

What the test suite checks, so you can decide how much to trust it.

```bash
pip install -e ".[dev,federated]"
pytest
```

No downloads and no environment variables: everything the tests need ships with the package.

| Layer | How it is checked |
|---|---|
| Scores | BDeu and BIC against independent count-and-`lgamma` implementations, including parent sets with more configurations than any integer code can hold, and the Gaussian BIC against a regression; a batch of families scores the same, bit for bit, as each family alone |
| Graph kernels | The compiled CPDAG completion, consistent extension and path search against the dense reference implementations they replaced, over random graphs |
| GES | Learns, on four bundled samples, the score and SHD recorded in the tests, so any change in what it learns shows; stops where no single Insert or Delete improves the score; never ends below the score of the graph it starts from; its Inserts are those the operator's definition gives |
| CPDAG completion | Two random DAGs share a CPDAG exactly when they have the same skeleton and v-structures (Verma & Pearl), and every CPDAG extends back into its own class |
| Fusion operators | Acyclicity, coverage of every input, and budget limits, over random DAGs in different orders |
| Metrics | SHD, SMHD and the edge scores against their definitions over edge sets |
| FedGES | Learns the recorded global model on two bundled samples; sequential and parallel sites learn the same models; every refinement keeps the site's score; convergence rules |
| FedRCD | The solvers and the cumulants against the original implementation's recorded output; the cumulants of sites' sums against the pooled k-statistics |
| Flower transports | The same models as the in-process loop, round by round, over a stub grid that hands each message to the real site handler; failed sites and mismatched variables stop the run; the app in `flower_app/` runs both families and writes its result |
| Estimators | Hyperparameters round-trip through `get_params`, `set_params` and `sklearn.base.clone`; `FedRCD` gives what its solver gives over the pooled statistics; a single table, an empty one or an out-of-range setting is refused with an error that names it |
| Layout | The core never imports the federated part, and runs with it and Flower made unimportable; the transport names the extra it needs |
| Documentation | Every `from fedcausal... import ...` shown in the docs runs, and every example in a docstring prints what it shows |

Search results are compared through scores and equivalence classes rather than edge lists where it matters. A search returns one member of a Markov equivalence class, and the choice among its members is arbitrary; BDeu and the CPDAG are the same for every member.

## Test selection

```bash
pytest -m "not slow"        # skip the long-running ones
```

The slow tests drive a real Flower simulation and need Ray (`fedcausal[federated-simulation]`).
