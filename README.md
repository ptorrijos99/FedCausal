# fedcausal

Causal structure learning for data in one place or spread across sites that cannot share it.

| Estimator | Data | What crosses the network |
|---|---|---|
| `GES` | Discrete or continuous, in one place | Nothing |
| `FedGES` | Discrete or continuous, across sites | Structures and scores |
| `FedRCD` | Continuous, linear non-Gaussian, across sites | Additive moment sums |

All three follow scikit-learn's conventions: hyperparameters go to the constructor, `fit` learns, and results end in an underscore. `GES` scores categorical data by BDeu or BIC and continuous data by a Gaussian BIC, and its `score(X)`, the held-out log likelihood, lets scikit-learn's parameter searches tune it.

```bash
pip install git+https://github.com/ptorrijos99/fedcausal
```

```python
from fedcausal import GES, FedGES
from fedcausal.data import contiguous_split
from fedcausal.datasets import load_alarm

data = load_alarm().data  # 37 variables, 5000 rows, bundled

model = GES().fit(data)  # also takes a DataFrame or an array
model.adjacency_  # the learned CPDAG, with variable names

sites = contiguous_split(data, 5)  # or one DataFrame per site
federated = FedGES(n_jobs=-1).fit(sites)  # the sites learn in parallel
federated.dag_, federated.history_
```

```python
from fedcausal import FedRCD

model = FedRCD().fit([x_site1, x_site2, x_site3])  # one array or DataFrame per site
model.causal_order_, model.coef_
```

## Speed

GES runs in compiled code where it matters (counting, path checks, CPDAG completion, the consensus order of the fusion) and re-scores only the part of the graph each step changes. On a laptop it learns `andes` (223 variables, 5000 rows) in under a second and `munin` (1041 variables) in 21 s, twenty to forty times faster than 0.1.0; causal-learn does not finish `andes` in half an hour. The [benchmarks](docs/benchmarks.md) have the full ladder.

## Documentation

**[ptorrijos99.github.io/fedcausal](https://ptorrijos99.github.io/fedcausal/)**: the user guide, the API reference and a gallery of examples that run with no setup. To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

[Deployment](docs/deployment.md) explains how to run FedGES or FedRCD across real machines with [Flower](https://flower.ai) (`pip install "fedcausal[flower]"`).

## Citing

FedGES:

> Torrijos, P., Gámez, J.A. & Puerta, J.M. FedGES: A Federated Learning Approach for Bayesian Network Structure Learning. *Machine Learning* **115**, 9 (2026). https://doi.org/10.1007/s10994-025-06939-2

FedRCD:

> Torrijos, P., Stella, F., Gámez, J.A. & Puerta, J.M. Federated Causal Discovery via Regression-Directed Cumulants. Probabilistic Graphical Models (PGM), 2026.

`CITATION.cff` carries both, for GitHub's "Cite this repository" button.

## License

MIT. Pablo Torrijos Arenas, Universidad de Castilla-La Mancha.
