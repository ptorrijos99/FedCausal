# fedcausal

[![CI](https://github.com/ptorrijos99/fedcausal/actions/workflows/ci.yml/badge.svg)](https://github.com/ptorrijos99/fedcausal/actions/workflows/ci.yml)
[![Docs](https://github.com/ptorrijos99/fedcausal/actions/workflows/docs.yml/badge.svg)](https://ptorrijos99.github.io/fedcausal/)
[![PyPI](https://img.shields.io/pypi/v/fedcausal.svg)](https://pypi.org/project/fedcausal/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ptorrijos99/fedcausal/blob/main/LICENSE)

fedcausal learns causal structure from data split across sites that cannot share them. The sites send only what each method needs, never their rows:

| Estimator | Data | What crosses the network |
|---|---|---|
| `FedGES` | Discrete or continuous, across sites | Structures and scores |
| `FedRCD` | Continuous, linear non-Gaussian, across sites | Additive moment sums |

Both follow scikit-learn's conventions: hyperparameters go to the constructor, `fit` learns, and results end in an underscore. They run in one process for experiments, or over [Flower](https://flower.ai) across real machines with the same round loop. Learning from data in one place (GES, XGES, the order and exact searches, and the scores, graphs and metrics the federated methods build on) is the job of [causalbase](https://github.com/ptorrijos99/causalbase), which fedcausal depends on.

## Installation

```bash
pip install fedcausal
```

This installs causalbase with it. The Flower transport is an extra, `pip install "fedcausal[federated]"`, and Flower's simulation engine, which runs on Ray, another one, `pip install "fedcausal[federated-simulation]"`.

## Quick start

```python
from causalbase.datasets import load_alarm
from fedcausal import FedGES
from fedcausal.data import contiguous_split

data = load_alarm().data  # 37 variables, 5000 rows, bundled with causalbase

sites = contiguous_split(data, 5)  # or one DataFrame per site
federated = FedGES(n_jobs=-1).fit(sites)  # the sites learn in parallel
federated.dag_, federated.history_
```

```python
from fedcausal import FedRCD

model = FedRCD().fit([x_site1, x_site2, x_site3])  # one array or DataFrame per site
model.causal_order_, model.coef_
```

## How the package is organised

| Layer | Modules | Needs |
|---|---|---|
| The methods | `fedcausal.fedges`, `fusion`, `lingam`, `data` (the site splitters) | causalbase; they run in one process |
| The transport | `fedcausal.flower` | `pip install "fedcausal[federated]"` |

The methods never import Flower, and `tests/test_layout.py` checks that, along with a run with Flower made unimportable. Importing `fedcausal.flower` without the extra fails with a message naming it.

## Documentation

- [Installation and quick start](https://ptorrijos99.github.io/fedcausal/quickstart.html)
- [FedGES](https://ptorrijos99.github.io/fedcausal/algorithms.html) and [FedRCD](https://ptorrijos99.github.io/fedcausal/fedrcd.html): what each method does and what its options change.
- [Deployment](https://ptorrijos99.github.io/fedcausal/deployment.html): FedGES or FedRCD across real machines with Flower.
- [API reference](https://ptorrijos99.github.io/fedcausal/api.html) and [examples](https://ptorrijos99.github.io/fedcausal/auto_examples/index.html).
- [causalbase's documentation](https://ptorrijos99.github.io/causalbase/), for the data, scores, searches, graphs and metrics.

## Development

```bash
git clone https://github.com/ptorrijos99/fedcausal
cd fedcausal
pip install -e ".[dev,federated]"
pytest
```

[CONTRIBUTING.md](https://github.com/ptorrijos99/fedcausal/blob/main/CONTRIBUTING.md) has the checks a pull request must pass; the documentation builds with `pip install -e ".[docs,federated]"` and `sphinx-build -b html docs docs/_build/html`.

## Citing

If you use FedGES, please cite:

> Torrijos, P., Gámez, J.A. & Puerta, J.M. FedGES: A Federated Learning Approach for Bayesian Network Structure Learning. *Machine Learning* **115**, 9 (2026). https://doi.org/10.1007/s10994-025-06939-2

and for FedRCD:

> Torrijos, P., Stella, F., Gámez, J.A. & Puerta, J.M. Federated Causal Discovery via Regression-Directed Cumulants. Probabilistic Graphical Models (PGM), 2026.

[`CITATION.cff`](https://github.com/ptorrijos99/fedcausal/blob/main/CITATION.cff) carries both, for GitHub's "Cite this repository" button.

## License

MIT. See [LICENSE](https://github.com/ptorrijos99/fedcausal/blob/main/LICENSE).
