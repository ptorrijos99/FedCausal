# fedcausal

Learn causal structure across several sites without moving any data between them.

Two data regimes, an algorithm family each. For **discrete** data, fedcausal implements [FedGES](https://doi.org/10.1007/s10994-025-06939-2): each site runs greedy equivalence search on the data it holds, a coordinator combines the resulting graphs into one consensus structure and sends it back, and the sites refine against it. Only graphs and scores cross the network. For **continuous** data under a linear non-Gaussian model it implements the FedRCD family: each site reduces its rows to nine cumulant statistics, and a causal order and coefficient matrix fall out of the aggregate. Only those statistics cross the network.

Either family runs in a single process for experiments, or over [Flower](https://flower.ai) for a real deployment. Both paths use the same code, and a test asserts they produce identical models round by round.

```bash
pip install git+https://github.com/ptorrijos99/fedcausal
```

```python
from fedcausal.datasets import load_asia
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

data = load_asia().data                  # 8 variables, 5000 rows, bundled
cpdag = GES(BDeuScore(data)).fit(data)   # a CPDAG: the learned equivalence class
print(cpdag.directed_edges())
```

Federating that across five sites is one call:

```python
from fedcausal.fedges import run_fedges

global_model, stats = run_fedges(data, n_clients=5, n_iterations=20)
```

## Documentation

**[ptorrijos99.github.io/fedcausal](https://ptorrijos99.github.io/fedcausal/)** — full documentation, API reference and example gallery.

[Quickstart](docs/quickstart.md) is the place to start, and the [example gallery](examples/) runs with no setup. After that:

| | |
|---|---|
| [Install](docs/install.md) | the extras, and which ones you actually need |
| [Bring your own data](docs/bring-your-own-data.md) | what the loader expects and what it refuses |
| [Algorithms](docs/algorithms.md) | the search, the fusion operators, and what each option costs |
| [FedRCD](docs/fedrcd.md) | the continuous-data family |
| [Deployment](docs/deployment.md) | a federation across real machines |
| [Benchmarks](docs/benchmarks.md) and [Validation](docs/validation.md) | how fast, how accurate, and what is checked |

The site is built and published from `main` by CI. To build it from a checkout:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Why this exists

Scale, first of all. Score-based structure learning over discrete data gets expensive fast. causal-learn, the closest Python comparison, does not finish `andes` at 223 variables inside half an hour. fedcausal gets through the whole size ladder up to `munin` at 1041 variables, single-threaded and in pure Python. The optional numba extra speeds up the dominant path check, but nothing needs compiling for the base install to work.

Then federation. Plenty of data cannot leave the institution that holds it, while a graph usually can. fedcausal is built for that case, which is why the server never touches client data at any point: each site scores its own model and reports one number, and anything that needs a ground-truth network to evaluate against happens outside the loop, through an observer.

## Layout

```
fedcausal/
  data  datasets  graph  io  metrics   datasets, graph types, XMLBIF, SHD/SMHD
  scores  search  fusion               BDeu, GES, the structural fusion operators
  fedges                               the FedGES round loop, in one process
  lingam                               the FedRCD family and its baselines
  flower                               both families over Flower, plus the shared plumbing
```

The algorithm packages never import `flwr`. Adding a method means a package beside `fedges` and `lingam`, and one module in `flower` if it federates. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If you use fedcausal, cite the FedGES paper:

> Torrijos, P., Gámez, J.A. & Puerta, J.M. FedGES: A Federated Learning Approach for Bayesian Network Structure Learning. *Machine Learning* **115**, 9 (2026). https://doi.org/10.1007/s10994-025-06939-2

```bibtex
@article{torrijos2026fedges,
  title   = {{FedGES}: A Federated Learning Approach for {Bayesian} Network Structure Learning},
  author  = {Torrijos, Pablo and G{\'a}mez, Jos{\'e} A. and Puerta, Jos{\'e} M.},
  journal = {Machine Learning},
  volume  = {115},
  number  = {1},
  pages   = {9},
  year    = {2026},
  doi     = {10.1007/s10994-025-06939-2},
}
```

If you use FedRCD specifically, also cite the PGM paper it comes from:

> Torrijos, P., Stella, F., Gámez, J.A. & Puerta, J.M. Federated Causal Discovery via Regression-Directed Cumulants. Probabilistic Graphical Models (PGM), 2026.

```bibtex
@inproceedings{torrijos2026fedrcd,
  title     = {Federated Causal Discovery via Regression-Directed Cumulants},
  author    = {Torrijos, Pablo and Stella, Fabio and G{\'a}mez, Jos{\'e} A. and Puerta, Jos{\'e} M.},
  booktitle = {Probabilistic Graphical Models (PGM)},
  year      = {2026},
}
```

The proceedings are not out yet, so pages, volume and DOI are still missing from that entry. `CITATION.cff` carries the same, so GitHub's "Cite this repository" button works.

## Credits

Pablo Torrijos Arenas, Universidad de Castilla-La Mancha.

The search follows Chickering (2002); the fusion follows Puerta et al. (*Information Fusion*, 2021). The original Java implementation is [BayesFL](https://github.com/ptorrijos99/BayesFL), which the regression tests pin against.

MIT licensed.
