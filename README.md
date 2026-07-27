# fedcausal

Learn the structure of a Bayesian network across several sites without moving any data between them.

fedcausal implements [FedGES](https://doi.org/10.1007/s10994-025-06939-2) (Torrijos, Gámez & Puerta, *Machine Learning* 115, 2026). Each site runs greedy equivalence search on the data it holds. A coordinator combines the resulting graphs into one consensus structure and sends it back, and the sites refine against it. Only graphs and scores cross the network.

It runs that loop in a single process for experiments, and over [Flower](https://flower.ai) for a real deployment. Both use the same code, and a test asserts they produce identical models round by round.

```python
from fedcausal.data import DiscreteDataset
from fedcausal.federated import run_fedges

data = DiscreteDataset.from_csv("data.csv")
global_model, stats = run_fedges(data, n_clients=5, n_iterations=20)
print(global_model.dag.edges())
```

## Install

```bash
pip install git+https://github.com/ptorrijos99/fedcausal
```

Optional extras:

```bash
pip install "fedcausal[flower]"      # the Flower transport
pip install "fedcausal[accel]"       # numba, for large graphs
```

## Why this exists

Two reasons, and scale is the first one.

Score-based structure learning over discrete data gets expensive fast. causal-learn, the closest Python comparison, does not finish `andes` at 223 variables inside half an hour. fedcausal gets through the whole size ladder up to `munin` at 1041 variables, single-threaded and in pure Python, with numbers in [Benchmarks](docs/benchmarks.md). The optional numba extra speeds up the dominant path check, but nothing needs compiling for the base install to work.

The second is federation. Plenty of data cannot leave the institution that holds it, while a graph usually can. fedcausal is built for that case, which is why the server never touches client data at any point: each site scores its own model and reports one number, and anything that needs a ground-truth network to evaluate against happens outside the loop, through an observer.

## Learning one structure

```python
from fedcausal.data import DiscreteDataset
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

data = DiscreteDataset.from_csv("data.csv")
cpdag = GES(BDeuScore(data)).fit(data)   # a PDAG: the learned CPDAG
dag = cpdag.to_dag()                     # any member of the equivalence class
```

`fes_only` and `bes_only` run the individual phases. See [docs/algorithms.md](docs/algorithms.md) for what the knobs do and which ones cost you the optimality guarantee.

## Combining structures across sites

```python
from fedcausal.fusion import consensus_union, MaxParents

fused = consensus_union(client_dags)      # the full consensus union
sparse = MaxParents(3).fuse(client_dags)  # a budgeted variant
```

Every operator transforms its inputs into one consensus order and unions the parent sets, so the result is always acyclic. The budgeted variants rank the transformed edges by how many clients agree on them and add them under a cap: a total edge count (`MaxEdges`), a minimum agreement (`MaxFrequency`), a percentage (`Consensus`), or a per-node limit (`MaxParents`). `Intersection` keeps only what everyone agrees on.

## Running the federation

In one process, which is what you want for experiments:

```python
global_model, stats = run_fedges(data, n_clients=5, n_iterations=20, truth=true_dag)
```

Pass `truth` and each row of `stats` carries SHD and SMHD for that round. `client_fusion`, `server_fusion`, `convergence`, and `refinement` are all overridable.

Over Flower, with each site loading its own data:

```bash
flwr run .                       # the bundled example, on synthetic data
```

For a federation across real machines, see [docs/flower-deployment.md](docs/flower-deployment.md) and the runnable app in [`examples/flower_deployment/`](examples/flower_deployment/). The short version: each SuperNode is started with a path to its own CSV and reads nothing else, and the server checks on the first round that every site agrees on the variables before it trusts any edge.

## Documentation

[Quickstart](docs/quickstart.md) is the place to start. After that: [Bring your own data](docs/bring-your-own-data.md) covers what the loader expects and what it refuses; [Flower deployment](docs/flower-deployment.md) walks through a federation over real machines; [Algorithms](docs/algorithms.md) explains the search, the fusion operators, and the compatibility options. For evidence, see [Validation](docs/validation.md) and [Benchmarks](docs/benchmarks.md).

## Development

```bash
pip install -e ".[dev,flower]"
pytest                    # 4 xfails are expected; see docs/validation.md
ruff check . && ruff format --check .
mypy src
```

The suite is self-contained: no downloads, no environment variables, no JVM. Tests marked `bigdata` need networks too large to ship and skip unless `FEDCAUSAL_NETWORKS` points at a directory holding them.

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

`CITATION.cff` carries the same, so GitHub's "Cite this repository" button works.

## Credits

Pablo Torrijos Arenas, Universidad de Castilla-La Mancha.

The search follows Chickering (2002); the fusion follows Puerta et al. (*Information Fusion*, 2021). The original Java implementation is [BayesFL](https://github.com/ptorrijos99/BayesFL), which the regression tests pin against.

MIT licensed.
