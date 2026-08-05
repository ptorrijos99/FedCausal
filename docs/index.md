# fedcausal

Learn causal structure across several sites without moving any data between them.

For discrete data, fedcausal implements FedGES: each site runs greedy equivalence search on the data it holds, a coordinator combines the resulting graphs into one consensus structure and sends it back, and the sites refine against it. Only graphs and scores cross the network. For continuous data under a linear non-Gaussian model it implements the FedRCD family: each site reduces its rows to a handful of cumulant statistics, and a causal order and coefficient matrix fall out of those. Only the statistics cross the network.

Either family runs in a single process for experiments, or over [Flower](https://flower.ai) for a real deployment. Both paths use the same code, and a test asserts they produce identical models round by round.

```python
from fedcausal.datasets import load_asia
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

data = load_asia().data
cpdag = GES(BDeuScore(data)).fit(data)
print(cpdag.directed_edges())
```

```{toctree}
:maxdepth: 1
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 1
:caption: User guide

bring-your-own-data
algorithms
fedrcd
deployment
```

```{toctree}
:maxdepth: 1
:caption: Examples

auto_examples/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
benchmarks
validation
```

Contributing, the test markers, and what the regression fixtures are for: [CONTRIBUTING.md](https://github.com/ptorrijos99/fedcausal/blob/main/CONTRIBUTING.md).
