# fedcausal

Causal structure learning from data spread across sites that cannot share it.

| Estimator | Data | What crosses the network |
|---|---|---|
| {class}`~fedcausal.fedges.FedGES` | Discrete or continuous, across sites | Structures and scores |
| {class}`~fedcausal.lingam.FedRCD` | Continuous, linear non-Gaussian, across sites | Additive moment sums |

Learning from data in one place (GES, XGES, the scores and graphs these methods build on) is [causalbase](https://github.com/ptorrijos99/causalbase)'s job.

```python
from fedcausal import FedGES
from fedcausal.data import contiguous_split
from causalbase.datasets import load_alarm

sites = contiguous_split(load_alarm().data, 5)
model = FedGES(n_jobs=-1).fit(sites)
model.adjacency_
```

Federated runs work in one process for experiments, or over [Flower](https://flower.ai) across real machines, with the same round loop in both.

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

How to contribute: [CONTRIBUTING.md](https://github.com/ptorrijos99/fedcausal/blob/main/CONTRIBUTING.md).
