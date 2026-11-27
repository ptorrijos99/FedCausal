# fedcausal

[Source](https://github.com/ptorrijos99/fedcausal) | [Issues](https://github.com/ptorrijos99/fedcausal/issues) | [Changelog](https://github.com/ptorrijos99/fedcausal/blob/main/CHANGELOG.md) | [Citing](citing.md) | [causalbase](https://ptorrijos99.github.io/causalbase/)

fedcausal learns causal structure from data split across sites that cannot share them. The sites send only what each method needs, never their rows:

| Estimator | Data | What crosses the network |
|---|---|---|
| {class}`~fedcausal.fedges.FedGES` | Discrete or continuous, across sites | Structures and scores |
| {class}`~fedcausal.lingam.FedRCD` | Continuous, linear non-Gaussian, across sites | Additive moment sums |

Learning from data in one place (GES, XGES, the scores and graphs these methods build on) is the job of [causalbase](https://ptorrijos99.github.io/causalbase/).

```python
from fedcausal import FedGES
from fedcausal.data import contiguous_split
from causalbase.datasets import load_alarm

sites = contiguous_split(load_alarm().data, 5)
model = FedGES(n_jobs=-1).fit(sites)
model.adjacency_
```

Federated runs work in one process for experiments, or over [Flower](https://flower.ai) across real machines, with the same round loop in both.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: quickstart
:link-type: doc

Installation and a first federated structure.
:::

:::{grid-item-card} Methods
:link: algorithms
:link-type: doc

FedGES and its fusion operators, and what their options change.
:::

:::{grid-item-card} Deployment
:link: deployment
:link-type: doc

FedGES and FedRCD across real machines with Flower.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc

Parameters, attributes and methods of every class and function.
:::
::::

```{toctree}
:maxdepth: 1
:hidden:
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: User guide

bring-your-own-data
algorithms
fedrcd
deployment
auto_examples/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api
benchmarks
validation
citing
Changelog <https://github.com/ptorrijos99/fedcausal/blob/main/CHANGELOG.md>
Contributing <https://github.com/ptorrijos99/fedcausal/blob/main/CONTRIBUTING.md>
```
