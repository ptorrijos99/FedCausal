# Quickstart

## Learn a structure from one table

```python
import pandas as pd

from fedcausal import GES

frame = pd.read_csv("survey.csv", dtype=str)
model = GES().fit(frame)
model.adjacency_
```

Every distinct value of a column is one level, so numbers are read as labels too. `fit` learns a CPDAG, the equivalence class of DAGs the data cannot tell apart. In `adjacency_`, `A[i, j] = 1` with `A[j, i] = 0` is a compelled edge `i -> j`, one every DAG of the class shares; both set is an edge whose direction the data does not decide. `cpdag_` is the same graph over column indices, and `dag_` is one DAG of the class.

## Measure it against a known network

```python
from fedcausal.datasets import load_asia
from fedcausal.metrics import edge_scores, shd

bundle = load_asia()                       # data sampled from a known network
model = GES().fit(bundle.data)
shd(bundle.truth, model.dag_)              # structural Hamming distance
edge_scores(bundle.truth, model.dag_).f1_skel
```

`bundle.truth` is already indexed like the data's columns. To compare against a network of your own, load it with `fedcausal.io.load_xbif` and re-index it onto the data's column order first.

## Learn across sites

```python
from fedcausal import FedGES

sites = [pd.read_csv(f"site{i}.csv", dtype=str) for i in range(3)]
model = FedGES(n_jobs=-1).fit(sites)
model.dag_, model.history_
```

Each site's table stays a separate input: its rows are only ever seen by its own GES. Levels are pooled across sites, so a value seen at one site is a level everywhere. `n_jobs=-1` runs the sites in parallel processes. To simulate sites from one table, `fedcausal.data.contiguous_split` and `dirichlet_split` split it.

## Learn a linear model across sites

```python
from fedcausal import FedRCD

model = FedRCD().fit([x1, x2, x3])         # one numeric array or DataFrame per site
model.causal_order_, model.coef_
```

## Tune like any scikit-learn estimator

```python
model = GES(expected_parents=0.5, ess=10)
model.set_params(max_t_size=None).get_params()
```

`sklearn.base.clone` works on all three estimators. `GES.score(X)` is the mean log likelihood of held-out rows, so a parameter search can choose the hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

frame = load_asia().data.to_pandas()
search = GridSearchCV(GES(), {"expected_parents": [0.5, 1.0, 2.0]}, cv=3).fit(frame)
search.best_params_
```

Continuous data takes `GES(criterion="gaussian_bic")`.

## Next

[Algorithms](algorithms.md) explains every option. [Deployment](deployment.md) runs a federation across real machines.
