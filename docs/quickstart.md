# Quickstart

Learning from one table, measuring the result against a known network and tuning the estimators are covered by [causalbase](https://github.com/ptorrijos99/causalbase).

## Learn across sites

```python
import pandas as pd

from fedcausal import FedGES

sites = [pd.read_csv(f"site{i}.csv", dtype=str) for i in range(3)]
model = FedGES(n_jobs=-1).fit(sites)
model.dag_, model.history_
```

Each site's table stays a separate input: its rows are only ever seen by its own GES. Levels are pooled across sites, so a value seen at one site is a level everywhere. `n_jobs=-1` runs the sites in parallel processes. To simulate sites from one table, `fedcausal.data.contiguous_split` and `dirichlet_split` split it.

## Learn a linear model across sites

```python
from fedcausal import FedRCD

model = FedRCD().fit([x1, x2, x3])  # one numeric array or DataFrame per site
model.causal_order_, model.coef_
```

## Next

[Algorithms](algorithms.md) explains every option. [Deployment](deployment.md) runs a federation across real machines.
