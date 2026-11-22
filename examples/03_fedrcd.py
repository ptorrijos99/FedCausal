"""
Learn a linear model from continuous data held by several sites
===============================================================

:class:`~fedcausal.lingam.FedRCD` assumes a linear structural equation model with
non-Gaussian noise, ``x = Bx + e``. Each site sends additive moment sums of
its rows; their totals give third- and fourth-order cumulants, from which the
causal order is identified and ``B`` estimated. No row leaves a site.
"""

import numpy as np

from fedcausal import FedRCD
from fedcausal.graph import DAG
from fedcausal.metrics import shd

rng = np.random.default_rng(0)
n_vars, n_rows = 6, 8000
b = np.zeros((n_vars, n_vars))
for child in range(1, n_vars):
    for parent in range(child):
        if rng.random() < 0.4:
            b[child, parent] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
noise = rng.uniform(-1, 1, size=(n_rows, n_vars))
x = noise @ np.linalg.inv(np.eye(n_vars) - b).T
truth = DAG.from_amat(b.T != 0)

sites = np.array_split(x, 4)

# %%
# ``"deflation"`` needs one round of communication. ``"exact"`` needs one per
# variable but carries no estimation error from one step to the next.

for method in ("deflation", "exact"):
    model = FedRCD(method=method).fit(sites)
    print(f"{method:9s}  order {model.causal_order_}  SHD {shd(truth, model.dag_)}")

# %%
# The published baselines are settings of the same estimator: FedISHC is
# ``FedRCD(method="ratio", order=3)`` and FedHC ``FedRCD(method="sort", order=3)``.

for method in ("ratio", "sort"):
    baseline = FedRCD(method=method, order=3).fit(sites)
    print(f"{method:9s}  order {baseline.causal_order_}  SHD {shd(truth, baseline.dag_)}")

# %%
# ``coef_[child, parent]`` holds the estimated coefficients; the adaptive lasso
# sets the absent ones to zero.

np.round(model.coef_, 2)
