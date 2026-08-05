"""
Continuous data: FedRCD
=======================

The other data regime. Where FedGES federates by exchanging *graphs*, the FedRCD
family federates by exchanging *sufficient statistics*: each site reduces its
rows to nine moment sums, and a causal order and coefficient matrix fall out of
the aggregate.

This needs a linear non-Gaussian model. Under Gaussian noise the causal order is
not identifiable from the joint distribution at all, which is why the noise below
is skewed.
"""

# %%
# A chain ``x0 -> x1 -> ... -> x4`` with centred exponential noise, which has
# skewness 2. That asymmetry is the signal the whole method runs on.

import numpy as np

from fedcausal.data import ContinuousDataset, contiguous_split
from fedcausal.graph import DAG
from fedcausal.lingam import (
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
    fedrcd,
    fedrcd_exact,
)
from fedcausal.metrics import shd

N_VARS = 5
N_ROWS = 4000
N_CLIENTS = 4
NAMES = [f"x{i}" for i in range(N_VARS)]

rng = np.random.default_rng(0)

b_true = np.zeros((N_VARS, N_VARS))
for child in range(1, N_VARS):
    b_true[child, child - 1] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])

e = rng.exponential(1.0, size=(N_ROWS, N_VARS)) - 1.0  # centred, skewed
values = np.zeros((N_ROWS, N_VARS))
for j in range(N_VARS):
    values[:, j] = e[:, j] + values @ b_true[j]

data = ContinuousDataset.from_arrays(values, NAMES)
truth = DAG.from_edges(
    N_VARS,
    [(p, c) for c in range(N_VARS) for p in range(N_VARS) if b_true[c, p] != 0.0],
)

print(f"{data.n_rows} rows, {data.n_vars} variables across {N_CLIENTS} sites")
print(f"true causal order: {' -> '.join(NAMES)}")

# %%
# What crosses the network. ``compute_local_raw_stats`` reduces a client's whole
# matrix to nine additive moment-sum arrays plus a row count. Additivity is the
# only property the server needs: it sums them and applies bias-corrected
# k-statistic formulas to get unbiased cumulant estimates.
#
# A client's rows cannot be recovered from its nine arrays. They are aggregates
# over its whole local sample, not a reversible encoding of it.

clients = contiguous_split(data, N_CLIENTS)

raw_stats = [compute_local_raw_stats(c.values, order=4) for c in clients]
stats = aggregate_and_compute_cumulants(raw_stats)

print(f"arrays sent per client: {len(raw_stats[0])}")

# %%
# One shot. A single round of communication: the server deflates the aggregated
# cumulants algebraically, identifying the most exogenous variable at each step.

one_shot = fedrcd(stats)

print(f"recovered order: {' -> '.join(NAMES[i] for i in one_shot.causal_order)}")
print(f"SHD against truth: {shd(truth, one_shot.to_dag(threshold=0.3))}")

# %%
# Multi-round. ``fedrcd_exact`` spends ``p - 1`` rounds instead, each client
# deflating its *own rows* and sending fresh statistics, so every round's
# cumulants come from real data rather than from already-deflated estimates.
#
# It costs more traffic and earns it back when finite-sample noise would let the
# one-shot deflation drift, which tends to mean small ``p`` or noise close to the
# Gaussian boundary. On a clean chain like this one, both get it right.

exact = fedrcd_exact([c.values for c in clients])

print(f"recovered order: {' -> '.join(NAMES[i] for i in exact.causal_order)}")
print(f"SHD against truth: {shd(truth, exact.to_dag(threshold=0.3))}")
