"""Recover a causal order from continuous data split across sites, with FedRCD.

Generates a small linear non-Gaussian SEM, splits it across simulated clients,
and runs the one-shot and multi-round FedRCD solvers over the aggregated
cumulant statistics -- never the raw rows:

    python examples/03_fedrcd.py
"""

from __future__ import annotations

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


def _make_sem(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build a chain SEM x0 -> x1 -> ... -> x{N_VARS-1} with skewed noise.

    Centred exponential noise is skewed (skewness 2), which is what the
    non-Gaussianity of LiNGAM-family methods needs: under Gaussian noise the
    causal order is not identifiable from the joint distribution at all.
    """
    b = np.zeros((N_VARS, N_VARS))
    for child in range(1, N_VARS):
        b[child, child - 1] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])

    e = rng.exponential(1.0, size=(N_ROWS, N_VARS)) - 1.0  # centred, skewed
    x = np.zeros((N_ROWS, N_VARS))
    for j in range(N_VARS):
        x[:, j] = e[:, j] + x @ b[j]
    return x, b


def main() -> None:
    """Split a synthetic SEM across sites and recover its causal order."""
    rng = np.random.default_rng(0)
    values, b_true = _make_sem(rng)
    data = ContinuousDataset.from_arrays(values, NAMES)
    truth = DAG.from_edges(
        N_VARS,
        [(p, c) for c in range(N_VARS) for p in range(N_VARS) if b_true[c, p] != 0.0],
    )

    print(f"{data.n_rows} rows, {data.n_vars} variables across {N_CLIENTS} sites")
    print(f"true causal order: {' -> '.join(NAMES)}\n")

    clients = contiguous_split(data, N_CLIENTS)

    # Only nine moment-sum arrays per client cross the network here, never a
    # row of `values`: compute_local_raw_stats is purely additive, so the
    # server can sum what each client sends without seeing any sample.
    raw_stats = [compute_local_raw_stats(c.values, order=4) for c in clients]
    stats = aggregate_and_compute_cumulants(raw_stats)

    # One-shot: a single round of communication, deflating the aggregated
    # cumulants algebraically on the server.
    one_shot = fedrcd(stats)
    order_names = " -> ".join(NAMES[i] for i in one_shot.causal_order)
    print(f"fedrcd (one round):     {order_names}")
    print(f"  SHD against truth:    {shd(truth, one_shot.to_dag(threshold=0.3))}")

    # Multi-round: p - 1 rounds, each client re-deflating its own local data
    # and sending fresh statistics, which avoids the algebraic error the
    # one-shot deflation can accumulate under finite samples.
    exact = fedrcd_exact([c.values for c in clients])
    order_names = " -> ".join(NAMES[i] for i in exact.causal_order)
    print(f"fedrcd_exact ({N_VARS - 1} rounds): {order_names}")
    print(f"  SHD against truth:    {shd(truth, exact.to_dag(threshold=0.3))}")


if __name__ == "__main__":
    main()
