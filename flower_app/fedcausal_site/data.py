"""Synthetic datasets, so the app runs with no data files at all.

Both are chains: every variable depends on the one before it and on nothing
else, which makes the answer obvious enough to eyeball. Replace these with your
own loader for anything real -- see the README.
"""

from __future__ import annotations

import numpy as np

from fedcausal.data import ContinuousDataset, DiscreteDataset

N_VARS = 4
N_ROWS = 4000


def synthetic_discrete(seed: int = 0) -> DiscreteDataset:
    """Return a binary chain X0 -> X1 -> X2 -> X3, each copying its parent 90% of the time."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, N_ROWS)]
    for _ in range(N_VARS - 1):
        previous = cols[-1]
        cols.append(np.where(rng.random(N_ROWS) < 0.9, previous, 1 - previous))
    values = np.stack(cols, axis=1).astype(np.int32)
    cards = np.array([2] * N_VARS, dtype=np.int32)
    names = [f"X{i}" for i in range(N_VARS)]
    return DiscreteDataset(values, cards, names, [["0", "1"]] * N_VARS)


def synthetic_continuous(seed: int = 0) -> ContinuousDataset:
    """Return the same chain over real values, with Laplace noise.

    Non-Gaussian noise is not a detail: under Gaussian noise the causal order is
    not identifiable from the joint distribution at all.
    """
    rng = np.random.default_rng(seed)
    e = rng.laplace(0.0, 1.0, size=(N_ROWS, N_VARS))
    x = np.zeros((N_ROWS, N_VARS))
    x[:, 0] = e[:, 0]
    for j in range(1, N_VARS):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return ContinuousDataset.from_arrays(x, [f"X{i}" for i in range(N_VARS)])
