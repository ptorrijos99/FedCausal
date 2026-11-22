"""Load a benchmark network and its sampled data from ``FEDCAUSAL_NETWORKS``.

The directory holds ``xbif/<net>.xbif`` and ``BBDD/<net>/<net>.<seed>.csv.gz``,
the layout of the Bayesian Network Repository samples these benchmarks use.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG
from fedcausal.io import load_xbif


def root() -> Path:
    """Return the networks directory, or exit with a message if it is not set."""
    path = os.environ.get("FEDCAUSAL_NETWORKS")
    if not path:
        raise SystemExit("set FEDCAUSAL_NETWORKS to the directory holding xbif/ and BBDD/")
    return Path(path)


def load(net: str, seed: int = 0, n_rows: int = 5000) -> tuple[DiscreteDataset, DAG]:
    """Return the first `n_rows` of sample `seed` of `net`, and its true DAG on those columns."""
    base = root()
    data = DiscreteDataset.from_csv(base / "BBDD" / net / f"{net}.{seed}.csv.gz")
    if data.n_rows > n_rows:
        data = data.subset(np.arange(n_rows))
    network = load_xbif(base / "xbif" / f"{net}.xbif")
    index = {name: i for i, name in enumerate(data.names)}
    truth = DAG.from_edges(
        data.n_vars,
        [(index[network.names[u]], index[network.names[v]]) for u, v in network.dag.edges()],
    )
    return data, truth
