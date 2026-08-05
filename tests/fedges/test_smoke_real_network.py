"""End-to-end FedGES smoke test on forward-sampled asia (committed fixture)."""

from __future__ import annotations

import pytest

from fedcausal.datasets import load_asia
from fedcausal.fedges import run_fedges
from fedcausal.io import forward_sample
from fedcausal.metrics import shd


@pytest.mark.slow
def test_fedges_on_sampled_asia_recovers_a_nontrivial_structure():
    bn = load_asia().network
    ds = forward_sample(bn, 5000, seed=0)
    global_model, stats = run_fedges(ds, n_clients=5, n_iterations=20, truth=bn.dag)

    assert global_model.dag.num_nodes == ds.n_vars
    assert len(stats) > 0
    # The federated union recovers a non-trivial structure (asia's truth has 8
    # directed edges); the fused global model should be in the right ballpark.
    assert global_model.dag.num_edges >= 4
    # Per-round stats carry SHD vs. truth when a ground-truth DAG is supplied.
    server_rows = [s for s in stats if s.stage == "server"]
    assert server_rows and all(s.shd is not None for s in server_rows)
    # The final global model is a reasonable recovery, not noise.
    assert shd(bn.dag, global_model.dag) <= 12
