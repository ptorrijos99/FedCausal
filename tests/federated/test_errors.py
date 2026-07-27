"""Invalid configurations and degenerate runs fail loudly, not silently."""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.exceptions import (
    ConfigurationError,
    ConvergedBeforeFusionError,
    FedCausalError,
)
from fedcausal.federated import Client, GESLearner, Server, run_fedges
from fedcausal.federated.model import BNModel
from fedcausal.fusion import Consensus, ConsensusUnion
from fedcausal.graph import DAG


def _ds(n_rows: int = 500, seed: int = 0) -> DiscreteDataset:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_rows)
    x1 = np.where(rng.random(n_rows) < 0.9, x0, 1 - x0)
    values = np.stack([x0, x1], axis=1).astype(np.int32)
    return DiscreteDataset(values, np.array([2, 2], np.int32), ["A", "B"], [["0", "1"]] * 2)


class _AlwaysConverged:
    """A criterion that converges on its first sighting, before any fusion."""

    def check_convergence(self, local_models: object) -> bool:
        return True


def test_unknown_refinement_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Unknown refinement: 'fes'"):
        GESLearner(refinement="fes")  # type: ignore[arg-type]


def test_known_refinements_are_accepted() -> None:
    for stage in ("None", "FES", "BES", "GES"):
        assert GESLearner(refinement=stage).refinement_name == stage  # type: ignore[arg-type]


def test_server_rejects_an_empty_federation() -> None:
    with pytest.raises(ConfigurationError, match="at least one client"):
        Server(Consensus(50), _AlwaysConverged(), [])


def test_server_rejects_negative_iterations() -> None:
    client = Client(ConsensusUnion(), GESLearner(), _ds())
    with pytest.raises(ConfigurationError, match="non-negative"):
        Server(Consensus(50), _AlwaysConverged(), [client], n_iterations=-1)


def test_fuse_before_build_raises() -> None:
    client = Client(ConsensusUnion(), GESLearner(), _ds())
    with pytest.raises(FedCausalError, match="before build_local_model"):
        client.fuse(BNModel(DAG(2)))


def test_converging_before_any_fusion_raises() -> None:
    """A criterion firing on round 1 leaves no global model; say so, don't assert."""
    with pytest.raises(ConvergedBeforeFusionError, match="without fusing"):
        run_fedges(_ds(), n_clients=2, n_iterations=3, convergence=_AlwaysConverged())


def test_zero_iterations_raises() -> None:
    with pytest.raises(ConvergedBeforeFusionError):
        run_fedges(_ds(), n_clients=2, n_iterations=0)


def test_learner_reuses_one_score_per_dataset() -> None:
    """The memoised BDeu score survives across calls for the same partition."""
    learner = GESLearner()
    data = _ds()
    first = learner._score(data)
    assert learner._score(data) is first
    assert learner._score(_ds(seed=1)) is not first
