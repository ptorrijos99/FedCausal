"""GES never ends worse than the reference implementation's recorded runs.

The runs use the compatibility setting ``h_pool="bayesfl"``. They are compared
on BDeu, which every member of an equivalence class shares, and on the CPDAG
when the scores tie. On ``child`` and ``alarm`` fedcausal finds a better
optimum; see ``docs/validation.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG, PDAG
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

GOLDEN_DIR = Path(__file__).resolve().parent
NETWORKS = GOLDEN_DIR.parent / "fixtures" / "networks"

GOLDEN = json.loads((GOLDEN_DIR / "ges_runs.json").read_text())


def _reference_cpdag(run: dict[str, Any], names: tuple[str, ...]) -> PDAG:
    """The reference run's equivalence class, with its edges resolved by name."""
    index = {name: i for i, name in enumerate(names)}
    edges = [(index[u], index[v]) for u, v in run["directed_edges"]]
    return dag_to_cpdag(DAG.from_edges(len(names), edges))


# Far below any score difference a different search step produces, far above
# the rounding of summing a few hundred lgamma terms.
_REL_TOL = 1e-9


@pytest.mark.regression
@pytest.mark.parametrize("run", GOLDEN["runs"], ids=lambda r: r["net"])
def test_ges_is_never_worse_than_the_reference(run: dict[str, Any]) -> None:
    ds = DiscreteDataset.from_csv(NETWORKS / f"{run['dataset']}.gz")

    score = BDeuScore(ds, sample_prior=run["sample_prior"], structure_prior=run["structure_prior"])
    result = GES(score, h_pool="bayesfl").fit(ds)

    ours = score.score_dag(result.to_dag())
    reference = run["bdeu"]
    tol = abs(reference) * _REL_TOL
    assert ours >= reference - tol, f"{run['net']}: BDeu {ours} is worse than {reference}"

    if abs(ours - reference) <= tol:
        assert result == _reference_cpdag(run, ds.names), (
            f"{run['net']}: same score as the reference, different equivalence class"
        )


@pytest.mark.regression
def test_the_empty_t_gate_explains_child() -> None:
    """With the reference's empty-T gate on, GES lands exactly on its ``child`` result."""
    run = next(r for r in GOLDEN["runs"] if r["net"] == "child")
    ds = DiscreteDataset.from_csv(NETWORKS / f"{run['dataset']}.gz")

    score = BDeuScore(ds, sample_prior=run["sample_prior"], structure_prior=run["structure_prior"])
    result = GES(score, h_pool="bayesfl", empty_t_gate=True).fit(ds)

    assert result == _reference_cpdag(run, ds.names)
