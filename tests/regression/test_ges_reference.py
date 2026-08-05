"""Pin ``GES`` against committed reference search runs.

The reference runs come from the implementation fedcausal was validated
against, in its compatibility configuration (``h_pool="bayesfl"``); see
``docs/algorithms.md``. BDeu is the binding assertion rather than the raw
directed-edge set, because a search returns one arbitrary member of the Markov
equivalence class while BDeu is identical across every DAG extension of a
CPDAG. The CPDAG itself is compared separately, in CPDAG space.

Two networks are expected to fail: on ``child`` and ``alarm`` fedcausal's
search reaches a strictly better optimum than the reference did. They are
marked ``strict`` so the suite fails loudly if that ever stops being true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fedcausal.datasets import load_network
from fedcausal.graph import DAG
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

GOLDEN_DIR = Path(__file__).resolve().parent

GOLDEN = json.loads((GOLDEN_DIR / "ges_runs.json").read_text())

# SHD is not invariant across a Markov equivalence class: it counts within-MEC
# reversals, so any consistent extension of our CPDAG can differ from the
# reference's specific pick by a few arbitrary reversals even when the CPDAGs
# are identical. Asserted as a tolerant upper bound rather than pinned exactly.
# SMHD moralizes both graphs and is MEC-invariant, so it is pinned.
_SHD_SLACK = 2

_BETTER_THAN_REFERENCE = {
    "child": (
        "fedcausal's FES considers every T subset independently, while the "
        "reference gates each candidate edge on the T=empty case before ever "
        "trying a non-empty T. Reproduced bit-exactly by emulating that gate. "
        "fedcausal's BDeu is strictly better, so this is not a defect."
    ),
    "alarm": (
        "The reference's returned graph is not a local optimum of its own "
        "operator: an admissible Insert(VENTLUNG, INTUBATION, T=empty) with "
        "delta +749.05 remains available in it, closing exactly the BDeu gap. "
        "Root cause unestablished. fedcausal's BDeu is strictly better, so "
        "this is not a defect."
    ),
}


def _param(run: dict[str, Any]) -> Any:
    net = run["net"]
    if net in _BETTER_THAN_REFERENCE:
        return pytest.param(
            run,
            id=net,
            marks=pytest.mark.xfail(strict=True, reason=_BETTER_THAN_REFERENCE[net]),
        )
    return pytest.param(run, id=net)


RUN_PARAMS = [_param(r) for r in GOLDEN["runs"]]


def _mapped_dag(n: int, name_to_idx: dict[str, int], named_edges: list[list[str]]) -> DAG:
    """Build a `DAG` over `n` nodes from a golden's named edge list, resolved by name."""
    return DAG.from_edges(n, [(name_to_idx[u], name_to_idx[v]) for u, v in named_edges])


@pytest.mark.regression
@pytest.mark.parametrize("run", RUN_PARAMS)
def test_ges_matches_reference(run: dict[str, Any]) -> None:
    """Pin fedcausal's GES (BDeu, CPDAG, SMHD) against a reference run."""
    net = run["net"]
    bundle = load_network(net)
    ds = bundle.data
    # Every name->index remap below silently compares the wrong nodes if the
    # column order ever disagrees with the golden's own variable order.
    assert list(ds.names) == run["variables"], (
        f"{net}: column order {list(ds.names)} != golden variables {run['variables']}"
    )
    name_to_idx = {name: i for i, name in enumerate(ds.names)}

    score = BDeuScore(ds, sample_prior=run["sample_prior"], structure_prior=run["structure_prior"])
    result = GES(score, h_pool="bayesfl").fit(ds)

    our_bdeu = score.score_dag(result.to_dag())
    assert our_bdeu == pytest.approx(run["bdeu"], rel=1e-6), (
        f"{net}: our BDeu {our_bdeu} != golden {run['bdeu']} (delta {our_bdeu - run['bdeu']})"
    )

    golden_cpdag = dag_to_cpdag(_mapped_dag(ds.n_vars, name_to_idx, run["directed_edges"]))
    assert result == golden_cpdag, f"{net}: our CPDAG != golden's CPDAG"

    # Check the .xbif-derived ground truth rather than trusting the golden's
    # committed truth_edges blindly: this is the only place load_xbif's
    # ground-truth parsing, and the re-indexing `load_network` does on top of
    # it, are cross-validated against an independent source.
    assert set(bundle.network.names) == set(ds.names), (
        f"{net}: xbif variables {sorted(bundle.network.names)} "
        f"!= dataset variables {sorted(ds.names)}"
    )
    truth_dag = bundle.truth
    assert truth_dag == _mapped_dag(ds.n_vars, name_to_idx, run["truth_edges"]), (
        f"{net}: load_xbif ground truth != golden's committed truth_edges"
    )

    our_dag = result.to_dag()
    our_smhd = smhd(our_dag, truth_dag)
    assert our_smhd == run["smhd_vs_truth"], (
        f"{net}: our SMHD {our_smhd} != golden {run['smhd_vs_truth']}"
    )

    our_shd = shd(our_dag, truth_dag)
    assert our_shd <= run["shd_vs_truth"] + _SHD_SLACK, (
        f"{net}: our SHD {our_shd} exceeds golden {run['shd_vs_truth']} + slack {_SHD_SLACK}"
    )
