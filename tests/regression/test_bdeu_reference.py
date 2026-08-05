"""Pin ``BDeuScore`` against committed reference local scores.

Each golden file records the priors its scores were computed under, so the
score object is configured from the file rather than from hardcoded values.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fedcausal.datasets import load_network
from fedcausal.scores import BDeuScore

GOLDEN_DIR = Path(__file__).resolve().parent

BDEU_GOLDENS = ["bdeu_scores_asia.json", "bdeu_scores_alarm.json", "bdeu_scores_asia_sp.json"]


@pytest.mark.regression
@pytest.mark.parametrize("golden_name", BDEU_GOLDENS)
def test_bdeu_matches_reference(golden_name: str) -> None:
    golden = json.loads((GOLDEN_DIR / golden_name).read_text(encoding="ascii"))

    # "asia.0.csv" names the sampled dataset bundled with the `asia` network.
    ds = load_network(golden["dataset"].split(".")[0]).data
    score = BDeuScore(
        ds,
        sample_prior=golden["sample_prior"],
        structure_prior=golden["structure_prior"],
    )

    name_to_idx = {name: i for i, name in enumerate(ds.names)}

    for rec in golden["scores"]:
        node = name_to_idx[rec["node"]]
        # Resolve parents by name: level encodings need not agree with the
        # reference, but the family being scored is identified by name.
        parents = tuple(sorted(name_to_idx[p] for p in rec["parents"]))
        got = score.local_score(node, parents)
        expected = rec["score"]
        if expected is None:
            assert not math.isfinite(got), (
                f"{rec['node']}|{rec['parents']}: reference is non-finite, got {got}"
            )
        else:
            assert got == pytest.approx(expected, rel=1e-9), (
                f"{rec['node']}|{rec['parents']}: got {got}, expected {expected}"
            )
