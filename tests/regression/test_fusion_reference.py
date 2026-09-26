"""Structural fusion matches the reference implementation's recorded runs.

Everything is compared exactly except ``max_edges``, which cuts through
frequency ties that the two implementations break differently; there only the
edge count and membership of the union are checked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fedcausal.fusion import consensus_pct, consensus_union, max_edges, max_frequency
from fedcausal.fusion.alpha import alpha_order
from fedcausal.fusion.beta import beta_to_alpha
from fedcausal.graph import DAG

GOLDEN = json.loads((Path(__file__).parent / "fusion_runs.json").read_text(encoding="utf-8"))


def _edges(name_pairs: list[list[str]], index: dict[str, int]) -> set[tuple[int, int]]:
    return {(index[a], index[b]) for a, b in name_pairs}


@pytest.mark.regression
@pytest.mark.parametrize("sc", GOLDEN["scenarios"], ids=lambda s: s["name"])
def test_fusion_matches_java(sc: dict[str, Any]) -> None:
    index = {name: i for i, name in enumerate(sc["variables"])}
    p = len(sc["variables"])
    dags = [DAG.from_edges(p, sorted(_edges(inp, index))) for inp in sc["inputs"]]

    # Consensus order (tie-robust: Java's node-list order == our ascending index).
    alpha = alpha_order(dags)
    assert alpha == [index[n] for n in sc["alpha"]]

    # Each minimal-I-map transform.
    for d, beta_edges in zip(dags, sc["betas"], strict=True):
        assert set(beta_to_alpha(d, alpha).edges()) == _edges(beta_edges, index)

    # Full union and the frequency and percentage variants: exact.
    union = _edges(sc["union"], index)
    assert set(consensus_union(dags).edges()) == union
    v = sc["variants"]
    assert set(max_frequency(dags, 2).edges()) == _edges(v["max_frequency_2"], index)
    assert set(consensus_pct(dags, 50).edges()) == _edges(v["consensus_50"], index)

    # max_edges: same budget and a subset of the union; the specific edges among
    # frequency ties are not cross-language stable (Java uses HashMap order).
    ours = set(max_edges(dags, "Max").edges())
    assert len(ours) == len(v["max_edges_Max"])
    assert ours <= union
