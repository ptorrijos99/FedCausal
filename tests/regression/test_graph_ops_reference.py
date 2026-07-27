"""Pin the graph operations against committed reference values.

Covers :func:`fedcausal.metrics.shd`/:func:`~fedcausal.metrics.smhd`,
:func:`fedcausal.graph.treewidth_java`, :func:`fedcausal.graph.moralize` and
:func:`fedcausal.graph.meek.dag_to_cpdag`.

Every DAG is rebuilt from the golden's named edge lists, so node indices come
from the golden's ``variables`` order and any variable-order difference is
irrelevant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fedcausal.graph import DAG, moralize, treewidth_java
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.metrics import shd, smhd

GOLDEN_DIR = Path(__file__).resolve().parent
GRAPH_OPS = json.loads((GOLDEN_DIR / "graph_ops.json").read_text(encoding="ascii"))


def _build_dag(variables: list[str], named_edges: list[list[str]]) -> DAG:
    idx = {name: i for i, name in enumerate(variables)}
    return DAG.from_edges(len(variables), [(idx[u], idx[v]) for u, v in named_edges])


def _moral_name_set(dag: DAG, variables: list[str]) -> set[tuple[str, str]]:
    moral = moralize(dag)
    n = len(variables)
    pairs: set[tuple[str, str]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if moral[i, j]:
                pairs.add(tuple(sorted((variables[i], variables[j]))))  # type: ignore[arg-type]
    return pairs


def _named_pair_set(pairs: list[list[str]], *, canon: bool) -> set[tuple[str, str]]:
    if canon:
        return {tuple(sorted(p)) for p in pairs}  # type: ignore[misc]
    return {(p[0], p[1]) for p in pairs}


# --- Parametrization helpers built at collection time. ---


def _graph_cases() -> list[tuple[str, str]]:
    return [
        (net["network"], graph["name"]) for net in GRAPH_OPS["networks"] for graph in net["graphs"]
    ]


# Known cross-language divergences in the MCS-triangulation treewidth heuristic.
# Java's Utils.getMaximumCardinalityOrdering breaks maximum-cardinality ties via
# a PriorityQueue whose order is implementation-defined; fedcausal breaks ties to
# the lowest node index (deterministic, documented in graph.undirected.mcs_order).
# On NON-CHORDAL moral graphs the two orders can triangulate differently and thus
# report different maximum-clique sizes -- both are valid heuristic upper bounds,
# and Java's value is itself reachable by SOME valid tie order (verified: the
# `remove` graph triangulates to a max clique of 4 or 5; the `add` graph to 5, 6
# or 7). These are the only two families in the goldens where they differ; the
# moral graphs themselves match exactly (see test_moral_edges_match_java). Marked
# strict so the pin fails loudly if either side's tie policy ever changes.
_TREEWIDTH_XFAIL: dict[tuple[str, str], str] = {
    ("alarm", "remove"): (
        "MCS tie-order divergence: Java treewidth_java=4, fedcausal=5 "
        "(lowest-index tie-break); both are valid MCS triangulations."
    ),
    ("alarm", "add"): (
        "MCS tie-order divergence: Java treewidth_java=6, fedcausal=5 "
        "(lowest-index tie-break); both are valid MCS triangulations."
    ),
}


def _treewidth_cases() -> list[Any]:
    cases: list[Any] = []
    for network_name, graph_name in _graph_cases():
        key = (network_name, graph_name)
        if key in _TREEWIDTH_XFAIL:
            cases.append(
                pytest.param(
                    network_name,
                    graph_name,
                    marks=pytest.mark.xfail(strict=True, reason=_TREEWIDTH_XFAIL[key]),
                )
            )
        else:
            cases.append(pytest.param(network_name, graph_name))
    return cases


def _pair_cases() -> list[tuple[str, str, str]]:
    return [
        (net["network"], pair["g1"], pair["g2"])
        for net in GRAPH_OPS["networks"]
        for pair in net["pairs"]
    ]


def _network(name: str) -> dict[str, Any]:
    return next(net for net in GRAPH_OPS["networks"] if net["network"] == name)


def _graph(network: dict[str, Any], graph_name: str) -> dict[str, Any]:
    return next(g for g in network["graphs"] if g["name"] == graph_name)


@pytest.mark.regression
@pytest.mark.parametrize(("network_name", "graph_name"), _treewidth_cases())
def test_treewidth_matches_java(network_name: str, graph_name: str) -> None:
    net = _network(network_name)
    graph = _graph(net, graph_name)
    dag = _build_dag(net["variables"], graph["edges"])
    got = treewidth_java(dag)
    expected = graph["treewidth_java"]
    assert got == expected, (
        f"{network_name}/{graph_name}: Python treewidth_java {got} != Java {expected}"
    )


@pytest.mark.regression
@pytest.mark.parametrize(("network_name", "graph_name"), _graph_cases())
def test_moral_edges_match_java(network_name: str, graph_name: str) -> None:
    net = _network(network_name)
    graph = _graph(net, graph_name)
    dag = _build_dag(net["variables"], graph["edges"])
    got = _moral_name_set(dag, net["variables"])
    expected = _named_pair_set(graph["moral_edges"], canon=True)
    assert got == expected, f"{network_name}/{graph_name}: moral edges differ: {got ^ expected}"


@pytest.mark.regression
@pytest.mark.parametrize(("network_name", "graph_name"), _graph_cases())
def test_cpdag_matches_java(network_name: str, graph_name: str) -> None:
    net = _network(network_name)
    variables = net["variables"]
    graph = _graph(net, graph_name)
    dag = _build_dag(variables, graph["edges"])
    pdag = dag_to_cpdag(dag)

    got_directed = {(variables[u], variables[v]) for u, v in pdag.directed_edges()}
    got_undirected = {
        tuple(sorted((variables[u], variables[v]))) for u, v in pdag.undirected_edges()
    }
    expected_directed = _named_pair_set(graph["cpdag_directed"], canon=False)
    expected_undirected = _named_pair_set(graph["cpdag_undirected"], canon=True)

    assert got_directed == expected_directed, (
        f"{network_name}/{graph_name}: CPDAG directed edges differ: "
        f"{got_directed ^ expected_directed}"
    )
    assert got_undirected == expected_undirected, (
        f"{network_name}/{graph_name}: CPDAG undirected edges differ: "
        f"{got_undirected ^ expected_undirected}"
    )


@pytest.mark.regression
@pytest.mark.parametrize(("network_name", "g1_name", "g2_name"), _pair_cases())
def test_shd_smhd_match_java(network_name: str, g1_name: str, g2_name: str) -> None:
    net = _network(network_name)
    variables = net["variables"]
    g1 = _build_dag(variables, _graph(net, g1_name)["edges"])
    g2 = _build_dag(variables, _graph(net, g2_name)["edges"])
    pair = next(p for p in net["pairs"] if p["g1"] == g1_name and p["g2"] == g2_name)
    assert shd(g1, g2) == pair["shd"], (
        f"{network_name} {g1_name}->{g2_name}: Python SHD {shd(g1, g2)} != Java {pair['shd']}"
    )
    assert smhd(g1, g2) == pair["smhd"], (
        f"{network_name} {g1_name}->{g2_name}: Python SMHD {smhd(g1, g2)} != Java {pair['smhd']}"
    )
