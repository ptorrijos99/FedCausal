"""Structural distances and edge-recovery scores between two DAGs.

* :func:`shd` compares the DAGs' directed edges, not their CPDAGs. Per node
  pair it costs 1 if only one graph has the edge and 1 if both have it in
  opposite directions. It is not invariant across a Markov equivalence class.
* :func:`smhd` compares the moral graphs, so it is invariant across the class.
  :func:`smhd_graphs` does the same for graphs already moralised.
* :func:`shd_skeleton` compares adjacencies, ignoring direction.
* :func:`edge_scores` gives precision, recall and F1: ``1.0`` when both edge
  sets are empty, ``0.0`` when only one is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fedcausal.graph.undirected import moralize

if TYPE_CHECKING:
    from fedcausal.graph.dag import DAG

__all__ = ["EdgeScores", "shd", "shd_skeleton", "smhd", "smhd_graphs", "edge_scores"]


@dataclass(frozen=True, slots=True)
class EdgeScores:
    """Precision/recall/F1 of an estimated DAG's edges against the truth.

    Attributes
    ----------
    precision : float
        Fraction of directed edges in the estimate that are correct (exact
        direction match against the true DAG).
    recall : float
        Fraction of the true DAG's directed edges recovered (exact
        direction match) by the estimate.
    f1 : float
        Harmonic mean of `precision` and `recall`.
    precision_skel : float
        Fraction of adjacent pairs in the estimate's skeleton that are
        adjacent (in either direction) in the true DAG's skeleton.
    recall_skel : float
        Fraction of the true DAG's skeleton pairs recovered (adjacency
        only, direction ignored) by the estimate.
    f1_skel : float
        Harmonic mean of `precision_skel` and `recall_skel`.
    """

    precision: float
    recall: float
    f1: float
    precision_skel: float
    recall_skel: float
    f1_skel: float


def _skeleton(adjacency: np.ndarray) -> np.ndarray:
    """Return the symmetric skeleton (adjacency ignoring direction)."""
    skeleton: np.ndarray = adjacency | adjacency.T
    return skeleton


def _prf(tp: int, n_pred: int, n_true: int) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)``, with the empty-set rules in the module docstring."""
    if n_pred == 0 and n_true == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0, 0.0, 0.0
    precision = tp / n_pred
    recall = tp / n_true
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def shd(g1: DAG, g2: DAG) -> int:
    """Return the structural Hamming distance between two DAGs.

    Parameters
    ----------
    g1 : DAG
        First DAG.
    g2 : DAG
        Second DAG, with the same `num_nodes` as `g1`.

    Returns
    -------
    int
        The number of edge additions, deletions, and reversals (each
        reversal costing 1) needed to turn `g1` into `g2`.

    Raises
    ------
    ValueError
        If `g1` and `g2` have a different number of nodes.
    """
    if g1.num_nodes != g2.num_nodes:
        raise ValueError(f"node count mismatch: {g1.num_nodes} != {g2.num_nodes}")
    a1 = g1.adjacency()
    a2 = g2.adjacency()
    differing = a1 ^ a2
    reversed_pairs = a1 & a2.T
    return int(differing.sum()) - int(reversed_pairs.sum())


def shd_skeleton(g1: DAG, g2: DAG) -> int:
    """Count node pairs whose adjacency differs, ignoring edge direction.

    Parameters
    ----------
    g1 : DAG
        First DAG.
    g2 : DAG
        Second DAG, with the same `num_nodes` as `g1`.

    Returns
    -------
    int
        The number of unordered node pairs adjacent (in either direction) in
        exactly one of `g1`'s and `g2`'s skeletons.

    Raises
    ------
    ValueError
        If `g1` and `g2` have a different number of nodes.
    """
    if g1.num_nodes != g2.num_nodes:
        raise ValueError(f"node count mismatch: {g1.num_nodes} != {g2.num_nodes}")
    return smhd_graphs(_skeleton(g1.adjacency()), _skeleton(g2.adjacency()))


def smhd_graphs(m1: np.ndarray, m2: np.ndarray) -> int:
    """Return the number of node pairs adjacent in exactly one of two undirected graphs.

    Parameters
    ----------
    m1 : np.ndarray
        Symmetric (boolean or 0/1) adjacency matrix of an undirected graph.
    m2 : np.ndarray
        Symmetric adjacency matrix of an undirected graph, same shape as
        `m1`.

    Returns
    -------
    int
        The number of unordered node pairs adjacent in exactly one of the
        two graphs.

    Raises
    ------
    ValueError
        If `m1` and `m2` have different shapes.
    """
    a = np.asarray(m1, dtype=bool)
    b = np.asarray(m2, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    return int((a ^ b).sum()) // 2


def smhd(g1: DAG, g2: DAG) -> int:
    """Return the structural moral Hamming distance between two DAGs.

    Parameters
    ----------
    g1 : DAG
        First DAG.
    g2 : DAG
        Second DAG, with the same `num_nodes` as `g1`.

    Returns
    -------
    int
        The number of unordered node pairs adjacent in exactly one of
        `g1`'s and `g2`'s moral graphs.

    Raises
    ------
    ValueError
        If `g1` and `g2` have a different number of nodes.
    """
    if g1.num_nodes != g2.num_nodes:
        raise ValueError(f"node count mismatch: {g1.num_nodes} != {g2.num_nodes}")
    return smhd_graphs(moralize(g1), moralize(g2))


def edge_scores(true: DAG, est: DAG) -> EdgeScores:
    """Return the precision, recall and F1 of `est`'s edges against `true`.

    Parameters
    ----------
    true : DAG
        The ground-truth DAG.
    est : DAG
        The estimated DAG, with the same `num_nodes` as `true`.

    Returns
    -------
    EdgeScores
        Directed-edge and skeleton scores.

    Raises
    ------
    ValueError
        If `true` and `est` have a different number of nodes.
    """
    if true.num_nodes != est.num_nodes:
        raise ValueError(f"node count mismatch: {true.num_nodes} != {est.num_nodes}")

    a_true = true.adjacency()
    a_est = est.adjacency()
    tp = int((a_true & a_est).sum())
    n_true = int(a_true.sum())
    n_est = int(a_est.sum())
    precision, recall, f1 = _prf(tp, n_est, n_true)

    skel_true = _skeleton(a_true)
    skel_est = _skeleton(a_est)
    tp_skel = int((skel_true & skel_est).sum()) // 2
    n_true_skel = int(skel_true.sum()) // 2
    n_est_skel = int(skel_est.sum()) // 2
    precision_skel, recall_skel, f1_skel = _prf(tp_skel, n_est_skel, n_true_skel)

    return EdgeScores(precision, recall, f1, precision_skel, recall_skel, f1_skel)
