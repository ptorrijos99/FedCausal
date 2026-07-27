"""Structural distance and edge-recovery metrics for causal structure learning.

The counting rules are pinned to the reference implementation, and are worth
stating because several differ from what the names suggest.

**SHD** (:func:`shd`) compares two DAGs' exact directed edge sets -- *not* their
CPDAGs, despite the usual convention in the literature. Per unordered pair:
0 when both graphs agree, 1 when an edge is in exactly one of them, and 1 (not
2) when both have the edge with opposite orientations. Because the comparison
happens at the DAG level, a reversed compelled edge and a reversed reversible
edge are indistinguishable -- both cost 1.

**SMHD** (:func:`smhd`) moralises both DAGs and takes the symmetric difference
of the resulting undirected skeletons. Moral graphs have no orientation to
reverse, so every pair adjacent in exactly one of them costs 1. It is invariant
across a Markov equivalence class, which SHD is not.
:func:`smhd_graphs` is the same count over two graphs already moralised.

**Skeleton SHD** (:func:`shd_skeleton`) counts undirected adjacency differences.

**Precision, recall, F1** (:func:`edge_scores`) have no counterpart in the
reference implementation, which never divides. For both the directed and the
skeleton group, all three are ``1.0`` when both edge sets are empty (a trivial
exact match on "no edges"), ``0.0`` when exactly one side is empty, and the
standard formulas otherwise.
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
    """Compute precision/recall/F1 under this module's zero-division policy.

    Parameters
    ----------
    tp : int
        Number of true positives.
    n_pred : int
        Number of predicted (estimated) positives -- the precision
        denominator.
    n_true : int
        Number of actual (true) positives -- the recall denominator.

    Returns
    -------
    tuple[float, float, float]
        `(precision, recall, f1)`. Both empty (`n_pred == n_true == 0`)
        yields `(1.0, 1.0, 1.0)`; exactly one empty yields
        `(0.0, 0.0, 0.0)`; otherwise the standard formulas, with `f1`
        guarded against the residual `precision + recall == 0` case (both
        nonempty but zero overlap).
    """
    if n_pred == 0 and n_true == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0, 0.0, 0.0
    precision = tp / n_pred
    recall = tp / n_true
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def shd(g1: DAG, g2: DAG) -> int:
    """Structural Hamming distance between two DAGs (port of ``Utils.SHD``).

    Counts, per unordered node pair, 0 if the two DAGs agree (same directed
    edge, or neither adjacent), 1 if the pair is an edge in exactly one
    graph, and 1 (not 2) if both graphs have an edge between the pair but in
    opposite directions. See the module docstring for the reference implementation derivation.

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
    """Structural moralized Hamming distance between two adjacency matrices.

    Port of ``Utils.SMHDwithoutMoralize``: counts the symmetric difference
    of two undirected graphs' edge sets, given as symmetric adjacency
    matrices (typically already-moralized graphs). No moralization is
    performed here -- see :func:`smhd` for the moralizing variant.

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
    """Structural moralized Hamming distance between two DAGs.

    Port of ``Utils.SMHD``: moralizes both DAGs (via
    :func:`fedcausal.graph.undirected.moralize`), then counts the symmetric
    difference of the two moral skeletons' edge sets (:func:`smhd_graphs`).

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
    """Precision/recall/F1 of `est`'s edges against `true`.

    Parameters
    ----------
    true : DAG
        The ground-truth DAG.
    est : DAG
        The estimated DAG, with the same `num_nodes` as `true`.

    Returns
    -------
    EdgeScores
        Directed-edge and skeleton precision/recall/F1. See the module
        docstring for the zero-division policy applied when one or both
        edge sets are empty.

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
